"""GPU permutation input contracts, using synthetic data only."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
import run_permutation_gpu as rp


class InputTests(unittest.TestCase):
    def test_package_median_takes_precedence_and_preserves_untested(self):
        meta = pd.DataFrame({'gene': ['g', 'untested'], 'median_coef': [-.001, np.nan],
                             'combined_coef': [-.004, -.2]})
        medians = rp.load_observed_medians(meta, 'unused')
        self.assertEqual(medians.iloc[0], -.001)
        self.assertTrue(np.isnan(medians.iloc[1]))

    def test_legacy_median_uses_valid_replicates_and_gene_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({
                'gene': ['g'] * 6 + ['insufficient'],
                'sample_id': list('abcdef') + ['a'],
                'coef': [-.001, -.001, -.01, np.inf, -.2, -.3, -.4],
                'se': [.001, .001, .001, .001, 0, np.nan, .01],
                'pval': [0, .2, .1, .2, .2, .2, .2]
            }).to_csv(Path(directory) / 'coef_per_sample.csv', index=False)
            meta = pd.DataFrame({'gene': ['missing', 'g', 'insufficient'],
                                 'combined_coef': [-1, -.004, -.4]})
            medians = rp.load_observed_medians(meta, directory)
            self.assertTrue(np.isnan(medians.iloc[0]))
            self.assertEqual(medians.iloc[1], -.001)
            self.assertTrue(np.isnan(medians.iloc[2]))

    def test_weighted_summary_alone_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'combined_coef is not a median'):
                rp.load_observed_medians(pd.DataFrame({'gene': ['g'], 'combined_coef': [-.004]}), directory)

    def test_duplicate_sample_fits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({'gene': ['g', 'g'], 'sample_id': ['a', 'a'],
                          'coef': [-.001, -.01], 'se': [.001, .001]}).to_csv(
                              Path(directory) / 'coef_per_sample.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'one row per gene and sample'):
                rp.load_observed_medians(pd.DataFrame({'gene': ['g']}), directory)

    def test_raw_count_selection_dense_and_sparse(self):
        counts = np.array([[0, 10], [20, 2]], dtype=float)
        for convert in (np.asarray, sparse.csr_matrix, sparse.csc_matrix):
            raw = convert(counts)
            for x in (convert(np.log1p(counts)), convert(counts * 2)):
                selected, source = rp.select_count_matrix(SimpleNamespace(X=x, layers={'counts': raw}))
                self.assertIs(selected, raw)
                self.assertEqual(source, "adata.layers['counts']")
            selected, source = rp.select_count_matrix(SimpleNamespace(X=raw, layers={}))
            self.assertIs(selected, raw)
            self.assertEqual(source, 'adata.X')

    def test_invalid_counts_anywhere_in_matrix_are_rejected(self):
        for bad_value in (.5, -1, np.nan, np.inf):
            for use_layer in (False, True):
                for convert in (np.asarray, sparse.csr_matrix):
                    bad = np.zeros((150, 2))
                    bad[-1, -1] = bad_value  # Beyond the old first-100-row check.
                    adata = SimpleNamespace(X=convert(np.ones_like(bad)) if use_layer else convert(bad),
                                            layers={'counts': convert(bad)} if use_layer else {})
                    with self.subTest(value=bad_value, layer=use_layer, convert=convert):
                        with self.assertRaisesRegex(ValueError, 'finite, nonnegative integer counts'):
                            rp.select_count_matrix(adata)

    def test_main_passes_matching_counts_offsets_and_medians(self):
        import anndata as ad
        rng = np.random.default_rng(11)
        counts = rng.poisson(4, (180, 3)).astype(np.float32)
        normalized = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000)
        obs = pd.DataFrame({'sample_id': np.repeat(['A', 'B', 'C'], 60),
                            'cell_type': np.tile(['Query'] * 5 + ['Other'] * 15 + ['LEC'] * 40, 3),
                            'condition': ['keep'] * 180}, index=[f'cell{i}' for i in range(180)])
        obs.iloc[[0, 25, 60, 86, 122, 150], obs.columns.get_loc('condition')] = 'drop'
        expected = np.flatnonzero((obs.cell_type == 'LEC') & (obs.condition == 'keep'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'spatial_analysis_Query' / 'test_v2' / 'per_celltype' / 'LEC'
            output.mkdir(parents=True)
            input_path = root / 'synthetic.h5ad'
            environment = dict(QUERY_CELLTYPE='Query', CELLTYPE_COLUMN='cell_type',
                               ANALYSIS_NAME='test_v2', K_NEIGHBORS='1',
                               PERMUTATION_POOL='non_target', X_COLUMN='', Y_COLUMN='')
            for convert in (np.asarray, sparse.csr_matrix):
                for legacy in (False, True):
                    with self.subTest(convert=convert, legacy=legacy):
                        adata = ad.AnnData(convert(normalized), obs=obs.copy(),
                                           var=pd.DataFrame(index=['g', 'h', 'unused']))
                        adata.layers['counts'] = convert(counts)
                        adata.obsm['spatial'] = rng.uniform(0, 100, (180, 2))
                        adata.write_h5ad(input_path)
                        meta = pd.DataFrame({'gene': ['g', 'h'], 'combined_coef': [-.004, -.002]})
                        if legacy:
                            pd.DataFrame({'gene': ['g'] * 3 + ['h'] * 3,
                                          'sample_id': list('ABC') * 2,
                                          'coef': [-.001, -.001, -.01, -.003, -.003, 0],
                                          'se': [.001] * 6}).to_csv(output / 'coef_per_sample.csv', index=False)
                        else:
                            meta['median_coef'] = [-.001, -.003]
                        meta.to_csv(output / 'meta_analysis_results.csv', index=False)
                        # The strongest median is h; the strongest legacy mean is g.
                        with patch.dict(os.environ, environment), patch.multiple(
                                rp, OUTPUT_DIR=str(root), ADATA_PATH=input_path,
                                SAMPLE_COL='sample_id', CONDITION_COL='condition', CONDITION_VAL='keep',
                                PERM_TOP_N=1, PERM_PRIORITY_GENES=[]), patch.object(
                                sys, 'argv', ['run_permutation_gpu.py', '--celltype', 'LEC', '--n-perms', '12']), \
                                patch.object(rp, 'run_permutation_test_gpu', return_value=.5) as permutation, \
                                contextlib.redirect_stdout(io.StringIO()):
                            rp.main()
                        permutation.assert_called_once()
                        args = permutation.call_args.kwargs
                        self.assertEqual(args['observed_coef'], -.003)
                        np.testing.assert_array_equal(args['gene_data'], counts[expected, 1])
                        np.testing.assert_allclose(args['log_total_counts_target'], np.log(counts[expected].sum(axis=1)))
                        np.testing.assert_array_equal(args['sample_ids_target'], obs.sample_id.iloc[expected])
                        result = pd.read_csv(output / 'permutation_pvals.csv')
                        self.assertEqual(result.loc[result.gene == 'h', 'perm_pval'].iloc[0], .5)


if __name__ == '__main__':
    unittest.main()
