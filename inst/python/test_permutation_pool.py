"""Regression tests for pseudo-query identity exclusion (CPU or CUDA).

Run with: python -m unittest discover -s inst/python -p test_permutation_pool.py
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_permutation_gpu as rp


class RecordingRng:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.draws = []

    def choice(self, candidates, size, replace):
        chosen = self.rng.choice(candidates, size=size, replace=replace)
        self.draws.append(chosen.copy())
        return chosen


class PoolTests(unittest.TestCase):
    def test_excludes_all_target_identities_in_interleaved_samples(self):
        samples = np.array(['A', 'B'] * 6)
        # Several excluded cells need not be present in a gene's fitted subset.
        mask = np.array([True, False, False, True, True, True,
                         False, False, False, False, True, True])
        pools = rp.prepare_permutation_pool(samples, {'A': 2, 'B': 2}, mask)
        np.testing.assert_array_equal(pools['A'], [2, 6, 8])
        np.testing.assert_array_equal(pools['B'], [1, 7, 9])

    def test_requires_an_aligned_boolean_mask(self):
        for mask in (None, [True], [0, 1, 0], [True, None, False],
                     np.array([[True, False, False]])):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, 'target_mask_all'):
                rp.prepare_permutation_pool(np.array(['A'] * 3), {'A': 1}, mask)

    def test_insufficient_pool_errors_per_sample(self):
        with self.assertRaisesRegex(ValueError, "sample 'A'.*query count \\(2\\)"):
            rp.prepare_permutation_pool(np.array(['A', 'A', 'B', 'B', 'B']),
                                         {'A': 2, 'B': 1},
                                         np.array([True, False, False, False, False]))

    def test_legacy_candidates_and_seeded_draws_are_unchanged(self):
        samples = np.array(['B', 'A', 'A', 'B', 'C', 'B'] * 5)
        query_counts = {'A': 4, 'B': 5, 'C': 2}
        pools = rp.prepare_permutation_pool(samples, query_counts, permutation_pool='all')
        new_rng, old_rng = np.random.default_rng(21), np.random.default_rng(21)
        for _ in range(15):
            for sample, count in query_counts.items():
                np.testing.assert_array_equal(
                    new_rng.choice(pools[sample], size=count, replace=False),
                    old_rng.choice(np.where(samples == sample)[0], size=count, replace=False))

    def test_invalid_pool_errors(self):
        with self.assertRaisesRegex(ValueError, 'permutation_pool'):
            rp.prepare_permutation_pool(np.array(['A']), {'A': 1}, permutation_pool='typo')

    def test_actual_permutation_loop_preserves_counts_samples_k_and_cap(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_rng = np.random.default_rng(8)
        # Each sample has 15 non-targets and 45 targets, with only 40 targets fitted.
        samples = np.repeat(['A', 'B'], 60)
        all_target = np.tile(np.arange(60) >= 15, 2)
        fitted = np.tile((np.arange(60) >= 15) & (np.arange(60) < 55), 2)
        xy = data_rng.uniform(0, 180, (120, 2)).astype(np.float32)
        xy[[54, 114]] = [1000, 1000]  # exercises the distance cap
        target_samples = samples[fitted]
        counts = data_rng.poisson(3, fitted.sum()).astype(float)
        for pool in ('non_target', 'all'):
            for k in (1, 3):
                for poisson in (False, True):
                    with self.subTest(pool=pool, k=k, poisson=poisson):
                        rng = RecordingRng(42)
                        fitted_distances = []
                        fit = rp.fit_poisson_sample if poisson else rp.fit_logistic_sample

                        def record_fit(y, distance, *args):
                            fitted_distances.append(distance.copy())
                            return fit(y, distance, *args)

                        name = 'fit_poisson_sample' if poisson else 'fit_logistic_sample'
                        with patch.object(rp, name, side_effect=record_fit):
                            pval = rp.run_permutation_test_gpu(
                                gene_data=counts if poisson else (counts > 2).astype(float),
                                target_coords_gpu=torch.tensor(xy[fitted], device=device),
                                all_coords_gpu=torch.tensor(xy, device=device),
                                sample_ids_target=target_samples, sample_ids_all=samples,
                                query_per_sample={'A': 4, 'B': 4}, observed_coef=0.0,
                                n_perms=12, device=device, rng=rng, k_neighbors=k,
                                use_poisson=poisson,
                                log_total_counts_target=np.full(fitted.sum(), np.log(500)),
                                target_mask_all=all_target, permutation_pool=pool)
                        self.assertEqual(pval, 1.0)
                        self.assertEqual(len(rng.draws), 24)
                        self.assertEqual(len(fitted_distances), 24)
                        for i, (draw, distances) in enumerate(zip(rng.draws, fitted_distances)):
                            sample = ['A', 'B'][i % 2]
                            self.assertEqual(len(np.unique(draw)), 4)
                            self.assertTrue(np.all(samples[draw] == sample))
                            if pool == 'non_target':
                                self.assertFalse(np.any(all_target[draw]))
                            target_xy = xy[fitted & (samples == sample)].astype(float)
                            reference = np.linalg.norm(target_xy[:, None] - xy[draw][None], axis=2)
                            reference.sort(axis=1)
                            reference = np.minimum(reference[:, :k].mean(axis=1), rp.MAX_DISTANCE_UM)
                            np.testing.assert_allclose(distances, reference, atol=0.05, rtol=1e-5)
                            if pool == 'non_target':
                                self.assertEqual(distances[-1], rp.MAX_DISTANCE_UM)

    def test_cli_outputs_pool_for_tested_and_untested_genes(self):
        import anndata as ad
        import pandas as pd

        rng = np.random.default_rng(20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adata = ad.AnnData(
                rng.poisson(3, (120, 2)).astype(np.float32),
                obs=pd.DataFrame({'sample_id': np.repeat(['A', 'B'], 60),
                                  'cell_type': np.tile(['Query'] * 5 + ['Other'] * 15
                                                       + ['LEC'] * 40, 2)},
                                 index=[f'cell{i}' for i in range(120)]),
                var=pd.DataFrame(index=['test_gene', 'untested_gene']))
            adata.obsm['spatial'] = rng.uniform(0, 150, (120, 2))
            input_path = root / 'synthetic.h5ad'
            adata.write_h5ad(input_path)
            output = root / 'spatial_analysis_Query' / 'test_v2' / 'per_celltype' / 'LEC'
            output.mkdir(parents=True)
            pd.DataFrame({'gene': ['test_gene', 'untested_gene'],
                          'median_coef': [0.0, np.nan]}).to_csv(
                              output / 'meta_analysis_results.csv', index=False)
            env = dict(os.environ, ADATA_PATH=str(input_path), OUTPUT_DIR=str(root),
                       QUERY_CELLTYPE='Query', CELLTYPE_COLUMN='cell_type',
                       SAMPLE_COLUMN='sample_id', CONDITION_COLUMN='', CONDITION_VALUE='',
                       X_COLUMN='', Y_COLUMN='', ANALYSIS_NAME='test_v2')
            for setting, extra_args, expected in (
                    (None, [], 'non_target'), ('all', [], 'all'),
                    ('all', ['--permutation-pool', 'non_target'], 'non_target')):
                with self.subTest(setting=setting, extra_args=extra_args):
                    env.pop('PERMUTATION_POOL', None)
                    if setting is not None:
                        env['PERMUTATION_POOL'] = setting
                    result = subprocess.run(
                        [sys.executable, rp.__file__, '--celltype', 'LEC',
                         '--n-perms', '12', *extra_args], env=env,
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    table = pd.read_csv(output / 'permutation_pvals.csv').set_index('gene')
                    self.assertEqual(set(table['permutation_pool']), {expected})
                    self.assertEqual(table.loc['test_gene', 'perm_pval'], 1.0)
                    self.assertTrue(np.isnan(table.loc['untested_gene', 'perm_pval']))

    def test_cli_help_and_invalid_environment(self):
        script = str(Path(rp.__file__))
        help_result = subprocess.run([sys.executable, script, '--help'],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('--permutation-pool {non_target,all}', help_result.stdout)
        env = dict(os.environ, PERMUTATION_POOL='invalid')
        result = subprocess.run([sys.executable, script, '--celltype', 'LEC'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("PERMUTATION_POOL must be", result.stderr)


if __name__ == '__main__':
    unittest.main()
