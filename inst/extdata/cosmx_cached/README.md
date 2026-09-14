# CosMx NSCLC example results

Source: public CosMx SMI NSCLC data from He et al. (2022),
<https://doi.org/10.1038/s41587-022-01483-z>.

The example contains five patients, tumor as query, and 17 target cell types.
Per-patient Poisson models use the nearest tumor cell (`k_neighbors = 1`)
and a 200 micrometer distance cap, retaining farther cells at the cap.
Cross-patient results use the strict sign gate, Fisher combination, and BH
adjustment within each target cell type. These tables include zero Wald
p-values in sign checks and floor them at 1e-15 for Fisher combination.

- `all_genes_results.csv`: 6,806 gene-by-target combinations, 599 significant
  at adjusted p-value < 0.05, involving 309 unique genes.
- `contamination_candidates.csv`: genes significant in at least four target
  cell types. These are inspection candidates, not confirmed contamination.
- `per_celltype/fibroblast/coef_per_sample.csv`: per-patient fibroblast fits.
- `fgsea_all_celltypes.csv`: Hallmark enrichment from median coefficients,
  excluding broad-class genes (threshold 4), seed 42. No additional
  query-marker filter was applied to this table.
- `k_diagnostics_cosmx.csv`: uncapped distance summaries by patient, target
  type and k. The vignette averages patient means with equal patient weight.
- `lr/fibroblast/`: ligand-receptor prioritization and supporting tables,
  retaining induced receptors and excluding broad-class receptors plus
  EPCAM, KRT8, KRT18, KRT19, KRT7 and KRT17.

The vignette uses coefficients, significance, specificity, enrichment and
ligand-receptor scores. Saved `decay_pattern` labels were not refitted during
the significance refresh; `not_recomputed` marks newly significant genes
without an updated shape classification. Per-cell QC files and raw counts
are not included in this compact cache.
