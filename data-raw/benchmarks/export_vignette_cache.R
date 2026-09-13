# Export completed synthetic runs for the benchmarks vignette.
# Usage: Rscript data-raw/benchmarks/export_vignette_cache.R CACHE_DIR MATCHED_DIR
# CACHE_DIR contains assembled bench_*_results.rds files with corrected metrics.
# MATCHED_DIR contains n3, n5, and n10 subdirectories from bench_power_matched.R.
suppressPackageStartupMessages(library(data.table))
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
read_runs <- function(kind) {
  readRDS(file.path(args[1], paste0("bench_", kind, "_results.rds")))$per_run
}
null_strict <- read_runs("null_hires")
null_relaxed <- read_runs("null_hires_relaxed")
null_nb <- read_runs("null_nb")
matched <- rbindlist(lapply(c(3L, 5L, 10L), function(n) {
  readRDS(file.path(args[2], paste0("n", n), "results.rds"))$per_run
}))
runtime <- read_runs("runtime")
stopifnot(nrow(null_strict) == 150L, nrow(null_relaxed) == 150L,
          nrow(null_nb) == 600L, nrow(matched) == 1080L,
          nrow(runtime) == 9L)
stopifnot(!anyDuplicated(matched[, .(beta, n_samples, iteration, gate)]),
          all(matched$tp + matched$fn == 5L),
          all(matched$n_grad_filtered == 0L))
for (d in list(null_strict, null_relaxed, null_nb)) {
  stopifnot(all(d$n_genes_tested > 0L),
            all(d$n_sig_fdr >= 0L & d$n_sig_fdr <= d$n_genes_tested))
}
# Include only synthetic per-run tables, with no private biological caches.
bundle <- lapply(list(null_strict = null_strict, null_relaxed = null_relaxed,
  null_nb = null_nb, power_matched = matched,
  runtime = runtime), as.data.frame)
bundle$provenance <- readRDS(file.path(args[1],
  "bench_runtime_results.rds"))$provenance
bundle$provenance$cache_schema <- 2L
saveRDS(bundle, "inst/extdata/benchmarks_current_results.rds", compress = "xz")
cat("Exported validated synthetic benchmark tables.\n")
