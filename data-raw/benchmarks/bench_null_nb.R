# ============================================================================
# NB (overdispersed) null calibration benchmark
# ============================================================================
# Addresses the "circular calibration" review point: the Poisson null benchmark
# (bench_null.R) simulates under the exact model RIPPLE fits, so it cannot speak
# to robustness when the Poisson assumption is violated. Here we generate
# all-null data with negative-binomial overdispersion (variance = phi * mean,
# phi > 1) and check whether RIPPLE's empirical false-positive rate stays
# controlled even though it still fits a Poisson GLM (there is no NB fallback).
#
# Design: dispersion phi in {1, 1.5, 2, 3} x sample sizes {3, 5, 10}
#         x n_iterations. phi = 1 reproduces the Poisson null as a sanity check.
#
# Run with (PowerShell; full run_ripple segfaults under Git Bash Rscript):
#   Rscript data-raw/benchmarks/bench_null_nb.R
#
# Output:
#   data-raw/benchmarks/results/bench_null_nb_results.rds
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  devtools::load_all(quiet = TRUE)
})
source("data-raw/benchmarks/benchmark_helpers.R")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Dispersions to run. RIPPLE_NB_DISP selects a subset so the grid can be
# split across SLURM array tasks; unset runs the full grid as before.
dispersions  <- c(1, 1.5, 2, 3)
if (nzchar(Sys.getenv("RIPPLE_NB_DISP"))) {
  dispersions <- as.numeric(strsplit(Sys.getenv("RIPPLE_NB_DISP"), ",")[[1]])
  cat("RIPPLE_NB_DISP set; running dispersions:",
      paste(dispersions, collapse = ", "), "\n")
}
sample_sizes <- c(3, 5, 10)
n_iterations <- 50
n_background <- 50
base_seed    <- 4242

total_runs <- length(dispersions) * length(sample_sizes) * n_iterations
cat("=== NB Null Calibration Benchmark ===\n")
cat(sprintf("  %d dispersions x %d sample sizes x %d iters = %d runs\n",
            length(dispersions), length(sample_sizes), n_iterations, total_runs))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
all_results <- list()
counter <- 0

for (phi in dispersions) {
  for (n_samp in sample_sizes) {
    for (iter in seq_len(n_iterations)) {
      counter <- counter + 1
      seed <- base_seed * 1000 + round(phi * 100) * 1000 + n_samp * 100 + iter

      if (counter %% 25 == 1 || counter == total_runs) {
        cat(sprintf("[%d/%d] phi=%.1f N=%d iter=%d\n",
                    counter, total_runs, phi, n_samp, iter))
      }

      spe <- generate_benchmark_data(
        n_samples      = n_samp,
        n_gradient_neg = 0,
        n_gradient_pos = 0,
        n_background   = n_background,
        dispersion     = phi,
        seed           = seed
      )

      res <- tryCatch(run_ripple_quiet(spe), error = function(e) {
        warning(sprintf("phi=%.1f N=%d iter=%d failed: %s", phi, n_samp, iter, e$message))
        NULL
      })
      if (is.null(res)) next

      tcell <- res[cell_type == "T_cell"]
      n_tested   <- nrow(tcell)
      n_sig_fdr  <- sum(tcell$fisher_fdr < 0.05, na.rm = TRUE)
      n_sig_pval <- sum(tcell$fisher_pval < 0.05, na.rm = TRUE)

      all_results[[counter]] <- data.table(
        dispersion       = phi,
        n_samples        = n_samp,
        iteration        = iter,
        n_genes_tested   = n_tested,
        n_sig_fdr        = n_sig_fdr,
        n_sig_pval       = n_sig_pval,
        empirical_fpr    = n_sig_fdr / max(n_tested, 1),
        any_fp           = as.integer(n_sig_fdr > 0),
        realized_disp    = stats::median(tcell$median_dispersion, na.rm = TRUE)
      )
    }
  }
}

results_dt <- rbindlist(all_results)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n=== Results (all-null data; every call is a false positive) ===\n")
summary_dt <- results_dt[, .(
  n_runs         = .N,
  mean_realized_disp = round(mean(realized_disp, na.rm = TRUE), 2),
  total_sig      = sum(n_sig_fdr),
  total_tested   = sum(n_genes_tested),
  pooled_fpr     = sum(n_sig_fdr) / sum(n_genes_tested),
  fwer           = mean(any_fp)
), by = .(dispersion, n_samples)]
setorder(summary_dt, dispersion, n_samples)
print(summary_dt)

# Set RIPPLE_BENCH_DIR to choose a separate output directory.
bench_dir <- Sys.getenv("RIPPLE_BENCH_DIR",
                        unset = "data-raw/benchmarks/results")
dir.create(bench_dir, recursive = TRUE, showWarnings = FALSE)
out_name <- Sys.getenv("RIPPLE_NB_OUT",
                       unset = "bench_null_nb_results.rds")
out_path <- file.path(bench_dir, out_name)
saveRDS(list(per_run = results_dt, summary = summary_dt), file = out_path)
cat("\nSaved:", out_path, "\n")
