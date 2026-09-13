# ============================================================================
# High-resolution FDR null calibration benchmark
# ============================================================================
# Same design as bench_null.R but with 500 background genes and 100
# iterations per sample size, giving 150,000 null tests per N. This
# resolves the floor artefact in the original 50-gene x 50-iter benchmark
# (where zero false positives at N = 5 and N = 10 simply meant we hit
# the lower bound of detectable FDR).
#
# Run with:
#   Rscript data-raw/benchmarks/bench_null_hires.R
#
# Output:
#   data-raw/benchmarks/results/bench_null_hires_results.rds
#   inst/extdata/bench_null_hires_results.rds (package-bundled for Rmd)
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  devtools::load_all(quiet = TRUE)
})
source("data-raw/benchmarks/benchmark_helpers.R")

n_iterations <- 50
sample_sizes <- c(3, 5, 10)
n_background <- 500
base_seed <- 2026

cat("=== FDR Null Calibration Benchmark (HIRES) ===\n")
cat(sprintf(
  "  %d iterations x %d sample sizes = %d runs\n",
  n_iterations, length(sample_sizes),
  n_iterations * length(sample_sizes)
))
cat(sprintf("  %d background genes/run = %d total null tests per N\n",
            n_background, n_iterations * n_background))

all_results <- list()
counter <- 0
total_runs <- n_iterations * length(sample_sizes)
t_start <- Sys.time()

for (n_samp in sample_sizes) {
  for (iter in seq_len(n_iterations)) {
    counter <- counter + 1
    seed <- base_seed * 1000 + n_samp * 100 + iter

    if (counter %% 5 == 1 || counter == total_runs) {
      elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
      rate <- counter / max(elapsed, 1)
      eta <- (total_runs - counter) / max(rate, 1e-6)
      cat(sprintf(
        "[%d/%d] N=%d iter=%d  elapsed=%.1fm  eta=%.1fm\n",
        counter, total_runs, n_samp, iter,
        elapsed / 60, eta / 60
      ))
    }

    spe <- generate_benchmark_data(
      n_samples      = n_samp,
      n_gradient_neg = 0,
      n_gradient_pos = 0,
      n_background   = n_background,
      seed           = seed
    )

    res <- tryCatch(
      run_ripple_quiet(spe),
      error = function(e) {
        warning(sprintf("N=%d iter=%d failed: %s", n_samp, iter, e$message))
        NULL
      }
    )

    if (!is.null(res)) {
      tcell_res <- res[cell_type == "T_cell"]
      n_tested <- nrow(tcell_res)
      n_sig_fdr <- sum(tcell_res$fisher_fdr < 0.05, na.rm = TRUE)
      n_sig_pval <- sum(tcell_res$fisher_pval < 0.05, na.rm = TRUE)

      all_results[[counter]] <- data.table(
        n_samples = n_samp,
        iteration = iter,
        seed = seed,
        n_genes_tested = n_tested,
        n_sig_fdr = n_sig_fdr,
        n_sig_pval = n_sig_pval,
        empirical_fdr = n_sig_fdr / max(n_tested, 1),
        empirical_fwer = as.integer(n_sig_fdr > 0)
      )
    }
  }
}

results_dt <- rbindlist(all_results)

cat("\n=== Results ===\n")
summary_dt <- results_dt[, .(
  n_runs = .N,
  mean_genes_tested = mean(n_genes_tested),
  mean_fdr = mean(empirical_fdr),
  sd_fdr = sd(empirical_fdr),
  max_fdr = max(empirical_fdr),
  mean_fwer = mean(empirical_fwer),
  total_sig_fdr = sum(n_sig_fdr),
  total_sig_pval = sum(n_sig_pval),
  total_tested = sum(n_genes_tested)
), by = n_samples]

summary_dt[, pooled_fdr := total_sig_fdr / total_tested]
summary_dt[, pooled_pval_rate := total_sig_pval / total_tested]
print(summary_dt)

# Set RIPPLE_BENCH_DIR to choose a separate output directory.
bench_dir <- Sys.getenv("RIPPLE_BENCH_DIR",
                        unset = "data-raw/benchmarks/results")
dir.create(bench_dir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(bench_dir, "bench_null_hires_results.rds")
saveRDS(list(per_run = results_dt, summary = summary_dt), file = out_path)
cat("\nSaved:", out_path, "\n")

# Update the bundled cache only when using the default output directory.
extdata_path <- "inst/extdata/bench_null_hires_results.rds"
if (!nzchar(Sys.getenv("RIPPLE_BENCH_DIR"))) {
  file.copy(out_path, extdata_path, overwrite = TRUE)
} else {
  cat("RIPPLE_BENCH_DIR is set; leaving", extdata_path, "untouched\n")
}
cat("Copied to:", extdata_path, "\n")

elapsed_total <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
cat(sprintf("\nTotal wall-clock: %.1f min\n", elapsed_total))
