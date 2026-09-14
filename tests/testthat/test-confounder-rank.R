test_that("controlled fits reject unidentifiable distances after cell filtering", {
  set.seed(93)
  n <- 100
  query <- runif(n, 0, 200)
  counts <- rpois(n, 20)
  total <- rep(500, n)
  controls <- list(query, 20 + 2 * query, 400 - query, rep(200, n),
                   query + rnorm(n, sd = 1e-8))
  for (control in controls) {
    expect_null(fit_poisson_controlled(counts, query, control, total))
    check <- ripple:::.fit_poisson_controlled_result(counts, query, control, total)
    expect_equal(check$fit_status, "rank_deficient")
  }
  expect_null(fit_poisson_controlled(counts, rep(200, n), query, total))
  expect_null(fit_poisson_controlled(counts, pmin(query + 300, 200),
                                    pmin(query + 400, 200), total))

  # The only cells distinguishing the predictors cannot contribute if their
  # count or offset is invalid. Check rank after removing those cells.
  control <- query
  control[1:3] <- c(400, 0, 100)
  counts[1] <- NA
  total[2:3] <- c(0, Inf)
  check <- ripple:::.fit_poisson_controlled_result(counts, query, control, total)
  expect_equal(check$fit_status, "rank_deficient")
  expect_equal(check$n_cells, 97)
})

test_that("estimable correlated controlled fits retain native GLM estimates", {
  set.seed(194)
  query <- runif(200, 0, 200)
  control <- query + rnorm(200, sd = 12)
  total <- rep(500, 200)
  counts <- rpois(200, 30 * exp(-0.005 * query + 0.002 * control))
  expect_gt(cor(query, control), 0.8)
  native <- glm(counts ~ query + control + offset(log(total)), family = poisson())
  expected <- summary(native)$coefficients["query", ]
  observed <- fit_poisson_controlled(counts, query, control, total)
  expect_named(observed, c("beta", "se", "pval", "dispersion", "n_cells"))
  expect_equal(observed$beta, unname(expected[1]), tolerance = 1e-12)
  expect_equal(observed$se, unname(expected[2]), tolerance = 1e-12)
  expect_equal(observed$pval, unname(expected[4]), tolerance = 1e-12)
  expect_equal(ripple:::.fit_poisson_controlled_result(
    counts, query, control, total)$fit_status, "ok")
  # Rank decisions should not depend on micrometers versus millimeters.
  scaled <- fit_poisson_controlled(counts, query / 1000, control / 1000, total)
  expect_equal(scaled$beta / 1000, observed$beta, tolerance = 1e-10)
})

test_that("Stage 4 reports rank-deficient fits without a biological classification", {
  skip_if_not_installed("SpatialExperiment")
  set.seed(318)
  query <- cbind(runif(30, 0, 20), runif(30, 0, 20))
  target <- cbind(runif(60, 20, 300), runif(60, 20, 300))
  xy <- rbind(query, query, target)
  xy <- rbind(xy, xy + 1000)
  ct <- rep(c(rep("query", 30), rep("control", 30), rep("target", 60)), 2)
  counts <- rbind(signal = rpois(nrow(xy), 20), background = 0)
  counts[2, ] <- 500 - counts[1, ]
  colnames(counts) <- paste0("cell", seq_len(ncol(counts)))
  spe <- SpatialExperiment::SpatialExperiment(
    assays = list(counts = methods::as(counts, "CsparseMatrix")),
    colData = S4Vectors::DataFrame(cell_type = ct,
      sample_id = rep(c("s1", "s2"), each = 120), row.names = colnames(counts)),
    spatialCoords = xy
  )
  results_dir <- tempfile("rank_stage1_")
  output_dir <- tempfile("rank_stage4_")
  dir.create(file.path(results_dir, "summary"), recursive = TRUE)
  on.exit(unlink(c(results_dir, output_dir), recursive = TRUE))
  data.table::fwrite(data.table::data.table(
    gene = "signal", cell_type = "target", median_coef = -0.01, fisher_fdr = 0.001
  ), file.path(results_dir, "summary", "all_genes_results.csv"))
  messages <- character()
  result <- withCallingHandlers(run_ripple_confounder(
    spe, results_dir, "query", "cell_type", "control", output_dir = output_dir,
    target_celltypes = "target", verbose = FALSE
  ), warning = function(w) {
    messages <<- c(messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  expect_true(any(grepl("rank-deficient", messages, fixed = TRUE)))
  expect_equal(result$classification, "no_stage2_result")
  expect_equal(result$stage2_n_samples, 0L)
  expect_equal(result$stage2_n_rank_deficient, 2L)
  expect_true(is.na(result$stage2_fisher_fdr))
  rows <- data.table::fread(file.path(output_dir, "per_celltype", "target",
                                     "coef_per_sample.csv"))
  expect_equal(rows$fit_status, rep("rank_deficient", 2))
  expect_true(all(is.na(rows$coef)))
  expect_true(all(is.na(rows$pval)))
})
