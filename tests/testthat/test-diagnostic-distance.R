make_diagnostic_spe <- function() {
  set.seed(713)
  query <- rbind(c(0, 0), c(35, 0), c(0, 40))
  target <- cbind(runif(80, 10, 450), runif(80, -90, 90))
  xy <- rbind(query, target, query + 50, target)
  ct <- rep(c(rep("query", 3), rep("target", 80)), 2)
  sid <- rep(c("s1", "s2"), each = 83)
  counts <- rbind(signal = rpois(nrow(xy), 15), background = 0)
  counts[2, ] <- 500 - counts[1, ]
  colnames(counts) <- paste0("c", seq_len(ncol(counts)))
  SpatialExperiment::SpatialExperiment(
    assays = list(counts = methods::as(counts, "CsparseMatrix")),
    colData = S4Vectors::DataFrame(cell_type = ct, sample_id = sid,
                                 row.names = colnames(counts)),
    spatialCoords = xy
  )
}

test_that("residual diagnostic matches capped per-sample mean-neighbor GLMs", {
  skip_if_not_installed("spdep")
  skip_if_not_installed("SpatialExperiment")
  spe <- make_diagnostic_spe()
  run_diag <- function(x, ...) suppressWarnings(check_spatial_autocorrelation(
    x, genes = "signal", celltype_column = "cell_type",
    target_celltype = "target", query_celltype = "query", k = 6,
    verbose = FALSE, ...
  ))
  observed <- run_diag(spe, k_neighbors = 3, max_distance_um = 100)
  xy <- SpatialExperiment::spatialCoords(spe)
  meta <- as.data.frame(SummarizedExperiment::colData(spe))
  counts <- as.matrix(SummarizedExperiment::assay(spe, "counts"))
  for (sid in c("s1", "s2")) {
    q <- which(meta$sample_id == sid & meta$cell_type == "query")
    t <- which(meta$sample_id == sid & meta$cell_type == "target")
    distance <- rowMeans(RANN::nn2(xy[q, ], xy[t, ], k = 3)$nn.dists)
    expect_gt(sum(distance > 100), 0)
    distance <- pmin(distance, 100)
    y <- counts["signal", t]
    exposure <- colSums(counts[, t])
    fit <- glm(y ~ distance + offset(log(exposure)), family = poisson())
    weights <- suppressWarnings(spdep::nb2listw(
      spdep::knn2nb(spdep::knearneigh(xy[t, ], k = 6)), style = "W"
    ))
    expected <- spdep::moran.test(residuals(fit, type = "deviance"),
                                  weights, alternative = "two.sided")
    row <- observed[sample_id == sid]
    expect_equal(row$n_cells, length(t))
    expect_equal(row$morans_i, unname(expected$estimate[1]), tolerance = 1e-10)
    expect_equal(row$morans_pvalue, expected$p.value, tolerance = 1e-10)
  }
  # Query-neighbor choice is independent of the fixed Moran neighborhood.
  nearest <- run_diag(spe, k_neighbors = 1, max_distance_um = 100)
  expect_false(isTRUE(all.equal(observed$morans_i, nearest$morans_i)))
  data.table::setindexv(observed, NULL)
  withr::local_options(ripple.k_neighbors = 3, ripple.max_distance_um = 100)
  expect_equal(run_diag(spe), observed)
  expect_equal(run_diag(spe, k_neighbors = 1), nearest)

  # Translating one entire sample cannot change either its fit or its graph.
  moved <- spe
  shifted <- xy
  shifted[meta$sample_id == "s2", ] <- shifted[meta$sample_id == "s2", ] + 10000
  SpatialExperiment::spatialCoords(moved) <- shifted
  expect_equal(run_diag(moved), observed, tolerance = 1e-10)

  # A sample without query cells must not borrow them from another section.
  SummarizedExperiment::colData(spe)$cell_type[
    meta$sample_id == "s2" & meta$cell_type == "query"
  ] <- "other"
  expect_equal(run_diag(spe)$sample_id, "s1")
})

test_that("diagnostic honors sample eligibility and cannot fit a constant cap", {
  skip_if_not_installed("spdep")
  skip_if_not_installed("SpatialExperiment")
  spe <- make_diagnostic_spe()
  run_diag <- function(...) suppressWarnings(check_spatial_autocorrelation(
    spe, "signal", "cell_type", "target", "query", k = 6,
    verbose = FALSE, ...
  ))
  withr::local_options(ripple.min_cells_per_sample = 81)
  expect_true(all(run_diag()$interpretation == "insufficient_cells"))
  expect_true(all(run_diag(min_cells_per_sample = 30, max_distance_um = 0.1)$
                    interpretation == "glm_failed"))
  expect_error(run_diag(k_neighbors = 0), "positive integer")
  expect_error(run_diag(max_distance_um = NA_real_), "must be positive")
  expect_error(suppressWarnings(check_spatial_autocorrelation(
    spe, "signal", "cell_type", "absent", "query", verbose = FALSE
  )), "No target cells")
})
