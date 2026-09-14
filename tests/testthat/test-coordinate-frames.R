# Regression tests for the pooled-coordinate bug.
#
# The bug: a nearest-neighbour search run over cells from more than one sample
# returns neighbours from a different sample whenever tissue sections share a
# coordinate frame, which is the default for per-section Xenium and Visium
# output. Aggregating per sample afterwards does not repair it.
#
# WHY THE EXISTING BENCHMARK FIXTURES CANNOT CATCH THIS:
# generate_benchmark_data() places every sample's query cluster at
# field_um/2 +/- 30, i.e. the SAME position in every sample. A pooled search
# therefore still lands near a genuine query cell and recovers the planted
# signal, so the suite passes with or without the bug.
#
# The fixture below is the opposite construction, and it is the realistic one:
# samples share one coordinate frame (overlapping, as sections actually are)
# but each sample's query cluster sits in a DIFFERENT corner. A pooled search
# then finds another sample's cluster, which is exactly the failure mode.

make_overlapping_frames <- function(n_target = 400, seed = 1) {
  set.seed(seed)
  field <- 500
  # Same field for every sample: frames OVERLAP, as real sections do.
  # Query cluster in a different corner per sample: pooling therefore breaks.
  centres <- list(A = c(100, 100), B = c(400, 400), C = c(100, 400))

  do.call(rbind, lapply(names(centres), function(s) {
    ctr <- centres[[s]]
    ang <- runif(40, 0, 2 * pi)
    rad <- 25 * sqrt(runif(40))
    q <- data.frame(
      x = ctr[1] + rad * cos(ang), y = ctr[2] + rad * sin(ang),
      cell_type = "Query", sample_id = s, stringsAsFactors = FALSE
    )
    t <- data.frame(
      x = runif(n_target, 0, field), y = runif(n_target, 0, field),
      cell_type = "Target", sample_id = s, stringsAsFactors = FALSE
    )
    rbind(q, t)
  }))
}

# A second fixture, tuned so the pooled path gets the coefficient MAGNITUDE
# badly wrong rather than merely attenuated.
#
# Sample B carries the query cell type densely scattered across the whole
# field, while A and C carry it as a focal cluster. Pooled, cells in A and C
# find one of B's cells at near-zero distance no matter where they sit, so the
# fitted slope is inflated several-fold.
#
# Note what this fixture demonstrates and what it does not. In this synthetic
# data, the bug corrupts
# magnitude and costs statistical power but does NOT flip signs. A test that
# asserted on sign alone would pass against the buggy implementation, which is
# why the assertion below is on agreement with the planted effect size.
make_attenuating_frames <- function(n_target = 500, seed = 7, field = 500) {
  set.seed(seed)
  mk <- function(s, qx, qy, nq, scatter) {
    q <- if (scatter) {
      data.frame(x = runif(nq, 0, field), y = runif(nq, 0, field))
    } else {
      a <- runif(nq, 0, 2 * pi)
      r <- 25 * sqrt(runif(nq))
      data.frame(x = qx + r * cos(a), y = qy + r * sin(a))
    }
    rbind(
      cbind(q, cell_type = "Query", sample_id = s),
      cbind(
        data.frame(x = runif(n_target, 0, field), y = runif(n_target, 0, field)),
        cell_type = "Target", sample_id = s
      )
    )
  }
  rbind(
    mk("A", 100, 100, 40, FALSE),
    mk("B", NA, NA, 2500, TRUE),
    mk("C", 400, 400, 40, FALSE)
  )
}

test_that("check_coordinate_frames detects overlapping sections", {
  d <- make_overlapping_frames()
  res <- suppressWarnings(
    check_coordinate_frames(as.matrix(d[, c("x", "y")]), d$sample_id,
      warn = FALSE
    )
  )
  # Three samples on one shared field: every pair overlaps and the summed
  # per-sample area far exceeds the global box.
  expect_true(res$overlaps)
  expect_equal(res$n_overlapping_pairs, 3L)
  expect_equal(res$n_pairs, 3L)
  expect_gt(res$ratio, 1)
})

test_that("check_coordinate_frames passes disjoint sections", {
  set.seed(2)
  d <- do.call(rbind, lapply(0:2, function(i) {
    data.frame(
      x = runif(100, i * 10000, i * 10000 + 500),
      y = runif(100, 0, 500), sample_id = LETTERS[i + 1]
    )
  }))
  res <- check_coordinate_frames(as.matrix(d[, c("x", "y")]), d$sample_id,
    warn = FALSE
  )
  expect_false(res$overlaps)
  expect_equal(res$n_overlapping_pairs, 0L)
  expect_lte(res$ratio, 1)
})

test_that("check_coordinate_frames is a no-op for a single sample", {
  coords <- matrix(runif(100), ncol = 2)
  res <- check_coordinate_frames(coords, rep("s1", 50), warn = FALSE)
  expect_false(res$overlaps)
  expect_equal(res$n_pairs, 0L)
})

test_that("by-sample distance never returns a cross-sample neighbour", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- d$cell_type == "Query"

  d_by <- calculate_distance_to_type_by_sample(coords, d$sample_id, qmask, k = 1)

  # Recompute the truth independently, one sample at a time.
  truth <- rep(NA_real_, nrow(coords))
  for (s in unique(d$sample_id)) {
    rows <- which(d$sample_id == s)
    tgt <- rows[qmask[rows]]
    truth[rows] <- RANN::nn2(coords[tgt, , drop = FALSE],
      coords[rows, , drop = FALSE],
      k = 1
    )$nn.dists[, 1]
  }
  expect_equal(d_by, truth, tolerance = 1e-10)
})

test_that("pooled search really is wrong on this fixture (bug reproducer)", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- d$cell_type == "Query"

  # The OLD behaviour: one pooled search over all samples.
  pooled <- RANN::nn2(coords[qmask, , drop = FALSE], coords, k = 1)
  d_pooled <- pooled$nn.dists[, 1]
  owner <- d$sample_id[qmask][pooled$nn.idx[, 1]]

  d_by <- calculate_distance_to_type_by_sample(coords, d$sample_id, qmask, k = 1)

  # A substantial share of cells get a neighbour from the wrong sample, and
  # the pooled distance is systematically too small. If this test ever stops
  # failing for the pooled path, the fixture has lost its teeth.
  expect_gt(mean(owner != d$sample_id), 0.25)
  expect_lt(median(d_pooled), median(d_by))
  expect_false(isTRUE(all.equal(d_pooled, d_by)))
})

test_that("by-sample distance preserves pooled semantics", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- d$cell_type == "Query"

  # Self-match at distance 0 for query cells, as the pooled call gave.
  d_by <- calculate_distance_to_type_by_sample(coords, d$sample_id, qmask, k = 1)
  expect_true(all(d_by[qmask] == 0))

  # k > 1 returns the row mean of the k nearest, and is >= the k = 1 result.
  d_k3 <- calculate_distance_to_type_by_sample(coords, d$sample_id, qmask, k = 3)
  expect_length(d_k3, nrow(coords))
  expect_true(all(d_k3 >= d_by - 1e-9))
})

test_that("by-sample distance returns NA for samples with no query cells", {
  d <- make_overlapping_frames()
  # Strip sample C's query cells entirely.
  d$cell_type[d$sample_id == "C" & d$cell_type == "Query"] <- "Target"
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- d$cell_type == "Query"

  expect_warning(
    d_by <- calculate_distance_to_type_by_sample(coords, d$sample_id, qmask, k = 1),
    "No target cells in sample"
  )
  expect_true(all(is.na(d_by[d$sample_id == "C"])))
  expect_false(any(is.na(d_by[d$sample_id != "C"])))
})

test_that("by-sample distance validates its inputs", {
  coords <- matrix(runif(20), ncol = 2)
  expect_error(
    calculate_distance_to_type_by_sample(coords, rep("s1", 5), rep(TRUE, 10)),
    "one entry per row"
  )
  expect_error(
    calculate_distance_to_type_by_sample(
      coords, rep("s1", 10), c(NA, rep(TRUE, 9))
    ),
    "contains NA"
  )
})

test_that("run_ripple recovers the planted effect SIZE, not just its sign", {
  skip_if_not_installed("SpatialExperiment")

  BETA <- -0.01
  d <- make_attenuating_frames(n_target = 500, seed = 7)
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- d$cell_type == "Query"

  # Plant the gradient against TRUE within-sample distance.
  truth <- calculate_distance_to_type_by_sample(coords, d$sample_id, qmask, k = 1)
  lib <- 500
  set.seed(11)
  n <- nrow(d)
  planted <- vapply(seq_len(n), function(i) {
    stats::rpois(1, lib * 0.02 * exp(BETA * truth[i]))
  }, numeric(1))
  bg <- matrix(stats::rpois(n * 15, lib * 0.02), nrow = 15, byrow = TRUE)

  cts <- rbind(PLANTED = planted, bg)
  rownames(cts) <- c("PLANTED", paste0("BG", 1:15))
  colnames(cts) <- paste0("c", seq_len(n))

  spe <- SpatialExperiment::SpatialExperiment(
    assays = list(counts = cts),
    colData = S4Vectors::DataFrame(
      cell_type = d$cell_type, sample_id = d$sample_id
    ),
    spatialCoords = coords
  )

  res <- suppressWarnings(run_ripple(
    input = spe, query_celltype = "Query", celltype_column = "cell_type",
    sample_column = "sample_id", target_celltypes = "Target",
    k_neighbors = 1, max_distance_um = 800, min_cells_per_sample = 30,
    min_expr_pct = 0, min_expr_floor = 0, output_dir = tempfile(),
    verbose = FALSE
  ))

  dt <- data.table::as.data.table(res)
  row <- dt[gene == "PLANTED"]
  expect_equal(nrow(row), 1L)
  expect_lt(row$fisher_pval, 0.05)

  # THE REGRESSION ASSERTION. Sign survives the bug, so it cannot discriminate.
  # Effect size can: the partitioned search recovers BETA to within a few
  # percent, whereas the pooled search inflates it roughly threefold.
  expect_lt(abs(row$median_coef - BETA) / abs(BETA), 0.25)
})

# The partitioned search can return NA where the pooled search never could:
# a sample with no query cells has nothing to measure a distance from, whereas
# the pooled search silently handed those cells a query cell from a different
# sample. Nothing downstream of the old call was written to expect NA, so these
# guard the handover.

test_that("check_coordinate_frames tolerates NA coordinates and labels", {
  coords <- matrix(c(1, 2, 3, NA, 1, 2, 3, 4), ncol = 2)
  res <- check_coordinate_frames(coords, c("a", "a", "b", "b"), warn = FALSE)
  expect_false(res$overlaps)
  expect_equal(res$n_pairs, 1L)

  # An NA sample label is dropped, leaving two measurable samples.
  set.seed(3)
  res2 <- check_coordinate_frames(
    matrix(runif(8), ncol = 2), c("a", "a", NA, "b"),
    warn = FALSE
  )
  expect_equal(res2$n_pairs, 1L)
  expect_false(is.na(res2$ratio))

  # Dropping NAs can leave fewer than two samples, which is a no-op.
  res3 <- check_coordinate_frames(
    matrix(c(1, 2, 1, 2), ncol = 2), c("a", NA),
    warn = FALSE
  )
  expect_equal(res3$n_pairs, 0L)
  expect_false(res3$overlaps)
})

test_that("run_ripple drops cells in samples that have no query cells", {
  skip_if_not_installed("SpatialExperiment")

  data(ripple_mock_data, envir = environment())
  spe <- ripple_mock_data
  # Sample 3 loses every query cell. The total query count stays well above
  # the 10-cell floor, so run_ripple() still proceeds.
  ct <- as.character(spe$cell_type)
  ct[spe$sample_id == "sample_3" & ct == "Tumor"] <- "Fibroblast"
  spe$cell_type <- ct

  out_dir <- tempfile()
  dir.create(out_dir)

  # Only the NA-distance warning is allowed through, so the expectation below
  # cannot be satisfied by one of the unrelated low-sample-count warnings.
  expect_warning(
    res <- withCallingHandlers(
      run_ripple(
        input = spe, query_celltype = "Tumor",
        celltype_column = "cell_type", sample_column = "sample_id",
        output_dir = out_dir, min_cells_per_sample = 30,
        min_expr_pct = 0, min_expr_floor = 10, verbose = FALSE
      ),
      warning = function(w) {
        if (!grepl("no distance to a query cell", conditionMessage(w))) {
          invokeRestart("muffleWarning")
        }
      }
    ),
    "no distance to a query cell"
  )
  expect_gt(nrow(res), 0)

  # The dropped sample must be absent from the per-cell distances, and no NA
  # distance may reach the output.
  dist_file <- file.path(out_dir, "ripple", "qc", "cell_distances.csv.gz")
  skip_if_not(file.exists(dist_file))
  cd <- data.table::fread(dist_file)
  expect_false("sample_3" %in% cd$sample_id)
  expect_false(anyNA(cd$dist_to_query))
})


# Per-sample is now the DEFAULT, not an opt-in. The old design took an
# optional sample_ids and fell back to a pooled search, which protected the
# caller who already knew about this bug and silently failed the one who did
# not. Since a spatial neighbour in a different tissue section does not exist
# physically, partitioning is not a safety measure bolted on top; it is the
# only meaningful definition, so the sample labels are required.

test_that("the three helpers require sample_ids", {
  set.seed(1)
  coords <- matrix(runif(200), ncol = 2)
  types <- rep(c("A", "B"), 50)

  expect_error(build_knn_graph(coords, k = 5), "sample_ids is required")
  expect_error(build_radius_graph(coords, radius = 0.1),
    "sample_ids is required"
  )
  expect_error(calculate_distance_to_type(coords, types, "B"),
    "sample_ids is required"
  )
  # And the length must line up.
  expect_error(
    build_knn_graph(coords, k = 5, sample_ids = rep("s", 3)),
    "one entry per row"
  )
})

test_that("build_knn_graph partitions and returns global indices", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  g <- suppressMessages(
    build_knn_graph(coords, k = 4, sample_ids = d$sample_id)
  )

  expect_equal(dim(g$indices), c(nrow(coords), 4L))
  expect_false(anyNA(g$indices))

  # Every neighbour must belong to the same sample as the cell it serves, and
  # the indices must be global row numbers so the caller need not track the
  # partition.
  own <- matrix(d$sample_id[g$indices], nrow = nrow(coords))
  expect_true(all(own == d$sample_id))
  # No cell is its own neighbour.
  expect_false(any(g$indices == seq_len(nrow(coords))))
  # Distances agree with an independent within-sample recomputation.
  expect_equal(
    g$distances[, 1],
    vapply(seq_len(nrow(coords)), function(i) {
      same <- which(d$sample_id == d$sample_id[i])
      same <- setdiff(same, i)
      min(sqrt(rowSums((coords[same, , drop = FALSE] -
        matrix(coords[i, ], nrow = length(same), ncol = 2, byrow = TRUE))^2)))
    }, numeric(1)),
    tolerance = 1e-8
  )
})

test_that("build_knn_graph pads and warns when a sample is too small", {
  coords <- matrix(runif(20), ncol = 2)
  sid <- c(rep("big", 8), "lonely", "lonely")
  expect_warning(
    g <- suppressMessages(build_knn_graph(coords, k = 5, sample_ids = sid)),
    "Fewer than k"
  )
  # "lonely" has 2 cells, so exactly one neighbour is available.
  lonely <- which(sid == "lonely")
  expect_false(anyNA(g$indices[lonely, 1]))
  expect_true(all(is.na(g$indices[lonely, 2:5])))
})

test_that("build_radius_graph partitions", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  nb <- suppressMessages(
    build_radius_graph(coords, radius = 40, sample_ids = d$sample_id)
  )
  expect_length(nb, nrow(coords))
  # No neighbour may come from another sample.
  bad <- vapply(seq_along(nb), function(i) {
    length(nb[[i]]) > 0 && any(d$sample_id[nb[[i]]] != d$sample_id[i])
  }, logical(1))
  expect_false(any(bad))
  # And it must actually find neighbours on this fixture.
  expect_gt(sum(lengths(nb)), 0)
})

test_that("calculate_distance_to_type always partitions", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  ct <- ifelse(d$cell_type == "Query", "Q", "T")

  got <- calculate_distance_to_type(coords, ct, "Q", sample_ids = d$sample_id)
  ref <- calculate_distance_to_type_by_sample(
    coords, d$sample_id, ct == "Q", k = 1
  )
  expect_equal(got, ref)

  # A pooled search on this fixture is measurably different, so the test
  # would notice a regression back to pooling.
  pooled <- RANN::nn2(coords[ct == "Q", , drop = FALSE], coords, k = 1)
  expect_false(isTRUE(all.equal(got, as.vector(pooled$nn.dists))))
})

# Severity. The area ratio is a ONE-SIDED test and must never be used as a
# gate: below 1 it says nothing at all.

test_that("a low area ratio does not rule out severe overlap", {
  set.seed(1)
  # Ten sections of 100x100 tiled with wide gaps across a 2000x2000 frame,
  # except the last two are placed at the identical location.
  pos <- list(
    c(0, 0), c(500, 0), c(1000, 0), c(1900, 0),
    c(0, 900), c(500, 900), c(1000, 900), c(1900, 900),
    c(700, 1900), c(700, 1900)
  )
  d <- do.call(rbind, lapply(seq_along(pos), function(i) {
    data.frame(
      x = runif(200, pos[[i]][1], pos[[i]][1] + 100),
      y = runif(200, pos[[i]][2], pos[[i]][2] + 100),
      sample_id = paste0("S", i)
    )
  }))
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- rep(c(TRUE, rep(FALSE, 9)), length.out = nrow(d))

  res <- suppressWarnings(check_coordinate_frames(
    coords, d$sample_id,
    target_mask = qmask, warn = FALSE
  ))

  # The ratio is far below 1, yet one pair overlaps perfectly. Gating on the
  # ratio would have suppressed the alarm on the worst case there is.
  expect_lt(res$ratio, 0.1)
  expect_true(res$overlaps)
  expect_equal(res$n_overlapping_pairs, 1L)

  # Severity catches what the ratio misses.
  expect_gt(res$cross_sample_fraction, 0.05)
  expect_true(res$severe)
})

test_that("severity grades overlap and drives the warning", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- d$cell_type == "Query"

  severe <- suppressWarnings(check_coordinate_frames(
    coords, d$sample_id,
    target_mask = qmask, warn = FALSE
  ))
  expect_true(severe$overlaps)
  expect_true(severe$severe)
  expect_gt(severe$cross_sample_fraction, 0.25)
  expect_gt(severe$n_cells_checked, 0)
  expect_warning(
    check_coordinate_frames(coords, d$sample_id, target_mask = qmask),
    "DIFFERENT sample"
  )

  # severe_fraction labels the result; it must NOT decide whether the user is
  # told. Overlap is a fact about the coordinates, so raising the threshold
  # above the measured fraction clears the flag and still warns.
  expect_warning(
    graded <- check_coordinate_frames(coords, d$sample_id,
      target_mask = qmask, severe_fraction = 0.99
    ),
    "overlapping coordinate bounding boxes"
  )
  expect_false(graded$severe)
  expect_true(graded$overlaps)

  # And a low-severity overlap still warns, which is the CosMx case: 2 of 10
  # pairs intersect and only 2.9% of cells take a cross-patient neighbour, but
  # the overlap is real and the user should hear about it.
  expect_warning(
    check_coordinate_frames(coords, d$sample_id,
      target_mask = qmask, severe_fraction = 0.99
    ),
    "DIFFERENT sample"
  )

  # Without a target mask severity cannot be judged, so any overlap counts as
  # severe rather than being waved through.
  blind <- suppressWarnings(
    check_coordinate_frames(coords, d$sample_id, warn = FALSE)
  )
  expect_true(is.na(blind$cross_sample_fraction))
  expect_true(blind$severe)
})

test_that("the severity estimate does not touch the RNG stream", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- d$cell_type == "Query"

  # A seeded permutation downstream must not shift because a diagnostic ran.
  set.seed(99)
  before <- runif(3)
  set.seed(99)
  invisible(suppressWarnings(check_coordinate_frames(
    coords, d$sample_id,
    target_mask = qmask, warn = FALSE, max_cells = 100L
  )))
  after <- runif(3)
  expect_equal(before, after)
})

test_that("disjoint sections report no overlap and no severity", {
  set.seed(2)
  d <- do.call(rbind, lapply(0:2, function(i) {
    data.frame(
      x = runif(100, i * 10000, i * 10000 + 500),
      y = runif(100, 0, 500), sample_id = LETTERS[i + 1]
    )
  }))
  coords <- as.matrix(d[, c("x", "y")])
  qmask <- rep(c(TRUE, FALSE, FALSE, FALSE), length.out = nrow(d))
  res <- check_coordinate_frames(coords, d$sample_id,
    target_mask = qmask, warn = TRUE
  )
  expect_false(res$overlaps)
  expect_false(res$severe)
  # Not measured, because there is no overlap to measure the cost of.
  expect_true(is.na(res$cross_sample_fraction))
})

# STRUCTURAL GUARANTEE: per-sample is not merely the default, it is the only
# path, and that is provable behaviourally rather than by reading the source.
#
# Within-sample distances are invariant when each sample is translated by its
# own arbitrary offset. Cross-sample distances are not. So if ANY pooled search
# survived anywhere in the pipeline, sliding the sections around would move the
# results. Identical output across three layouts (overlapping as shipped,
# pushed apart into disjoint space, and stacked exactly on top of each other)
# is therefore a proof that no cross-sample neighbour is ever consulted.
#
# This is the test to keep if any other in this file is ever dropped: it does
# not care how the partitioning is implemented, only that it is total.

translate_by_sample <- function(spe, offsets) {
  xy <- SpatialExperiment::spatialCoords(spe)
  sid <- as.character(spe$sample_id)
  for (s in names(offsets)) {
    i <- sid == s
    xy[i, 1] <- xy[i, 1] + offsets[[s]][1]
    xy[i, 2] <- xy[i, 2] + offsets[[s]][2]
  }
  SpatialExperiment::spatialCoords(spe) <- xy
  spe
}

run_quiet <- function(spe) {
  set.seed(1)
  without_frame_warning(suppressWarnings(run_ripple(
    input = spe, query_celltype = "Tumor",
    celltype_column = "cell_type", sample_column = "sample_id",
    output_dir = tempfile(), min_cells_per_sample = 30,
    min_expr_pct = 0, min_expr_floor = 10, verbose = FALSE
  )))
}

test_that("results are invariant to per-sample translation", {
  skip_if_not_installed("SpatialExperiment")

  data(ripple_mock_data, envir = environment())
  base <- ripple_mock_data

  # As shipped: three sections sharing one frame.
  as_is <- check_coordinate_frames(
    SpatialExperiment::spatialCoords(base), base$sample_id,
    warn = FALSE
  )
  expect_true(as_is$overlaps)

  # Pushed far apart, so no two sections share any coordinate.
  apart <- translate_by_sample(base, list(
    sample_1 = c(0, 0),
    sample_2 = c(50000, 0),
    sample_3 = c(0, 50000)
  ))
  expect_false(check_coordinate_frames(
    SpatialExperiment::spatialCoords(apart), apart$sample_id,
    warn = FALSE
  )$overlaps)

  # Stacked: every section shifted so its own bounding box starts at the
  # origin, the worst case for a pooled search.
  xy <- SpatialExperiment::spatialCoords(base)
  sid <- as.character(base$sample_id)
  offs <- lapply(split(seq_len(nrow(xy)), sid), function(i) {
    c(-min(xy[i, 1]), -min(xy[i, 2]))
  })
  stacked <- translate_by_sample(base, offs)

  r_as_is <- run_quiet(base)
  r_apart <- run_quiet(apart)
  r_stacked <- run_quiet(stacked)

  key <- c("gene", "cell_type")
  cols <- c("median_coef", "fisher_pval", "fisher_fdr", "sign_consistency")
  for (nm in key) {
    expect_identical(as.character(r_apart[[nm]]), as.character(r_as_is[[nm]]))
    expect_identical(as.character(r_stacked[[nm]]), as.character(r_as_is[[nm]]))
  }
  for (nm in cols) {
    expect_equal(r_apart[[nm]], r_as_is[[nm]], tolerance = 1e-12)
    expect_equal(r_stacked[[nm]], r_as_is[[nm]], tolerance = 1e-12)
  }
})

test_that("the helpers are invariant to per-sample translation too", {
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  sid <- d$sample_id
  ct <- ifelse(d$cell_type == "Query", "Q", "T")

  shift <- coords
  for (k in seq_along(unique(sid))) {
    s <- unique(sid)[k]
    shift[sid == s, 1] <- shift[sid == s, 1] + k * 25000
    shift[sid == s, 2] <- shift[sid == s, 2] - k * 13000
  }

  expect_equal(
    calculate_distance_to_type(shift, ct, "Q", sample_ids = sid),
    calculate_distance_to_type(coords, ct, "Q", sample_ids = sid),
    tolerance = 1e-9
  )

  g0 <- suppressMessages(build_knn_graph(coords, k = 4, sample_ids = sid))
  g1 <- suppressMessages(build_knn_graph(shift, k = 4, sample_ids = sid))
  expect_identical(g1$indices, g0$indices)
  expect_equal(g1$distances, g0$distances, tolerance = 1e-9)

  n0 <- suppressMessages(build_radius_graph(coords, 40, sample_ids = sid))
  n1 <- suppressMessages(build_radius_graph(shift, 40, sample_ids = sid))
  expect_identical(n1, n0)
})

test_that("no reachable code path errors on overlapping frames", {
  # Overlap is now harmless, so nothing may refuse to run because of it. The
  # earlier design errored out of the graph builders; that path is gone.
  d <- make_overlapping_frames()
  coords <- as.matrix(d[, c("x", "y")])
  sid <- d$sample_id
  ct <- ifelse(d$cell_type == "Query", "Q", "T")

  expect_no_error(suppressMessages(build_knn_graph(coords, 4, sample_ids = sid)))
  expect_no_error(
    suppressMessages(build_radius_graph(coords, 40, sample_ids = sid))
  )
  expect_no_error(calculate_distance_to_type(coords, ct, "Q", sample_ids = sid))
  expect_no_error(
    suppressWarnings(check_coordinate_frames(coords, sid, target_mask = ct == "Q"))
  )
})
