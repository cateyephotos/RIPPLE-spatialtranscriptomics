make_pool_fixture <- function() {
  set.seed(1819)
  n <- 160
  xy <- cbind(runif(2 * n, 0, 500), runif(2 * n, 0, 500))
  mask <- rep(c(rep(TRUE, 100), rep(FALSE, 60)), 2)
  sid <- rep(c("s1", "s2"), each = n)
  list(xy = xy, mask = mask, sid = sid,
       counts = rpois(sum(mask), 5), query_n = c(s1 = 12, s2 = 15))
}

test_that("non-target permutation pool preserves cell identity and sample counts", {
  d <- make_pool_fixture()
  # A non-target cell can share coordinates with a target cell and remains
  # eligible. Coordinates alone must not be used to decide exclusion.
  d$xy[101, ] <- d$xy[1, ]
  pool <- ripple:::.prepare_permutation_pool(
    d$xy, d$sid, d$query_n, d$mask, "non_target"
  )
  expect_equal(pool$coords, d$xy[!d$mask, ])
  expect_equal(pool$coords[1, ], d$xy[1, ])
  expect_equal(table(pool$sample_ids), table(d$sid[!d$mask]))
  expect_error(ripple:::.prepare_permutation_pool(
    d$xy, d$sid, d$query_n, NULL, "non_target"
  ), "target_mask_all")
  expect_error(ripple:::.prepare_permutation_pool(
    d$xy, d$sid, d$query_n, c(NA, d$mask[-1]), "non_target"
  ), "target_mask_all")
  expect_error(ripple:::.prepare_permutation_pool(
    d$xy, d$sid, c(s1 = 61, s2 = 15), d$mask, "non_target"
  ), "Too few non-target candidates in sample 's1'")
})

test_that("both permutation APIs use the same non-target null and retain legacy mode", {
  d <- make_pool_fixture()
  run_one <- function(xy = d$xy, ...) run_permutation_test(
    counts = d$counts, coords_target = xy[d$mask, ], coords_all = xy,
    sample_ids = d$sid[d$mask], sample_ids_all = d$sid,
    query_per_sample = d$query_n, n_perms = 25, observed_coef = -0.001,
    k_neighbors = 3, total_counts = rep(500, sum(d$mask)), max_distance = 100, ...
  )
  set.seed(32)
  result <- run_one(target_mask_all = d$mask)
  # Independent reference: explicitly restrict candidates, then run the
  # unchanged legacy sampler. Target data, k and cap remain identical.
  set.seed(32)
  reference <- run_permutation_test(
    d$counts, d$xy[d$mask, ], d$xy[!d$mask, ], d$sid[d$mask],
    n_perms = 25, observed_coef = -0.001, sample_ids_all = d$sid[!d$mask],
    query_per_sample = d$query_n, k_neighbors = 3,
    total_counts = rep(500, sum(d$mask)), max_distance = 100,
    permutation_pool = "all"
  )
  expect_equal(result$null_coefs, reference$null_coefs)
  expect_equal(result$perm_pval, reference$perm_pval)
  expect_equal(result$permutation_pool, "non_target")
  set.seed(32)
  legacy <- run_one(permutation_pool = "all")
  expect_equal(legacy$permutation_pool, "all")
  expect_false(isTRUE(all.equal(result$null_coefs, legacy$null_coefs)))

  moved <- d$xy
  moved[d$sid == "s2", ] <- moved[d$sid == "s2", ] + 10000
  set.seed(32)
  translated <- run_one(xy = moved, target_mask_all = d$mask)
  expect_equal(translated$null_coefs, result$null_coefs, tolerance = 1e-10)

  barcodes <- paste0("c", seq_along(d$counts))
  counts <- matrix(d$counts, nrow = 1, dimnames = list("g", barcodes))
  set.seed(32)
  batch <- run_permutation_tests(
    "g", counts, barcodes, d$xy[d$mask, ], d$xy, d$sid[d$mask], d$sid,
    d$query_n, c(g = -0.001), 25, 3, 100, 30, 5, rep(500, sum(d$mask)),
    target_mask_all = d$mask
  )
  expect_equal(batch$perm_pval, result$perm_pval)
  expect_equal(batch$permutation_pool, "non_target")
})

test_that("imported permutation results replace stale pool labels", {
  base <- tempfile("permutation_import_")
  ct <- file.path(base, "per_celltype", "target")
  dir.create(ct, recursive = TRUE)
  on.exit(unlink(base, recursive = TRUE), add = TRUE)
  meta_path <- file.path(ct, "meta_analysis_results.csv")
  perm_path <- file.path(ct, "permutation_pvals.csv")
  data.table::fwrite(data.table::data.table(
    gene = "g", perm_pval = 0.1, permutation_pool = "non_target"
  ), meta_path)
  data.table::fwrite(data.table::data.table(gene = "g", perm_pval = 0.2), perm_path)
  suppressMessages(merge_permutation_results(base))
  imported <- data.table::fread(meta_path)
  expect_equal(imported$perm_pval, 0.2)
  expect_equal(imported$permutation_pool, "unspecified")
  data.table::fwrite(data.table::data.table(
    gene = "g", perm_pval = 0.3, permutation_pool = "all"
  ), perm_path)
  suppressMessages(merge_permutation_results(base))
  expect_equal(data.table::fread(meta_path)$permutation_pool, "all")
})
