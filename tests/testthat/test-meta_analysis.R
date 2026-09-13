test_that("zero p-values retain strong replicates in Fisher combination", {
  result <- compute_fisher_pval(c(0, 0, 0), c(-0.02, -0.03, -0.04))
  expected_stat <- -2 * 3 * log(1e-15)
  expect_equal(result$n_valid, 3L)
  expect_equal(result$n_sig_samples, 3L)
  expect_equal(result$median_coef, -0.03)
  expect_equal(result$sign_consistency, 1)
  expect_equal(result$fisher_stat, expected_stat)
  expect_equal(result$fisher_pval,
    stats::pchisq(expected_stat, df = 6, lower.tail = FALSE))
})

test_that("an underflowed opposing replicate still fails the sign gate", {
  result <- compute_fisher_pval(c(0, 0.001, 0.001), c(0.02, -0.01, -0.01))
  expect_equal(result$n_valid, 3L)
  expect_equal(result$n_sig_samples, 3L)
  expect_equal(result$sign_consistency, 2 / 3)
  expect_equal(result$fisher_pval, 1)
})

test_that("zero p-values count towards the optional significance gate", {
  result <- compute_fisher_pval(c(0, 0.01, 0.8), c(-0.03, -0.02, -0.01),
    min_sig_fraction = 2 / 3)
  expect_equal(result$n_sig_samples, 2L)
  expect_lt(result$fisher_pval, 0.05)
  expect_equal(result$median_coef, -0.02)
})

test_that("invalid probabilities and nonfinite coefficients are excluded", {
  result <- compute_fisher_pval(
    c(0, 0.01, NA, NaN, Inf, -0.1, 1.1, 0.01),
    c(-0.03, -0.01, -1, -1, -1, -1, -1, Inf))
  expect_equal(result$n_valid, 2L)
  expect_equal(result$median_coef, -0.02)
  expect_true(is.finite(result$fisher_pval))
})

test_that("compute_fisher_pval combines consistent results", {
  # 4 samples all showing negative effect
  pvals <- c(0.01, 0.02, 0.05, 0.03)
  coefs <- c(-0.005, -0.003, -0.004, -0.006)

  result <- compute_fisher_pval(pvals, coefs)

  expect_type(result, "list")
  expect_true(result$fisher_pval < 0.01)
  expect_equal(result$sign_consistency, 1.0)
  expect_equal(result$median_coef, median(coefs))
})

test_that("compute_fisher_pval gates inconsistent signs", {
  # 4 samples with mixed signs
  pvals <- c(0.01, 0.02, 0.05, 0.03)
  coefs <- c(-0.005, 0.003, -0.004, 0.006) # mixed signs

  result <- compute_fisher_pval(pvals, coefs)

  expect_equal(result$fisher_pval, 1.0) # Gated
  expect_true(result$sign_consistency < 1.0)
})

test_that("compute_fisher_pval handles single sample", {
  result <- compute_fisher_pval(c(0.01), c(-0.005), min_samples = 1)
  expect_type(result, "list")
})

# run_meta_analysis() was removed when the REML inverse-variance step was
# dropped from the hot path. gradient_score is now the median of per-sample
# coefficients (see test below).

test_that("median_coef equals median of per-sample coefficients", {
  coefs <- c(-0.005, -0.003, -0.004, -0.006)
  pvals <- c(0.01, 0.02, 0.05, 0.03)

  result <- compute_fisher_pval(pvals, coefs)
  expect_equal(result$median_coef, median(coefs))
})

test_that("n_sig_samples counts per-sample significant replicates", {
  # 2 of 4 samples below 0.05
  pvals <- c(0.001, 0.6, 0.5, 0.02)
  coefs <- c(-0.005, -0.001, -0.002, -0.004)

  result <- compute_fisher_pval(pvals, coefs)
  expect_equal(result$n_sig_samples, 2L)
  # diagnostic is reported but does not gate by default
  expect_true(result$fisher_pval < 1.0)
})

test_that("min_sig_fraction gate blocks one-replicate-driven calls", {
  # One strong replicate, three near-null but same sign
  pvals <- c(1e-6, 0.6, 0.5, 0.7)
  coefs <- c(-0.005, -0.001, -0.002, -0.003)

  # Default (gate off): significant, driven by the one replicate
  open <- compute_fisher_pval(pvals, coefs)
  expect_true(open$fisher_pval < 0.05)
  expect_equal(open$n_sig_samples, 1L)

  # Require majority individually significant: gated out
  gated <- compute_fisher_pval(pvals, coefs, min_sig_fraction = 0.5)
  expect_equal(gated$fisher_pval, 1.0)
  expect_equal(gated$n_sig_samples, 1L) # diagnostic still reported
})

test_that("min_sig_fraction passes genuinely reproducible calls", {
  # All four samples individually significant
  pvals <- c(0.01, 0.02, 0.03, 0.04)
  coefs <- c(-0.005, -0.003, -0.004, -0.006)

  gated <- compute_fisher_pval(pvals, coefs, min_sig_fraction = 0.5)
  expect_true(gated$fisher_pval < 0.05)
  expect_equal(gated$n_sig_samples, 4L)
})
