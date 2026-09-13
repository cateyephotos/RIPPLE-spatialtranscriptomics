#' @title Permutation Testing Functions
#'
#' @description Functions for label-permutation testing to validate query cell
#'   specificity of distance-expression gradients, and for merging GPU
#'   permutation results.
#'
#' @name permutation
NULL

# Select candidates by cell identity, not coordinate equality: distinct cells
# can legitimately share coordinates. The mask includes every cell of the
# target population, including target cells not retained for gene fitting.
.prepare_permutation_pool <- function(coords_all, sample_ids_all,
                                      query_per_sample, target_mask_all,
                                      permutation_pool) {
  if (length(sample_ids_all) != nrow(coords_all)) {
    stop("sample_ids_all must align with coords_all.", call. = FALSE)
  }
  if (permutation_pool == "non_target") {
    if (!is.logical(target_mask_all) ||
        length(target_mask_all) != nrow(coords_all) || anyNA(target_mask_all)) {
      stop("For permutation_pool = 'non_target', supply target_mask_all as a ",
        "logical vector marking all target-population cells in coords_all. ",
        "Use permutation_pool = 'all' to reproduce the previous full-pool null.",
        call. = FALSE)
    }
    keep <- !target_mask_all
    coords_all <- coords_all[keep, , drop = FALSE]
    sample_ids_all <- sample_ids_all[keep]
    for (samp in names(query_per_sample)) {
      required <- query_per_sample[[samp]]
      if (is.finite(required) && required > 0 &&
          sum(sample_ids_all == samp, na.rm = TRUE) < required) {
        stop("Too few non-target candidates in sample '", samp,
          "' to preserve its query count (", required, ").", call. = FALSE)
      }
    }
  }
  list(coords = coords_all, sample_ids = sample_ids_all)
}

#' Single-gene permutation test
#'
#' Validates a distance-expression gradient by shuffling query cell labels
#' within each sample and recalculating the distance-expression coefficient.
#' Uses stratified sampling to preserve per-sample query cell counts.
#'
#' @param counts Integer vector of raw transcript counts for target cells.
#' @param coords_target Numeric matrix (n_target x 2). Coordinates of target
#'   cells.
#' @param coords_all Numeric matrix (n_all x 2). Coordinates of ALL cells
#'   (for drawing pseudo-query cells from).
#' @param sample_ids Character vector (length n_target). Sample IDs for each
#'   target cell.
#' @param n_perms Integer. Number of permutations (default: 500).
#' @param observed_coef Numeric. The observed combined coefficient from the
#'   real analysis.
#' @param sample_ids_all Character vector (length n_all). Sample IDs for all
#'   cells.
#' @param query_per_sample Named integer vector. Number of query cells per
#'   sample (names = sample IDs).
#' @param k_neighbors Integer. Number of nearest neighbors for distance
#'   calculation (default: 1).
#' @param total_counts Numeric vector (length n_target). Total counts per
#'   target cell (for Poisson offset).
#' @param max_distance Numeric. Distance cap in um (default: 200).
#'   Farther target cells remain in the fit at this distance.
#' @param min_cells_per_sample Integer. Minimum target cells per sample
#'   (default: 30).
#' @param min_expr_cells Integer. Minimum expressing cells for GLM fit
#'   (default: 5).
#' @param permutation_pool Candidate population for pseudo-query cells:
#'   \code{"non_target"} (default) excludes the target population, keeping
#'   query and target roles distinct; \code{"all"} reproduces the previous
#'   full-cell-pool sampling, including target cells.
#' @param target_mask_all Logical vector aligned with \code{coords_all}, marking
#'   every cell of the target population. Required for \code{"non_target"}.
#'   Exclusion uses cell identity, not coordinate matching.
#'
#' @return A list with:
#' \describe{
#'   \item{\code{null_coefs}}{Numeric vector of null distribution coefficients.}
#'   \item{\code{perm_pval}}{Numeric. Two-sided empirical p-value.}
#'   \item{\code{permutation_pool}}{Candidate population used.}
#' }
#'
#' @details For each permutation:
#' \enumerate{
#'   \item Randomly sample pseudo-query cells WITHIN each sample (stratified),
#'     preserving the original query cell count per sample. By default, target
#'     cells are excluded from the candidate pool. The observed target cells,
#'     query-neighbor count and distance cap stay fixed.
#'   \item Compute within-sample distances to the nearest pseudo-query cell,
#'     or the mean of the \code{k_neighbors} nearest cells, then cap distances.
#'   \item Fit per-sample Poisson GLMs and take the median of per-sample
#'     coefficients -- the same statistic as the observed \code{median_coef},
#'     so the empirical p-value compares like with like.
#' }
#'
#' The empirical p-value is calculated as:
#'   \code{(sum(|null| >= |observed|) + 1) / (n_valid_perms + 1)}
#' The default null compares the observed query locations with random locations
#' among non-target cells. It does not test specificity against a particular
#' alternative cell type. Too few non-target candidates raises an error rather
#' than changing the query count or falling back to the full pool.
#' Fisher combination and the sign gate are not applied to permutations.
#' The standalone R and GPU scripts retain the legacy full-cell pool;
#' their null is not the default non-target null used by these package APIs.
#'
#' @examples
#' \dontrun{
#' all_xy <- matrix(runif(2000, 0, 500), ncol = 2)
#' all_samples <- rep(c("s1", "s2"), each = 500)
#' target_mask <- rep(c(rep(TRUE, 50), rep(FALSE, 450)), 2)
#' result <- run_permutation_test(
#'   counts = rpois(100, 5),
#'   coords_target = all_xy[target_mask, ],
#'   coords_all = all_xy,
#'   sample_ids = all_samples[target_mask],
#'   n_perms = 100,
#'   observed_coef = -0.005,
#'   sample_ids_all = all_samples,
#'   query_per_sample = c(s1 = 50, s2 = 60),
#'   target_mask_all = target_mask,
#'   k_neighbors = 1,
#'   total_counts = rpois(100, 5000)
#' )
#' }
#'
#' @importFrom RANN nn2
#' @importFrom stats glm poisson median
#' @export
run_permutation_test <- function(counts, coords_target, coords_all,
                                 sample_ids, n_perms = 500,
                                 observed_coef, sample_ids_all,
                                 query_per_sample, k_neighbors = 1,
                                 total_counts,
                                 max_distance = 200,
                                 min_cells_per_sample = 30,
                                 min_expr_cells = 5,
                                 permutation_pool = c("non_target", "all"),
                                 target_mask_all = NULL) {
  permutation_pool <- match.arg(permutation_pool)
  pool <- .prepare_permutation_pool(coords_all, sample_ids_all, query_per_sample,
                                    target_mask_all, permutation_pool)
  coords_all <- pool$coords
  sample_ids_all <- pool$sample_ids
  null_coefs <- numeric(n_perms)
  unique_samples <- names(query_per_sample)

  for (i in seq_len(n_perms)) {
    # STRATIFIED SAMPLING: draw pseudo-query cells WITHIN each sample, AND
    # search within each sample.
    #
    # Drawing per sample but then rbind-ing the draws and running one pooled
    # RANN::nn2 over all target cells is stratified in COUNT but not in SPACE.
    # Because sections routinely share a coordinate frame, that pooled search
    # returns pseudo-query cells from other samples, so the null carries the
    # same defect as a pooled observed statistic and perm_pval comes out
    # plausible either way. The permutation then cannot detect the very bug it
    # would need to. Both the draw and the search must be per sample.
    perm_distances <- rep(NA_real_, nrow(coords_target))
    n_pseudo_total <- 0L

    for (samp in unique_samples) {
      n_to_sample <- query_per_sample[samp]
      if (is.na(n_to_sample) || n_to_sample <= 0) next

      samp_coords <- coords_all[sample_ids_all == samp, , drop = FALSE]
      if (nrow(samp_coords) < n_to_sample) next

      pseudo <- samp_coords[sample(nrow(samp_coords), n_to_sample), ,
        drop = FALSE
      ]
      n_pseudo_total <- n_pseudo_total + nrow(pseudo)

      # Only this sample's target cells search only this sample's pseudo-query
      tgt_idx <- which(sample_ids == samp)
      if (!length(tgt_idx)) next

      eff_k <- min(k_neighbors, nrow(pseudo))
      nn_s <- RANN::nn2(pseudo, coords_target[tgt_idx, , drop = FALSE],
        k = eff_k
      )
      d_s <- if (eff_k == 1) {
        as.vector(nn_s$nn.dists)
      } else {
        rowMeans(nn_s$nn.dists)
      }
      perm_distances[tgt_idx] <- pmin(d_s, max_distance)
    }

    if (n_pseudo_total < 5) {
      null_coefs[i] <- NA
      next
    }

    # Calculate Poisson coefficients per sample for the equal-weight median.
    coefs <- numeric(length(unique_samples))
    ses <- numeric(length(unique_samples))

    for (j in seq_along(unique_samples)) {
      samp <- unique_samples[j]
      idx <- which(sample_ids == samp)

      if (length(idx) >= min_cells_per_sample) {
        samp_counts <- counts[idx]
        samp_dist <- perm_distances[idx]
        samp_log_total <- log(total_counts[idx])

        if (length(samp_counts) >= min_cells_per_sample &&
          sum(samp_counts > 0) >= min_expr_cells) {
          fit <- tryCatch(
            {
              suppressWarnings(stats::glm(samp_counts ~ samp_dist + offset(samp_log_total),
                family = stats::poisson()
              ))
            },
            error = function(e) NULL
          )

          if (!is.null(fit) && fit$converged) {
            coef_summary <- summary(fit)$coefficients
            if (nrow(coef_summary) >= 2) {
              coefs[j] <- coef_summary[2, "Estimate"]
              ses[j] <- coef_summary[2, "Std. Error"]
            }
          }
        }
      }
    }

    # Null statistic must match the observed statistic (median_coef, the
    # equal-weight median of per-sample coefficients from compute_fisher_pval).
    # ses > 0 identifies samples whose GLM actually converged. It also covers a
    # sample skipped by the draw loop above: its perm_distances stay NA, the
    # glm() call fails, tryCatch returns NULL, and coefs/ses stay at their zero
    # initial value, so the sample is excluded here rather than contributing a
    # spurious zero coefficient.
    valid <- !is.na(coefs) & !is.na(ses) & ses > 0
    if (sum(valid) >= 2) {
      null_coefs[i] <- stats::median(coefs[valid])
    } else {
      null_coefs[i] <- NA
    }
  }

  # Calculate empirical p-value (two-sided)
  null_coefs <- null_coefs[!is.na(null_coefs)]
  if (length(null_coefs) < 10) {
    warning("Only ", length(null_coefs), " of ", n_perms, " permutations ",
      "produced a valid null coefficient (need >= 10); returning NA. This ",
      "usually means too few samples reach min_cells_per_sample.", call. = FALSE)
    return(list(null_coefs = null_coefs, perm_pval = NA_real_,
                permutation_pool = permutation_pool))
  }

  perm_pval <- (sum(abs(null_coefs) >= abs(observed_coef)) + 1) / (length(null_coefs) + 1)

  list(null_coefs = null_coefs, perm_pval = perm_pval,
       permutation_pool = permutation_pool)
}


#' Run permutation tests for multiple genes
#'
#' Batch version of \code{\link{run_permutation_test}} that loops over a vector
#' of genes and returns a \code{data.table} of gene-level permutation p-values.
#' Used internally by \code{\link{run_ripple}} to validate the top significant
#' distance-expression gradients.
#'
#' @param genes Character vector. Gene names to test.
#' @param count_matrix Sparse or dense matrix. Raw count matrix (genes x cells).
#' @param target_barcodes Character vector. Barcodes of target cells.
#' @param coords_target Numeric matrix (n_target x 2). Coordinates of target
#'   cells.
#' @param coords_all Numeric matrix (n_all x 2). Coordinates of ALL cells.
#' @param sample_ids_target Character vector. Sample IDs for target cells.
#' @param sample_ids_all Character vector. Sample IDs for all cells.
#' @param query_per_sample Named integer vector. Number of query cells per
#'   sample.
#' @param observed_coefs Named numeric vector. Observed combined coefficients
#'   per gene (names = gene names).
#' @param n_perms Integer. Number of permutations per gene.
#' @param k_neighbors Integer. Number of nearest neighbors for distance
#'   calculation.
#' @param max_distance_um Numeric. Distance cap in micrometers; farther target
#'   cells remain in the fit at this distance.
#' @param min_cells_per_sample Integer. Minimum target cells per sample for
#'   GLM fitting.
#' @param min_expr_cells Integer. Minimum expressing cells for GLM fitting.
#' @param total_counts_target Numeric vector. Total UMI counts per target cell
#'   (for Poisson offset).
#' @inheritParams run_permutation_test
#'
#' @return A \code{data.table} with columns \code{gene}, \code{perm_pval},
#'   and \code{permutation_pool}.
#'
#' @examples
#' \dontrun{
#' perm_dt <- run_permutation_tests(
#'   genes = c("Cxcl12", "Ccl21a"),
#'   count_matrix = counts,
#'   target_barcodes = barcodes,
#'   coords_target = target_xy,
#'   coords_all = all_xy,
#'   sample_ids_target = target_samples,
#'   sample_ids_all = all_samples,
#'   query_per_sample = c(s1 = 50, s2 = 60),
#'   target_mask_all = all_celltypes == "T_cell", # aligned with all_xy
#'   observed_coefs = c(Cxcl12 = -0.005, Ccl21a = -0.003),
#'   n_perms = 500,
#'   k_neighbors = 1,
#'   max_distance_um = 200,
#'   min_cells_per_sample = 30,
#'   min_expr_cells = 5,
#'   total_counts_target = total_counts
#' )
#' }
#'
#' @importFrom RANN nn2
#' @importFrom stats glm poisson median
#' @importFrom data.table data.table rbindlist
#' @export
run_permutation_tests <- function(genes, count_matrix, target_barcodes,
                                  coords_target, coords_all,
                                  sample_ids_target, sample_ids_all,
                                  query_per_sample, observed_coefs,
                                  n_perms, k_neighbors, max_distance_um,
                                  min_cells_per_sample, min_expr_cells,
                                  total_counts_target,
                                  permutation_pool = c("non_target", "all"),
                                  target_mask_all = NULL) {
  permutation_pool <- match.arg(permutation_pool)
  pool <- .prepare_permutation_pool(coords_all, sample_ids_all, query_per_sample,
                                    target_mask_all, permutation_pool)
  coords_all <- pool$coords
  sample_ids_all <- pool$sample_ids
  unique_samples <- names(query_per_sample)

  results <- lapply(genes, function(g) {
    count_vec <- as.numeric(count_matrix[g, target_barcodes])
    obs_coef <- observed_coefs[g]

    null_coefs <- numeric(n_perms)
    for (i in seq_len(n_perms)) {
      # Stratified draw AND stratified search. See run_permutation_test() for
      # why pooling the rbind-ed draws defeats the test: the null inherits the
      # same cross-sample defect as a pooled observed statistic.
      perm_distances <- rep(NA_real_, nrow(coords_target))
      n_pseudo_total <- 0L

      for (samp in unique_samples) {
        n_to_sample <- query_per_sample[samp]
        if (is.na(n_to_sample) || n_to_sample <= 0) next

        samp_coords <- coords_all[sample_ids_all == samp, , drop = FALSE]
        if (nrow(samp_coords) < n_to_sample) next

        pseudo <- samp_coords[sample(nrow(samp_coords), n_to_sample), ,
          drop = FALSE
        ]
        n_pseudo_total <- n_pseudo_total + nrow(pseudo)

        tgt_idx <- which(sample_ids_target == samp)
        if (!length(tgt_idx)) next

        eff_k <- min(k_neighbors, nrow(pseudo))
        nn_res <- RANN::nn2(pseudo, coords_target[tgt_idx, , drop = FALSE],
          k = eff_k
        )
        d_s <- if (eff_k == 1) {
          as.vector(nn_res$nn.dists)
        } else {
          rowMeans(nn_res$nn.dists)
        }
        perm_distances[tgt_idx] <- pmin(d_s, max_distance_um)
      }

      if (n_pseudo_total < 5) {
        null_coefs[i] <- NA
        next
      }

      coefs <- numeric(length(unique_samples))
      ses <- numeric(length(unique_samples))
      for (j in seq_along(unique_samples)) {
        samp <- unique_samples[j]
        idx <- which(sample_ids_target == samp)
        if (length(idx) >= min_cells_per_sample) {
          samp_counts <- count_vec[idx]
          samp_dist <- perm_distances[idx]
          samp_log_total <- log(total_counts_target[idx])
          if (sum(samp_counts > 0) >= min_expr_cells) {
            fit <- tryCatch(
              {
                suppressWarnings(stats::glm(
                  samp_counts ~ samp_dist + offset(samp_log_total),
                  family = stats::poisson
                ))
              },
              error = function(e) NULL
            )
            if (!is.null(fit) && fit$converged) {
              cs <- summary(fit)$coefficients
              if (nrow(cs) >= 2) {
                coefs[j] <- cs[2, "Estimate"]
                ses[j] <- cs[2, "Std. Error"]
              }
            }
          }
        }
      }

      # Null statistic must match the observed median_coef (see
      # run_permutation_test). ses > 0 marks samples whose GLM converged.
      valid <- !is.na(coefs) & !is.na(ses) & ses > 0
      if (sum(valid) >= 2) {
        null_coefs[i] <- stats::median(coefs[valid])
      } else {
        null_coefs[i] <- NA
      }
    }

    null_coefs <- null_coefs[!is.na(null_coefs)]
    if (length(null_coefs) < 10) {
      perm_pval <- NA_real_
    } else {
      perm_pval <- (sum(abs(null_coefs) >= abs(obs_coef)) + 1) /
        (length(null_coefs) + 1)
    }

    data.table::data.table(gene = g, perm_pval = perm_pval,
                          permutation_pool = permutation_pool)
  })

  out <- data.table::rbindlist(results)
  n_na <- sum(is.na(out$perm_pval))
  if (n_na > 0) {
    warning(n_na, " of ", nrow(out), " gene(s) got an NA permutation p-value ",
      "(fewer than 10 valid permutations, usually too few samples reaching ",
      "min_cells_per_sample).", call. = FALSE)
  }
  out
}


#' Merge GPU permutation results into meta-analysis CSVs
#'
#' Reads GPU-produced \code{permutation_pvals.csv} files from per-celltype
#' result directories and merges them into the corresponding
#' \code{meta_analysis_results.csv} files.
#'
#' @param results_dir Character. Path to the analysis results directory
#'   (e.g., \code{"./results/spatial_analysis_Tumor/hymy_distance_correlation_v2"}).
#'   Must contain a \code{per_celltype/} subdirectory.
#'
#' @return Invisible integer. Number of cell types successfully merged.
#'
#' @details For each cell type directory under \code{results_dir/per_celltype/}:
#' \enumerate{
#'   \item Reads \code{permutation_pvals.csv} (GPU output with gene and perm_pval columns).
#'   \item Reads \code{meta_analysis_results.csv} (existing R output).
#'   \item Replaces any existing \code{perm_pval} column with GPU results.
#'   \item Replaces the recorded \code{permutation_pool} with the imported
#'     value, or \code{"unspecified"} if the imported file does not record it.
#'   \item Overwrites \code{meta_analysis_results.csv} with updated data.
#' }
#'
#' Cell types missing either file are skipped with a message.
#'
#' @examples
#' \dontrun{
#' n_merged <- merge_permutation_results(
#'   "./results/spatial_analysis_Tumor/hymy_distance_correlation_v2"
#' )
#' }
#'
#' @importFrom data.table fread fwrite setDT
#' @export
merge_permutation_results <- function(results_dir) {
  ct_base <- file.path(results_dir, "per_celltype")

  if (!dir.exists(ct_base)) {
    stop("Per-celltype directory not found: ", ct_base)
  }

  cell_types <- basename(list.dirs(ct_base, recursive = FALSE))
  message("Found ", length(cell_types), " cell type directories")

  n_merged <- 0L

  for (ct in cell_types) {
    ct_dir <- file.path(ct_base, ct)
    meta_file <- file.path(ct_dir, "meta_analysis_results.csv")
    perm_file <- file.path(ct_dir, "permutation_pvals.csv")

    # Check meta-analysis results exist
    if (!file.exists(meta_file)) {
      message("  [SKIP] ", ct, ": meta_analysis_results.csv not found")
      next
    }

    # Check GPU permutation results exist
    if (!file.exists(perm_file)) {
      message("  [SKIP] ", ct, ": permutation_pvals.csv not found")
      next
    }

    meta <- data.table::fread(meta_file)
    perm <- data.table::fread(perm_file)

    # Validate columns
    if (!"gene" %in% names(perm) || !"perm_pval" %in% names(perm)) {
      message("  [ERROR] ", ct, ": permutation_pvals.csv missing gene/perm_pval columns")
      next
    }

    # Imported p-values must not retain the previous run's pool label.
    if (!"permutation_pool" %in% names(perm)) {
      perm[, permutation_pool := "unspecified"]
    }
    if ("permutation_pool" %in% names(meta)) {
      meta[, permutation_pool := NULL]
    }
    # Remove old perm_pval column and merge new one
    if ("perm_pval" %in% names(meta)) {
      meta[, perm_pval := NULL]
    }
    meta <- merge(meta, perm[, .(gene, perm_pval, permutation_pool)],
                  by = "gene", all.x = TRUE)
    data.table::setDT(meta)

    # Summary stats
    n_tested <- sum(!is.na(perm$perm_pval))
    n_sig <- sum(perm$perm_pval < 0.05, na.rm = TRUE)
    n_genes <- nrow(meta)

    # Save updated meta-analysis results
    data.table::fwrite(meta, meta_file)

    message(sprintf(
      "  [OK] %s: %d/%d genes with perm_pval (%d significant at p<0.05)",
      ct, n_tested, n_genes, n_sig
    ))
    n_merged <- n_merged + 1L
  }

  message(sprintf("\nMerged %d cell types", n_merged))
  invisible(n_merged)
}
