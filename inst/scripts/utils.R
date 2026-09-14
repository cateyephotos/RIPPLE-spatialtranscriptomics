#' =============================================================================
#' Shared utilities for RIPPLE spatial analysis pipeline
#'
#' This module provides common functions for:
#' - Data loading and preprocessing
#' - Spatial neighbor graph operations
#' - Statistical tests and permutation analysis
#' - Visualization helpers
#'
#' Author: CMM Project
#' =============================================================================

# Source lightweight config (platform detection + env var resolution)
# This must come before package loads so config vars are available immediately.
.utils_dir <- local({
  # Try sys.frame ofile — iterate from innermost frame outward to find the
  # source() call for THIS file (utils.R), not the outer calling script.
  n <- sys.nframe()
  for (i in rev(seq_len(n))) {
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  }
  # Try commandArgs for Rscript (when utils.R is the main script)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  # Fallback
  getwd()
})
source(file.path(.utils_dir, "config.R"))

suppressPackageStartupMessages({
  library(Seurat)
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(RANN)  # For fast nearest neighbor search
  library(BiocNeighbors)  # For complete radius neighborhoods
  library(Matrix)
  library(scales)
  library(rstatix)  # For tidy statistical tests (matches IMC pipeline)
  library(dplyr)    # For tidy data manipulation (used with rstatix)
  library(tidyr)    # For pivot_wider/longer
})

# ------------------------------------------------------------------------------
# Spatial Analysis Parameters
# ------------------------------------------------------------------------------
PROXIMITY_THRESHOLD_UM <- 50  # um - paracrine signaling range (~3-5 cell diameters)

# Target cell type for spatial NicheNet analysis
# Can be overridden by environment variable: TARGET_TYPE
# Options: "LEC" (lymphatic endothelial cells), "FRC" (fibroblastic reticular cells)
TARGET_TYPE <- Sys.getenv("TARGET_TYPE", unset = "LEC")
message(sprintf(">>> Target cell type for proximity analysis: %s", TARGET_TYPE))

# Input data paths
if (nchar(INPUT_PATH) > 0) {
  # External mode: INPUT_PATH points directly to the Seurat .rds

  SEURAT_PATH <- INPUT_PATH

  # ADATA_PATH: check env var first, otherwise look for .h5ad alongside .rds
  ADATA_PATH_ENV <- Sys.getenv("ADATA_PATH", unset = "")
  if (nchar(ADATA_PATH_ENV) > 0) {
    ADATA_PATH <- ADATA_PATH_ENV
  } else {
    .h5ad_candidates <- list.files(dirname(INPUT_PATH), pattern = "\\.h5ad$",
                                   full.names = TRUE)
    ADATA_PATH <- if (length(.h5ad_candidates) > 0) .h5ad_candidates[1] else ""
  }
} else {
  # Legacy fallback paths (relative to RIPPLE_BASE_PATH)
  SEURAT_PATH <- file.path(MXENIUM_ROOT, "results", "cell_type_assignment",
                           "HyMy_annotation", "seurat_xenium_filtered.rds")
  ADATA_PATH_ENV <- Sys.getenv("ADATA_PATH", unset = "")
  if (nchar(ADATA_PATH_ENV) > 0) {
    ADATA_PATH <- ADATA_PATH_ENV
  } else {
    ADATA_PATH <- file.path(MXENIUM_ROOT, "results", "sopa", "adata.h5ad")
  }
  HYMY_ANNOTATION_PATH <- file.path(
    MXENIUM_ROOT, "results", "cell_type_assignment",
    "HyMy_annotation", "cell_type_with_HyMy_GMM.csv"
  )
  SPACEL_DIR <- file.path(MXENIUM_ROOT, "results", "spacel")
}

# Plotting defaults
theme_set(theme_bw(base_size = 11))

# =============================================================================
# Cell Type Definitions
# =============================================================================

#' Query cell type signature genes (for contamination checking)
QUERY_SIGNATURE_GENES_ENV <- Sys.getenv("QUERY_SIGNATURE_GENES", unset = "")
if (nchar(QUERY_SIGNATURE_GENES_ENV) > 0) {
  QUERY_SIGNATURE <- trimws(strsplit(QUERY_SIGNATURE_GENES_ENV, ",")[[1]])
} else if (QUERY_CELLTYPE %in% c("HyMy_GMM", "IL1B_myeloid")) {
  QUERY_SIGNATURE <- c("Il1b", "S100a9", "S100a8", "Cxcl2", "C5ar1", "Ccr1", "Csf3r",
                        "Trem1", "Il1r2", "Tnfaip2", "Ptgs2", "Nlrp3", "Cd14", "Itgam",
                        "Acod1", "Pilra")
} else {
  QUERY_SIGNATURE <- character(0)
  message(">>> No query signature genes defined; contamination check will be skipped")
}
HYMY_SIGNATURE <- QUERY_SIGNATURE  # backward compat alias

#' Myeloid cell types for comparison (only defined for HyMy/IL1B legacy modes)
if (QUERY_CELLTYPE %in% c("HyMy_GMM", "IL1B_myeloid")) {
  if (ANNOTATION_LEVEL == "L1") {
    # L1: No HyMy_GMM, IL1B_myeloid contains the HyMy cells
    MYELOID_CELL_TYPES <- c(
      "Macrophages", "Monocyte", "IL1B_myeloid",
      "cDC1", "cDC2", "mature_migDC", "pDC"
    )
    # For segregation analysis: compare IL1B_myeloid vs others (not itself)
    SEGREGATION_COMPARISON_TYPES <- c("Monocyte", "Macrophages")
  } else {
    # HyMy: Includes HyMy_GMM as distinct type
    MYELOID_CELL_TYPES <- c(
      "HyMy_GMM", "Macrophages", "Monocyte", "IL1B_myeloid",
      "cDC1", "cDC2", "mature_migDC", "pDC"
    )
    # For segregation analysis: compare HyMy vs multiple myeloid types
    SEGREGATION_COMPARISON_TYPES <- c("Monocyte", "IL1B_myeloid", "Macrophages")
  }
} else {
  # Non-legacy query types: no predefined myeloid lists
  MYELOID_CELL_TYPES <- character(0)
  SEGREGATION_COMPARISON_TYPES <- character(0)
}

#' T cell subsets
T_CELL_SUBSETS <- c(
  "Naive_CD8", "Naive_CD4", "Activated_CD8", "Cytotoxic_CD8",
  "Tpex", "Tfh", "Treg", "gdT_cell"
)

#' Lymphatic endothelial cells
LEC_TYPES <- c("LEC")


# =============================================================================
# Data Loading Functions
# =============================================================================

#' Load Seurat object
#'
#' Uses existing Seurat object from HyMy annotation pipeline
#' (created by annotate_HyMy_with_UCell.R)
#'
#' The object already contains:
#' - cell_type_with_HyMy: HyMy-annotated cell types
#' - spatial_x, spatial_y: spatial coordinates
#' - HyMy_UCell: UCell signature scores
#' - sample_id, group, experiment: metadata
#'
#' @param path Path to RDS file. Defaults to SEURAT_PATH.
#' @return Seurat object
load_seurat <- function(path = NULL) {
  path <- if (is.null(path)) SEURAT_PATH else path
  message("Loading Seurat object from ", path, "...")

  if (!file.exists(path)) {
    stop("Seurat file not found at: ", path, "\n",
         "Run annotate_HyMy_with_UCell.R first to create the Seurat object.")
  }

  obj <- readRDS(path)
  message(sprintf("Loaded %s cells x %s genes", ncol(obj), nrow(obj)))

  # Verify cell type column exists
  if (!CELLTYPE_COL %in% colnames(obj@meta.data)) {
    # Not a hard error — downstream code (e.g. HyMy merge) may add it
    message(sprintf("  Note: cell type column '%s' not yet in metadata (may be added later)",
                    CELLTYPE_COL))
  }

  return(obj)
}


#' Load HyMy cell type annotations (GMM-thresholded)
#'
#' @param path Path to CSV file
#' @return data.table with cell_id and cell_type_with_HyMy columns
load_hymy_annotations <- function(path = NULL) {
  path <- if (is.null(path)) HYMY_ANNOTATION_PATH else path
  message("Loading HyMy annotations from ", path, "...")

  dt <- fread(path)
  # Standardize column names
  setnames(dt, c("cell_id", "cell_type_with_HyMy_GMM"), c("cell_id", "cell_type_with_HyMy"))
  # Remove quotes if present

  dt[, cell_id := gsub('"', '', cell_id)]
  dt[, cell_type_with_HyMy := gsub('"', '', cell_type_with_HyMy)]

  message(sprintf("Loaded annotations for %s cells", nrow(dt)))
  message(sprintf("  HyMy_GMM cells: %d", sum(dt$cell_type_with_HyMy == "HyMy_GMM")))
  message(sprintf("  IL1B_myeloid cells: %d", sum(dt$cell_type_with_HyMy == "IL1B_myeloid")))
  return(dt)
}


#' Load spatial domains from SPACEL
#'
#' @param experiment Experiment name or pattern
#' @return data.table with spatial domain assignments
load_spatial_domains <- function(experiment = "all") {
  pattern <- sprintf("spatial_domains__%s.*\\.csv", experiment)
  domain_files <- list.files(SPACEL_DIR, pattern = pattern, full.names = TRUE)

  if (length(domain_files) == 0) {
    warning(sprintf("No domain files found matching %s", pattern))
    return(data.table())
  }

  # Use most recent file
  domain_file <- sort(domain_files, decreasing = TRUE)[1]
  message("Loading spatial domains from ", domain_file, "...")

  dt <- fread(domain_file)
  return(dt)
}


#' Merge HyMy annotations into Seurat object
#'
#' @param obj Seurat object
#' @param hymy_df data.table with HyMy annotations (optional, will load if NULL)
#' @return Seurat object with cell_type_with_HyMy in metadata
merge_hymy_annotations <- function(obj, hymy_df = NULL) {
  if (is.null(hymy_df)) {
    hymy_df <- load_hymy_annotations()
  }

  # Match by cell barcode
  hymy_df <- as.data.frame(hymy_df)
  rownames(hymy_df) <- hymy_df$cell_id

  common_cells <- intersect(colnames(obj), hymy_df$cell_id)
  message(sprintf("Merging annotations for %s cells", length(common_cells)))

  obj$cell_type_with_HyMy <- hymy_df[colnames(obj), "cell_type_with_HyMy"]

  return(obj)
}


# =============================================================================
# Spatial Coordinate Utilities
# =============================================================================

#' Extract spatial coordinates from Seurat object
#'
#' @param obj Seurat object
#' @param image Name of spatial image/assay (for Xenium data)
#' @return Matrix with columns x, y
get_spatial_coords <- function(obj, image = NULL) {
  # First try spatial_x, spatial_y (from annotation pipeline)
  if (all(c("spatial_x", "spatial_y") %in% colnames(obj@meta.data))) {
    coords <- obj@meta.data[, c("spatial_x", "spatial_y")]
    colnames(coords) <- c("x", "y")
    return(as.matrix(coords))
  }

  # Try to get coordinates from Images slot (Xenium/Visium)
  if (length(Images(obj)) > 0) {
    if (is.null(image)) {
      image <- Images(obj)[1]
    }
    coords <- GetTissueCoordinates(obj, image = image)
    return(as.matrix(coords[, c("x", "y")]))
  }

  # Try other metadata column variants
  if (all(c("x_centroid", "y_centroid") %in% colnames(obj@meta.data))) {
    return(as.matrix(obj@meta.data[, c("x_centroid", "y_centroid")]))
  }

  if (all(c("spatial_X", "spatial_Y") %in% colnames(obj@meta.data))) {
    coords <- obj@meta.data[, c("spatial_X", "spatial_Y")]
    colnames(coords) <- c("x", "y")
    return(as.matrix(coords))
  }

  # Try reductions
  if ("spatial" %in% names(obj@reductions)) {
    return(Embeddings(obj, "spatial")[, 1:2])
  }

  stop("Could not find spatial coordinates in Seurat object")
}


#' Resolve coordinate column names from metadata
#'
#' If X_COL_ENV and Y_COL_ENV are set (from config.R), verifies they exist and
#' returns them.  Otherwise auto-detects by trying common pairs in order.
#'
#' @param meta data.table (or data.frame) with metadata columns
#' @return Character vector of length 2: c(x_col, y_col)
get_coord_columns <- function(meta) {
  col_names <- names(meta)

  # Priority 1: user-specified via env vars

  if (nchar(X_COL_ENV) > 0 && nchar(Y_COL_ENV) > 0) {
    if (!X_COL_ENV %in% col_names)
      stop(sprintf("X_COLUMN '%s' not found in metadata. Available: %s",
                    X_COL_ENV, paste(head(col_names, 20), collapse = ", ")))
    if (!Y_COL_ENV %in% col_names)
      stop(sprintf("Y_COLUMN '%s' not found in metadata. Available: %s",
                    Y_COL_ENV, paste(head(col_names, 20), collapse = ", ")))
    return(c(X_COL_ENV, Y_COL_ENV))
  }

  # Priority 2: auto-detect common pairs
  candidates <- list(
    c("spatial_x", "spatial_y"),
    c("x", "y"),
    c("x_centroid", "y_centroid")
  )
  for (pair in candidates) {
    if (all(pair %in% col_names)) {
      message(sprintf("  Auto-detected coordinate columns: %s, %s", pair[1], pair[2]))
      return(pair)
    }
  }

  stop("Could not find spatial coordinate columns in metadata.\n",
       "  Tried: spatial_x/spatial_y, x/y, x_centroid/y_centroid\n",
       "  Set X_COLUMN and Y_COLUMN env vars to specify custom column names.\n",
       "  Available columns: ", paste(head(col_names, 30), collapse = ", "))
}


# =============================================================================
# Neighbor Graph Functions
# =============================================================================

#' Build k-nearest neighbor graph using RANN
#'
#' @param coords Matrix of spatial coordinates (n x 2)
#' @param k Number of neighbors
#' @return List with indices and distances matrices (n x k)
build_knn_graph <- function(coords, k = 20, sample_ids) {
  if (missing(sample_ids)) {
    stop("sample_ids is required. Spatial neighbours are only meaningful ",
         "within one tissue section: sections routinely share a coordinate ",
         "frame, so a search pooled across samples silently returns ",
         "neighbours from a different section. For genuinely single-sample ",
         "data pass rep(\"s1\", nrow(coords)).", call. = FALSE)
  }
  if (length(sample_ids) != nrow(coords)) {
    stop("sample_ids must have one entry per row of coords (got ",
         length(sample_ids), " for ", nrow(coords), " rows).", call. = FALSE)
  }

  # Partitioned by sample. A spatial neighbour in a different tissue section
  # does not exist physically, and sections routinely share a coordinate
  # frame, so a pooled search returns those non-existent neighbours silently.
  # Returned indices are global row numbers into coords.
  n <- nrow(coords)
  samples <- unique(sample_ids[!is.na(sample_ids)])
  message(sprintf(
    "Building %d-nearest neighbor graph for %d cells in %d sample(s)...",
    k, n, length(samples)))

  indices <- matrix(NA_integer_, nrow = n, ncol = k)
  distances <- matrix(NA_real_, nrow = n, ncol = k)
  short <- character(0)

  for (s in samples) {
    rows <- which(!is.na(sample_ids) & sample_ids == s)
    if (length(rows) < 2) { short <- c(short, as.character(s)); next }
    kk <- min(k, length(rows) - 1L)
    if (kk < k) short <- c(short, as.character(s))
    nn <- nn2(coords[rows, , drop = FALSE], coords[rows, , drop = FALSE],
              k = kk + 1L)
    local_idx <- nn$nn.idx[, -1, drop = FALSE]
    indices[rows, seq_len(kk)] <- matrix(rows[local_idx], nrow = length(rows))
    distances[rows, seq_len(kk)] <- nn$nn.dists[, -1, drop = FALSE]
  }

  if (length(short)) {
    warning("Fewer than k + 1 cells in sample(s): ",
            paste(unique(short), collapse = ", "),
            ". Those rows are padded with NA.", call. = FALSE)
  }
  list(indices = indices, distances = distances)
}


#' Build radius-based neighbor graph
#'
#' @param coords Matrix of spatial coordinates
#' @param radius Search radius in coordinate units (microns)
#' @return List of neighbor indices for each cell
build_radius_graph <- function(coords, radius, sample_ids) {
  if (missing(sample_ids)) {
    stop("sample_ids is required. Spatial neighbours are only meaningful ",
         "within one tissue section: sections routinely share a coordinate ",
         "frame, so a search pooled across samples silently returns ",
         "neighbours from a different section. For genuinely single-sample ",
         "data pass rep(\"s1\", nrow(coords)).", call. = FALSE)
  }
  if (length(sample_ids) != nrow(coords)) {
    stop("sample_ids must have one entry per row of coords (got ",
         length(sample_ids), " for ", nrow(coords), " rows).", call. = FALSE)
  }

  # Search each sample independently and return all neighbors in range.
  n <- nrow(coords)
  samples <- unique(sample_ids[!is.na(sample_ids)])
  message(sprintf("Building radius neighbor graph (r=%s) in %d sample(s)...",
                  radius, length(samples)))

  neighbors <- rep(list(integer(0)), n)
  for (s in samples) {
    rows <- which(!is.na(sample_ids) & sample_ids == s)
    if (length(rows) < 2) next
    sc <- coords[rows, , drop = FALSE]
    nn <- BiocNeighbors::findNeighbors(
      sc, threshold = radius, get.distance = FALSE,
      BNPARAM = BiocNeighbors::KmknnParam(distance = "Euclidean")
    )
    for (j2 in seq_along(rows)) {
      neighbors[[rows[j2]]] <- rows[nn$index[[j2]]]
    }
  }
  neighbors
}


#' Get neighbor cell types for a set of query cells
#'
#' @param cell_types Vector of cell types for all cells
#' @param query_indices Indices of query cells
#' @param knn_result Result from build_knn_graph
#' @return data.table with query_cell, neighbor_cell, neighbor_type
get_neighbor_cell_types <- function(cell_types, query_indices, knn_result) {
  results <- rbindlist(lapply(query_indices, function(i) {
    neighbor_idx <- knn_result$indices[i, ]
    data.table(
      query_cell = i,
      neighbor_cell = neighbor_idx,
      neighbor_type = cell_types[neighbor_idx]
    )
  }))
  return(results)
}


#' Calculate neighbor composition for a cell type
#'
#' @param cell_types Vector of cell types for all cells
#' @param query_cell_type Cell type to analyze neighborhoods of
#' @param knn_result Result from build_knn_graph
#' @return data.table with neighbor type counts and proportions
calculate_neighbor_composition <- function(cell_types, query_cell_type, knn_result) {
  query_idx <- which(cell_types == query_cell_type)

  if (length(query_idx) == 0) {
    warning(sprintf("No cells of type '%s' found", query_cell_type))
    return(data.table())
  }

  # Get all neighbor types
  neighbor_types <- as.vector(knn_result$indices[query_idx, ])
  neighbor_types <- cell_types[neighbor_types]

  # Count and calculate proportions
  counts <- table(neighbor_types)
  dt <- data.table(
    neighbor_type = names(counts),
    count = as.integer(counts),
    proportion = as.numeric(counts) / sum(counts)
  )

  dt[, query_cell_type := query_cell_type]
  setcolorder(dt, c("query_cell_type", "neighbor_type", "count", "proportion"))

  return(dt)
}


# =============================================================================
# Distance Calculations
# =============================================================================

#' Report whether per-sample coordinate frames overlap
#'
#' Mirrors check_coordinate_frames() in R/spatial.R. Compares the sum of
#' per-sample bounding-box areas with the global bounding-box area. Sections
#' laid out in disjoint space sum to at most the global area, so a ratio above
#' 1 means at least two sections occupy the same coordinate region, which is
#' the condition under which a pooled nearest-neighbour search silently
#' returns cross-sample neighbours.
#'
#' @param coords Matrix of spatial coordinates (n x 2)
#' @param sample_ids Vector of length n giving each cell sample
#' @param warn Logical. Emit a warning when overlap is detected.
#' @return Invisibly, a list with ratio, n_overlapping_pairs, n_pairs, overlaps
check_coordinate_frames <- function(coords, sample_ids, target_mask = NULL,
                                    warn = TRUE, severe_fraction = 0.05,
                                    max_cells = 10000L) {
  ok <- !is.na(sample_ids) & !is.na(coords[, 1]) & !is.na(coords[, 2])
  if (!all(ok)) {
    coords <- coords[ok, , drop = FALSE]
    sample_ids <- sample_ids[ok]
    if (!is.null(target_mask)) target_mask <- target_mask[ok]
  }
  samples <- unique(sample_ids)
  if (length(samples) < 2) {
    return(invisible(list(ratio = NA_real_, n_overlapping_pairs = 0L,
                          n_pairs = 0L, overlaps = FALSE,
                          cross_sample_fraction = NA_real_,
                          n_cells_checked = 0L, severe = FALSE)))
  }

  box <- do.call(rbind, lapply(samples, function(s) {
    i <- sample_ids == s
    c(xmin = min(coords[i, 1]), xmax = max(coords[i, 1]),
      ymin = min(coords[i, 2]), ymax = max(coords[i, 2]))
  }))
  areas <- (box[, "xmax"] - box[, "xmin"]) * (box[, "ymax"] - box[, "ymin"])
  global <- (max(coords[, 1]) - min(coords[, 1])) *
    (max(coords[, 2]) - min(coords[, 2]))
  ratio <- if (global > 0) sum(areas) / global else NA_real_

  n_ov <- 0L
  n_pairs <- 0L
  for (i in seq_len(nrow(box) - 1)) {
    for (j in seq(i + 1, nrow(box))) {
      n_pairs <- n_pairs + 1L
      ox <- min(box[i, "xmax"], box[j, "xmax"]) -
        max(box[i, "xmin"], box[j, "xmin"])
      oy <- min(box[i, "ymax"], box[j, "ymax"]) -
        max(box[i, "ymin"], box[j, "ymin"])
      if (isTRUE(ox > 0) && isTRUE(oy > 0)) n_ov <- n_ov + 1L
    }
  }

  # Severity: how often would a pooled search actually cross a sample
  # boundary? Deterministically thinned subsample, so it never consumes random
  # numbers and cannot shift a seeded permutation downstream.
  #
  # Note the area ratio is a ONE-SIDED test. Above 1 it proves overlap by
  # pigeonhole; below 1 it proves nothing, because sections tiled with gaps can
  # hide a perfectly superimposed pair at a ratio near 0.02. The pairwise box
  # intersection is the direct measurement, and this fraction is the severity.
  cross <- NA_real_
  n_checked <- 0L
  if (!is.null(target_mask) && n_ov > 0) {
    tgt <- which(!is.na(target_mask) & target_mask)
    if (length(tgt) > 0) {
      n <- nrow(coords)
      idx <- if (n > max_cells) {
        unique(as.integer(seq.int(1L, n, length.out = max_cells)))
      } else {
        seq_len(n)
      }
      nn <- nn2(coords[tgt, , drop = FALSE], coords[idx, , drop = FALSE], k = 1)
      owner <- sample_ids[tgt][as.vector(nn$nn.idx)]
      cross <- mean(owner != sample_ids[idx])
      n_checked <- length(idx)
    }
  }
  severe <- n_ov > 0 && (is.na(cross) || cross > severe_fraction)

  res <- list(ratio = ratio, n_overlapping_pairs = n_ov, n_pairs = n_pairs,
              overlaps = n_ov > 0, cross_sample_fraction = cross,
              n_cells_checked = n_checked, severe = severe)
  # ANY bounding-box intersection warns; the measured fraction sizes the
  # problem but does not gate whether the user is told.
  if (warn && n_ov > 0) {
    warning(n_ov, " of ", n_pairs, " sample pairs have overlapping coordinate ",
            "bounding boxes, so the sections share a coordinate frame",
            if (is.na(cross)) "" else
              paste0(" and ", round(100 * cross, 1), "% of cells have their ",
                     "nearest target cell in a DIFFERENT sample"),
            ". Any POOLED nearest-neighbour search on these coordinates ",
            "returns cross-sample neighbours. Use ",
            "calculate_distance_to_type_by_sample().", call. = FALSE)
  }
  invisible(res)
}


#' Distance to the nearest cell of a target type, computed within each sample
#'
#' Mirrors calculate_distance_to_type_by_sample() in R/spatial.R. Tissue
#' sections routinely occupy overlapping coordinate ranges, since each section
#' coordinates start near zero in its own frame. A nearest-neighbour search run
#' over cells from more than one sample then silently returns neighbours from a
#' different sample, and no amount of downstream per-sample aggregation repairs
#' it, because the damage happens before aggregation. This partitions the
#' search by sample, so a cell can only ever match a target cell from its own
#' sample.
#'
#' @param coords Matrix of spatial coordinates (n x 2)
#' @param sample_ids Vector of length n giving each cell sample
#' @param target_mask Logical vector of length n, TRUE for valid search targets
#' @param k Number of nearest targets to average over
#' @return Numeric vector of length n in the original row order. Cells in a
#'   sample with no target cells receive NA.
calculate_distance_to_type_by_sample <- function(coords, sample_ids,
                                                 target_mask, k = 1) {
  n <- nrow(coords)
  if (length(sample_ids) != n || length(target_mask) != n) {
    stop("sample_ids and target_mask must each have one entry per row of ",
         "coords (got ", length(sample_ids), ", ", length(target_mask),
         " for ", n, " rows).", call. = FALSE)
  }
  if (anyNA(target_mask)) {
    stop("target_mask contains NA; mask the cell-type comparison NA-safely ",
         "before calling.", call. = FALSE)
  }

  out <- rep(NA_real_, n)
  empty <- character(0)
  for (s in unique(sample_ids)) {
    rows <- which(sample_ids == s)
    tgt <- rows[target_mask[rows]]
    if (!length(tgt)) {
      empty <- c(empty, as.character(s))
      next
    }
    kk <- min(k, length(tgt))
    nn <- nn2(coords[tgt, , drop = FALSE], coords[rows, , drop = FALSE], k = kk)
    out[rows] <- if (kk == 1) as.vector(nn$nn.dists) else rowMeans(nn$nn.dists)
  }
  if (length(empty)) {
    warning("No target cells in sample(s): ", paste(empty, collapse = ", "),
            ". Those cells receive NA distances.", call. = FALSE)
  }
  out
}


#' Calculate distance from each cell to nearest cell of target type
#'
#' @param coords Matrix of spatial coordinates
#' @param cell_types Vector of cell type labels
#' @param target_type Cell type to measure distance to
#' @param sample_ids Vector of length n giving each cell sample. Required; the
#'   search is partitioned by it, so a cell can only ever match a target cell
#'   from its own sample. For single-sample data pass rep("s1", nrow(coords)).
#' @return Named vector of distances
calculate_distance_to_type <- function(coords, cell_types, target_type,
                                       sample_ids) {
  # NA-safe: cell_types == target_type yields NA (not FALSE) for unannotated
  # cells, which would inject NA-coordinate rows into the RANN reference set.
  target_mask <- !is.na(cell_types) & cell_types == target_type

  if (!any(target_mask)) {
    warning(sprintf("No cells of type '%s' found", target_type))
    return(rep(NA_real_, nrow(coords)))
  }

  calculate_distance_to_type_by_sample(
    coords = coords, sample_ids = sample_ids,
    target_mask = target_mask, k = 1
  )
}


# =============================================================================
# Statistical Functions
# =============================================================================

#' Permutation test
#'
#' @param observed Observed test statistic
#' @param null_distribution Vector of null statistics from permutations
#' @param alternative "two.sided", "greater", or "less"
#' @return p-value
permutation_pvalue <- function(observed, null_distribution, alternative = "two.sided") {
  n_perms <- length(null_distribution)

  p <- switch(alternative,
    "two.sided" = sum(abs(null_distribution) >= abs(observed)) / n_perms,
    "greater" = sum(null_distribution >= observed) / n_perms,
    "less" = sum(null_distribution <= observed) / n_perms,
    stop("Unknown alternative: ", alternative)
  )

  # Correct for finite permutations
  p <- max(p, 1 / (n_perms + 1))

  return(p)
}


#' Calculate enrichment scores
#'
#' @param observed Named vector of observed counts
#' @param expected Named vector of expected counts
#' @return data.table with enrichment statistics
calculate_enrichment <- function(observed, expected) {
  # Ensure same order
  common_names <- intersect(names(observed), names(expected))
  observed <- observed[common_names]
  expected <- expected[common_names]

  # Avoid division by zero
  expected_safe <- ifelse(expected == 0, NA_real_, expected)

  data.table(
    category = common_names,
    observed = observed,
    expected = expected,
    enrichment = observed / expected_safe,
    log2_enrichment = log2(observed / expected_safe)
  )
}


#' Shannon entropy
#'
#' @param counts Vector of counts
#' @return Shannon entropy (log2 scale)
shannon_entropy <- function(counts) {
  if (sum(counts) == 0) return(0)

  probs <- counts / sum(counts)
  probs <- probs[probs > 0]  # Remove zeros for log

  -sum(probs * log2(probs))
}


#' Calculate neighborhood entropy for each cell
#'
#' @param cell_types Vector of cell type labels
#' @param knn_result Result from build_knn_graph
#' @return Vector of entropy values
calculate_neighborhood_entropy <- function(cell_types, knn_result) {
  n <- nrow(knn_result$indices)
  entropies <- numeric(n)

  for (i in seq_len(n)) {
    neighbor_types <- cell_types[knn_result$indices[i, ]]
    counts <- table(neighbor_types)
    entropies[i] <- shannon_entropy(counts)
  }

  return(entropies)
}


# =============================================================================
# Gene Scoring Functions
# =============================================================================

#' Score cells for a gene signature (like UCell/AddModuleScore)
#'
#' @param obj Seurat object
#' @param genes Gene list
#' @param name Name for the score
#' @param ctrl Number of control genes
#' @return Seurat object with score added to metadata
score_gene_signature <- function(obj, genes, name, ctrl = 100) {
  # Filter to available genes
  available <- intersect(genes, rownames(obj))
  missing <- setdiff(genes, rownames(obj))

  if (length(missing) > 0) {
    message(sprintf("Note: %d genes not found: %s",
                    length(missing), paste(head(missing, 3), collapse = ", ")))
  }

  if (length(available) < 2) {
    warning("Fewer than 2 genes available for scoring")
    obj@meta.data[[name]] <- NA_real_
    return(obj)
  }

  obj <- AddModuleScore(
    obj,
    features = list(available),
    name = name,
    ctrl = ctrl,
    seed = 42
  )

  # AddModuleScore appends "1" to name
  colnames(obj@meta.data)[colnames(obj@meta.data) == paste0(name, "1")] <- name

  return(obj)
}


#' Score multiple gene modules
#'
#' @param obj Seurat object
#' @param modules Named list of gene vectors
#' @param prefix Prefix for score names
#' @return Seurat object with scores added
score_multiple_modules <- function(obj, modules, prefix = "module_") {
  for (module_name in names(modules)) {
    score_name <- paste0(prefix, module_name)
    message("Scoring module: ", module_name)
    obj <- score_gene_signature(obj, modules[[module_name]], score_name)
  }
  return(obj)
}


# =============================================================================
# Visualization Helpers
# =============================================================================

#' Create spatial scatter plot (legacy - for single combined plot)
#'
#' @param coords Matrix of coordinates (n x 2)
#' @param color_by Vector to color points by
#' @param point_size Point size
#' @param alpha Transparency
#' @param palette Named vector of colors (for categorical)
#' @param title Plot title
#' @return ggplot object
plot_spatial_scatter <- function(coords, color_by, point_size = 0.5, alpha = 0.5,
                                  palette = NULL, title = NULL) {
  df <- data.frame(x = coords[, 1], y = coords[, 2], color = color_by)

  p <- ggplot(df, aes(x = x, y = y, color = color)) +
    geom_point(size = point_size, alpha = alpha) +
    coord_fixed() +
    labs(x = "X (µm)", y = "Y (µm)", title = title) +
    theme_minimal() +
    theme(legend.position = "right")

  if (is.factor(color_by) || is.character(color_by)) {
    if (!is.null(palette)) {
      p <- p + scale_color_manual(values = palette)
    } else {
      p <- p + scale_color_brewer(palette = "Set3")
    }
  } else {
    p <- p + scale_color_viridis_c()
  }

  return(p)
}


#' Create spatial plot for a single sample (recommended approach)
#'
#' Uses theme_void() for clean spatial visualization.
#' This is the preferred function for spatial plots.
#'
#' @param data data.table with x, y (or spatial_x, spatial_y) and color column
#' @param color_var Name of column to color by
#' @param title Plot title
#' @param palette Named vector of colors (for categorical)
#' @param point_size Point size
#' @param alpha Transparency
#' @param continuous Whether color variable is continuous
#' @param highlight_value If set, highlight only this value, grey out others
#' @return ggplot object
plot_spatial_single <- function(data, color_var, title = NULL,
                                 palette = NULL, point_size = 0.5, alpha = 0.8,
                                 continuous = FALSE, highlight_value = NULL) {

  # Handle coordinate column names
  if ("spatial_x" %in% names(data) && !"x" %in% names(data)) {
    data <- copy(data)
    data[, x := spatial_x]
    data[, y := spatial_y]
  }

  # If highlighting specific value, mask others
  if (!is.null(highlight_value)) {
    data <- copy(data)
    data[, plot_color := fifelse(get(color_var) == highlight_value,
                                  highlight_value, "Other")]
    data[, plot_color := factor(plot_color, levels = c(highlight_value, "Other"))]
    color_var <- "plot_color"

    if (is.null(palette)) {
      palette <- c("#E74C3C", "lightgrey")
      names(palette) <- c(highlight_value, "Other")
    } else {
      palette <- c(palette[highlight_value], "lightgrey")
      names(palette) <- c(highlight_value, "Other")
    }
  }

  p <- ggplot(data, aes(x = x, y = y, color = .data[[color_var]])) +
    geom_point(size = point_size, alpha = alpha) +
    coord_fixed() +
    theme_void() +
    theme(
      plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
      legend.position = "right",
      legend.text = element_text(size = 8)
    ) +
    labs(title = title)

  # Apply color scale
  if (continuous) {
    p <- p + scale_color_viridis_c(option = "magma", name = color_var)
  } else if (!is.null(palette)) {
    p <- p + scale_color_manual(values = palette, name = "Cell Type")
  }

  return(p)
}


#' Create spatial plots faceted by sample (proper approach)
#'
#' Creates individual plots per sample and combines with wrap_plots().
#' This avoids the coord_fixed() + facet_wrap(scales="free") incompatibility.
#'
#' @param data data.table with x, y, sample column, and color column
#' @param color_var Name of column to color by
#' @param sample_col Name of sample column (default "sample")
#' @param title Overall plot title
#' @param palette Named vector of colors
#' @param point_size Point size
#' @param alpha Transparency
#' @param ncol Number of columns in layout (default: auto)
#' @param continuous Whether color variable is continuous
#' @return patchwork combined plot
plot_spatial_by_sample <- function(data, color_var, sample_col = "sample",
                                    title = NULL, palette = NULL,
                                    point_size = 0.3, alpha = 0.6,
                                    ncol = NULL, continuous = FALSE) {

  # Handle coordinate column names
  if ("spatial_x" %in% names(data) && !"x" %in% names(data)) {
    data <- copy(data)
    data[, x := spatial_x]
    data[, y := spatial_y]
  }

  samples <- unique(data[[sample_col]])
  n_samples <- length(samples)

  if (is.null(ncol)) {
    ncol <- min(4, ceiling(sqrt(n_samples)))
  }

  # Create individual plots
  plot_list <- lapply(samples, function(samp) {
    samp_data <- data[get(sample_col) == samp]

    p <- ggplot(samp_data, aes(x = x, y = y, color = .data[[color_var]])) +
      geom_point(size = point_size, alpha = alpha) +
      coord_fixed() +
      theme_void() +
      theme(
        plot.title = element_text(size = 9, hjust = 0.5),
        legend.position = "none"
      ) +
      labs(title = samp)

    if (continuous) {
      p <- p + scale_color_viridis_c(option = "magma")
    } else if (!is.null(palette)) {
      p <- p + scale_color_manual(values = palette)
    }

    return(p)
  })

  # Combine with patchwork
  combined <- wrap_plots(plot_list, ncol = ncol)

  if (!is.null(title)) {
    combined <- combined + plot_annotation(title = title)
  }

  # Add a shared legend
  # Create a dummy plot just for the legend
  if (!continuous && !is.null(palette)) {
    legend_data <- data.table(
      x = 1, y = 1,
      color = factor(names(palette), levels = names(palette))
    )
    legend_plot <- ggplot(legend_data, aes(x = x, y = y, color = color)) +
      geom_point(size = 3) +
      scale_color_manual(values = palette, name = "Cell Type") +
      theme_void() +
      theme(legend.position = "right")

    # Extract legend
    legend_grob <- cowplot::get_legend(legend_plot)

    # This approach requires cowplot - if not available, skip legend
    if (requireNamespace("cowplot", quietly = TRUE)) {
      combined <- combined | wrap_elements(legend_grob)
    }
  }

  return(combined)
}


#' Create violin plot for group comparisons
#'
#' @param data data.frame or data.table
#' @param x Column name for x-axis (groups)
#' @param y Column name for y-axis (values)
#' @param fill Column name for fill color (optional)
#' @param palette Named vector of colors
#' @param title Plot title
#' @return ggplot object
plot_violin <- function(data, x, y, fill = NULL, palette = NULL, title = NULL) {
  if (is.null(fill)) fill <- x

  p <- ggplot(data, aes(x = .data[[x]], y = .data[[y]], fill = .data[[fill]])) +
    geom_violin(scale = "width", trim = TRUE) +
    geom_boxplot(width = 0.1, outlier.size = 0.5, fill = "white") +
    labs(title = title) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  if (!is.null(palette)) {
    p <- p + scale_fill_manual(values = palette)
  }

  return(p)
}


#' Create bar plot for neighbor composition
#'
#' @param data data.table with columns: query_cell_type, neighbor_type, proportion
#' @param title Plot title
#' @return ggplot object
plot_neighbor_composition <- function(data, title = NULL) {
  ggplot(data, aes(x = neighbor_type, y = proportion, fill = neighbor_type)) +
    geom_col() +
    facet_wrap(~ query_cell_type, scales = "free_x") +
    labs(x = "Neighbor Cell Type", y = "Proportion", title = title) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
}


#' Create enrichment bar plot
#'
#' @param data data.table from calculate_enrichment
#' @param title Plot title
#' @return ggplot object
plot_enrichment <- function(data, title = NULL) {
  data <- copy(data)
  data[, category := factor(category, levels = category[order(log2_enrichment)])]

  ggplot(data, aes(x = category, y = log2_enrichment, fill = log2_enrichment > 0)) +
    geom_col() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_fill_manual(values = c("TRUE" = "firebrick", "FALSE" = "steelblue"),
                      guide = "none") +
    coord_flip() +
    labs(x = NULL, y = "log2(Enrichment)", title = title) +
    theme_bw()
}


# =============================================================================
# I/O Utilities
# =============================================================================

#' Ensure directory exists
ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  return(path)
}


#' Save data.table to CSV
save_results <- function(dt, output_dir, filename) {
  ensure_dir(output_dir)
  filepath <- file.path(output_dir, paste0(filename, ".csv"))
  fwrite(dt, filepath)
  message("Saved: ", filepath)
}


#' Save ggplot to multiple formats
save_plot <- function(p, output_dir, filename, width = 8, height = 6,
                      formats = c("png", "pdf")) {
  ensure_dir(output_dir)

  for (fmt in formats) {
    filepath <- file.path(output_dir, paste0(filename, ".", fmt))
    ggsave(filepath, p, width = width, height = height, dpi = 300)
    message("Saved: ", filepath)
  }
}


#' Infer condition from sample name (LEGACY)
#'
#' Only used when CONDITION_COL is not set. Assumes the legacy internal naming
#' convention (naive_*, TDLN_*). External users should set CONDITION_COLUMN
#' and CONDITION_VALUE env vars instead.
#'
#' @param sample_name Sample identifier
#' @return "naive" or "TDLN"
get_condition <- function(sample_name) {

  sample_lower <- tolower(sample_name)
  # IMPORTANT: Check "tdln" BEFORE "naive" because sample names like
  # "tdln_vs_naive__m18" contain BOTH strings - we want to match TDLN
  # However, this is a fallback - prefer using the `group` column directly
  if (grepl("^tdln$", sample_lower) || grepl("^tdln_", sample_lower)) {
    return("TDLN")
  } else if (grepl("^naive$", sample_lower) || grepl("^naive_", sample_lower)) {
    return("naive")
  } else if (grepl("tdln", sample_lower) && !grepl("naive", sample_lower)) {
    return("TDLN")
  } else if (grepl("naive", sample_lower) && !grepl("tdln", sample_lower)) {
    return("naive")
  } else {
    # Both or neither - cannot determine
    warning("Could not infer condition from: ", sample_name)
    return("unknown")
  }
}


# =============================================================================
# Test
# =============================================================================

if (sys.nframe() == 0) {
  message("Testing utility functions...")
  message("\nProject root: ", PROJECT_ROOT)
  message("Seurat path exists: ", file.exists(SEURAT_PATH))
  if (exists("HYMY_ANNOTATION_PATH")) {
    message("HyMy annotation exists: ", file.exists(HYMY_ANNOTATION_PATH))
  }
  message("\n✓ Utility module loaded successfully")
}
