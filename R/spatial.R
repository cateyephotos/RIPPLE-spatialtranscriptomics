#' @title Spatial Analysis Functions
#'
#' @description Functions for spatial coordinate handling, neighbor graph
#'   construction, distance calculations, and neighborhood composition analysis.
#'
#' @name spatial
NULL

#' Auto-detect or verify coordinate columns
#'
#' Resolves spatial coordinate column names from metadata. If explicit column
#' names are provided, verifies they exist. Otherwise, auto-detects by trying
#' common spatial coordinate column pairs in order.
#'
#' @param meta A \code{data.table} or \code{data.frame} with metadata columns.
#' @param x_col Character or NULL. Explicit X coordinate column name to verify.
#' @param y_col Character or NULL. Explicit Y coordinate column name to verify.
#'
#' @return A character vector of length 2: \code{c(x_col, y_col)}.
#'
#' @details Auto-detection tries the following pairs in order:
#' \enumerate{
#'   \item \code{spatial_x}, \code{spatial_y}
#'   \item \code{x}, \code{y}
#'   \item \code{x_centroid}, \code{y_centroid}
#' }
#'
#' @examples
#' \dontrun{
#' meta <- data.table(spatial_x = runif(10), spatial_y = runif(10))
#' coords <- get_coord_columns(meta)
#' # Returns c("spatial_x", "spatial_y")
#'
#' # Or specify explicitly:
#' coords <- get_coord_columns(meta, x_col = "spatial_x", y_col = "spatial_y")
#' }
#'
#' @export
get_coord_columns <- function(meta, x_col = NULL, y_col = NULL) {
  col_names <- names(meta)


  # Priority 1: user-specified via arguments
  if (!is.null(x_col) && !is.null(y_col) &&
    nzchar(x_col) && nzchar(y_col)) {
    if (!x_col %in% col_names) {
      stop(sprintf(
        "X column '%s' not found in metadata. Available: %s",
        x_col, paste(head(col_names, 20), collapse = ", ")
      ))
    }
    if (!y_col %in% col_names) {
      stop(sprintf(
        "Y column '%s' not found in metadata. Available: %s",
        y_col, paste(head(col_names, 20), collapse = ", ")
      ))
    }
    return(c(x_col, y_col))
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

  stop(
    "Could not find spatial coordinate columns in metadata.\n",
    "  Tried: spatial_x/spatial_y, x/y, x_centroid/y_centroid\n",
    "  Provide x_col and y_col arguments to specify custom column names.\n",
    "  Available columns: ", paste(head(col_names, 30), collapse = ", ")
  )
}


#' Build k-nearest neighbor graph
#'
#' Constructs a k-nearest neighbor graph from spatial coordinates using
#' RANN for fast kNN queries.
#'
#' @param coords Numeric matrix of spatial coordinates (n x 2).
#' @param k Integer. Number of neighbors (default: 20).
#' @param sample_ids Vector of length n giving the sample each cell belongs to.
#'   Required. The graph is built within each sample separately.
#'
#' @return A list with two components:
#' \describe{
#'   \item{\code{indices}}{Integer matrix (n x k) of neighbor indices, given as
#'     row numbers into \code{coords}. Padded with \code{NA} where a sample
#'     holds fewer than \code{k + 1} cells.}
#'   \item{\code{distances}}{Numeric matrix (n x k) of distances to neighbors,
#'     \code{NA} in the same positions.}
#' }
#'
#' @details Uses \code{RANN::nn2} for fast kNN queries. The self-neighbor
#'   (distance 0) is excluded.
#'
#'   The search is partitioned by \code{sample_ids}, so a cell can only ever
#'   be linked to a cell from its own sample. This is not a safety measure
#'   bolted onto a pooled graph, it is the only meaningful definition: a
#'   spatial neighbour in a different tissue section does not exist
#'   physically, and sections routinely share a coordinate frame, so a pooled
#'   search returns those non-existent neighbours silently. Returned indices
#'   are global row numbers, so the output can be used without tracking the
#'   partition.
#'
#' @examples
#' \dontrun{
#' coords <- matrix(runif(200), ncol = 2)
#' knn <- build_knn_graph(coords, k = 10, sample_ids = rep("s1", 100))
#' }
#'
#' @importFrom RANN nn2
#' @export
build_knn_graph <- function(coords, k = 20, sample_ids) {
  if (missing(sample_ids)) {
    stop("sample_ids is required. Spatial neighbours are only meaningful ",
      "within one tissue section: sections routinely share a coordinate ",
      "frame, so a search pooled across samples silently returns neighbours ",
      "from a different section. For genuinely single-sample data pass ",
      "rep(\"s1\", nrow(coords)).",
      call. = FALSE
    )
  }
  if (length(sample_ids) != nrow(coords)) {
    stop("sample_ids must have one entry per row of coords (got ",
      length(sample_ids), " for ", nrow(coords), " rows).",
      call. = FALSE
    )
  }

  n <- nrow(coords)
  samples <- unique(sample_ids[!is.na(sample_ids)])
  message(sprintf(
    "Building %d-nearest neighbor graph for %d cells in %d sample(s)...",
    k, n, length(samples)
  ))

  indices <- matrix(NA_integer_, nrow = n, ncol = k)
  distances <- matrix(NA_real_, nrow = n, ncol = k)
  short <- character(0)

  for (s in samples) {
    rows <- which(!is.na(sample_ids) & sample_ids == s)
    if (length(rows) < 2) {
      short <- c(short, as.character(s))
      next
    }
    # k is clamped per sample, since a sample cannot supply more neighbours
    # than it has cells (less the cell itself).
    kk <- min(k, length(rows) - 1L)
    if (kk < k) short <- c(short, as.character(s))

    nn <- RANN::nn2(
      coords[rows, , drop = FALSE],
      coords[rows, , drop = FALSE],
      k = kk + 1L
    )
    # Drop the self-neighbour, then map local row numbers back to global ones.
    local_idx <- nn$nn.idx[, -1, drop = FALSE]
    indices[rows, seq_len(kk)] <- matrix(
      rows[local_idx], nrow = length(rows)
    )
    distances[rows, seq_len(kk)] <- nn$nn.dists[, -1, drop = FALSE]
  }

  if (length(short)) {
    warning(
      "Fewer than k + 1 cells in sample(s): ",
      paste(unique(short), collapse = ", "),
      ". Those rows are padded with NA.",
      call. = FALSE
    )
  }
  n_na_sample <- sum(is.na(sample_ids))
  if (n_na_sample > 0) {
    warning(n_na_sample, " cell(s) have an NA sample label and get no ",
      "neighbours.",
      call. = FALSE
    )
  }

  list(indices = indices, distances = distances)
}


#' Build radius-based neighbor graph
#'
#' Constructs a neighbor graph where cells are connected if they are within
#' a specified radius.
#'
#' @param coords Numeric matrix of spatial coordinates (n x 2).
#' @param radius Numeric. Search radius in coordinate units (typically microns).
#' @param sample_ids Vector of length n giving the sample each cell belongs to.
#'   Required, and the graph is built within each sample separately, for the
#'   reasons given in \code{\link{build_knn_graph}}.
#'
#' @return A list of length n, where each element is an integer vector of
#'   neighbor indices within the radius, given as row numbers into
#'   \code{coords}. Cells with no neighbour in range, and cells with an
#'   \code{NA} sample label, get \code{integer(0)}.
#'
#' @details Uses \code{\link[BiocNeighbors]{findNeighbors}} with an exact
#'   Euclidean search within each sample. All neighbors within the radius are
#'   returned, excluding the cell itself.
#'
#' @examples
#' \dontrun{
#' coords <- matrix(runif(200), ncol = 2)
#' neighbors <- build_radius_graph(coords, radius = 50,
#'                                 sample_ids = rep("s1", 100))
#' }
#'
#' @export
build_radius_graph <- function(coords, radius, sample_ids) {
  if (missing(sample_ids)) {
    stop("sample_ids is required. Spatial neighbours are only meaningful ",
      "within one tissue section: sections routinely share a coordinate ",
      "frame, so a search pooled across samples silently returns neighbours ",
      "from a different section. For genuinely single-sample data pass ",
      "rep(\"s1\", nrow(coords)).",
      call. = FALSE
    )
  }
  if (length(sample_ids) != nrow(coords)) {
    stop("sample_ids must have one entry per row of coords (got ",
      length(sample_ids), " for ", nrow(coords), " rows).",
      call. = FALSE
    )
  }

  n <- nrow(coords)
  samples <- unique(sample_ids[!is.na(sample_ids)])
  message(sprintf("Building radius neighbor graph (r=%s) in %d sample(s)...",
    radius, length(samples)
  ))

  neighbors <- rep(list(integer(0)), n)

  for (s in samples) {
    rows <- which(!is.na(sample_ids) & sample_ids == s)
    if (length(rows) < 2) next
    sc <- coords[rows, , drop = FALSE]

    nn <- BiocNeighbors::findNeighbors(
      sc, threshold = radius, get.distance = FALSE,
      BNPARAM = BiocNeighbors::KmknnParam(distance = "Euclidean")
    )
    for (j in seq_along(rows)) {
      neighbors[[rows[j]]] <- rows[nn$index[[j]]]
    }
  }

  neighbors
}


#' Calculate distance to nearest cell of a given type
#'
#' For each cell, computes the Euclidean distance to the nearest cell of a
#' specified target type using kNN search.
#'
#' @param coords Numeric matrix of spatial coordinates for all cells (n x 2).
#' @param cell_types Character vector of cell type labels (length n).
#' @param target_type Character. The cell type to measure distance to.
#' @param sample_ids Vector of length n giving the sample each cell belongs to.
#'   Required. The search is partitioned by it, so a cell can only ever match a
#'   target cell from its own sample. For genuinely single-sample data pass
#'   \code{rep("s1", nrow(coords))}.
#'
#' @return A numeric vector of length n with distances. Returns \code{NA} for
#'   all cells if no target cells are found anywhere, and \code{NA} for cells
#'   in a sample that contains no target cells.
#'
#' @details Sections routinely occupy overlapping coordinate ranges, since each
#'   section's coordinates start near zero in its own frame. A pooled search
#'   therefore returns the nearest target cell from any sample, and aggregating
#'   per sample afterwards does not repair it, because the damage happens
#'   before aggregation. There is no pooled mode: this function is a
#'   convenience wrapper over
#'   \code{\link{calculate_distance_to_type_by_sample}} that takes a cell-type
#'   label instead of a logical mask.
#'
#' @examples
#' \dontrun{
#' coords <- matrix(runif(200), ncol = 2)
#' types <- sample(c("A", "B", "C"), 100, replace = TRUE)
#' dists <- calculate_distance_to_type(coords, types, "A",
#'                                     sample_ids = rep("s1", 100))
#' }
#'
#' @importFrom RANN nn2
#' @export
calculate_distance_to_type <- function(coords, cell_types, target_type,
                                       sample_ids) {
  if (missing(sample_ids)) {
    stop("sample_ids is required. Spatial neighbours are only meaningful ",
      "within one tissue section: sections routinely share a coordinate ",
      "frame, so a search pooled across samples silently returns neighbours ",
      "from a different section. For genuinely single-sample data pass ",
      "rep(\"s1\", nrow(coords)).",
      call. = FALSE
    )
  }
  if (length(sample_ids) != nrow(coords)) {
    stop("sample_ids must have one entry per row of coords (got ",
      length(sample_ids), " for ", nrow(coords), " rows).",
      call. = FALSE
    )
  }

  # NA-safe: cell_types == target_type yields NA (not FALSE) for unannotated
  # cells, which would inject NA-coordinate rows into the RANN reference set
  # and corrupt every nearest-target distance.
  target_mask <- !is.na(cell_types) & cell_types == target_type

  if (!any(target_mask)) {
    warning(sprintf("No cells of type '%s' found", target_type))
    return(rep(NA_real_, nrow(coords)))
  }

  calculate_distance_to_type_by_sample(
    coords      = coords,
    sample_ids  = sample_ids,
    target_mask = target_mask,
    k           = 1
  )
}


#' Distance to the nearest cell of a target type, computed within each sample
#'
#' Sample-aware counterpart to \code{\link{calculate_distance_to_type}}. In
#' multi-sample spatial transcriptomics, tissue sections routinely occupy
#' overlapping coordinate ranges: each section's coordinates start near zero in
#' its own frame, so stacking sections places every one on top of every other. A
#' nearest-neighbour search run over cells from more than one sample then
#' silently returns neighbours from a different sample, and no amount of
#' downstream per-sample aggregation repairs it, because the damage happens
#' before aggregation.
#'
#' This function partitions the search by \code{sample_ids}, so a cell can only
#' ever match a target cell from its own sample.
#'
#' @param coords Numeric matrix (n x 2) of spatial coordinates.
#' @param sample_ids Vector of length n giving the sample each cell belongs to.
#' @param target_mask Logical vector of length n, \code{TRUE} for cells that are
#'   valid search targets. Must already be NA-safe.
#' @param k Integer. Number of nearest targets to average over. \code{k = 1}
#'   returns the distance to the single nearest target.
#'
#' @return Numeric vector of length n, in the original row order of
#'   \code{coords}. Cells in a sample containing no target cells receive
#'   \code{NA_real_}. A target cell matches itself at distance 0, matching the
#'   behaviour of the pooled implementation it replaces.
#'
#' @details Semantics are deliberately identical to a pooled
#'   \code{RANN::nn2(target_coords, coords, k)} call except for the sample
#'   partition: self-match at distance 0, and the row mean of the k nearest
#'   distances when \code{k > 1}. \code{k} is clamped per sample to the number
#'   of targets available in that sample.
#'
#' @seealso \code{\link{check_coordinate_frames}} to detect whether sections
#'   overlap in the first place.
#'
#' @examples
#' \dontrun{
#' d <- calculate_distance_to_type_by_sample(
#'   coords, sample_ids, cell_types == "Tumor", k = 1
#' )
#' }
#'
#' @importFrom RANN nn2
#' @export
calculate_distance_to_type_by_sample <- function(coords, sample_ids,
                                                 target_mask, k = 1) {
  n <- nrow(coords)
  if (length(sample_ids) != n || length(target_mask) != n) {
    stop("sample_ids and target_mask must each have one entry per row of ",
      "coords (got ", length(sample_ids), ", ", length(target_mask),
      " for ", n, " rows).",
      call. = FALSE
    )
  }
  if (anyNA(target_mask)) {
    stop("target_mask contains NA; mask the cell-type comparison NA-safely ",
      "before calling (see calculate_distance_to_type).",
      call. = FALSE
    )
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
    nn <- RANN::nn2(
      coords[tgt, , drop = FALSE],
      coords[rows, , drop = FALSE],
      k = kk
    )
    out[rows] <- if (kk == 1) {
      as.vector(nn$nn.dists)
    } else {
      rowMeans(nn$nn.dists)
    }
  }

  if (length(empty)) {
    warning(
      "No target cells in sample(s): ", paste(empty, collapse = ", "),
      ". Those cells receive NA distances.",
      call. = FALSE
    )
  }

  out
}


# Internal: calculate_distance_to_type_by_sample() with the "no target cells"
# warning muffled. For callers that drop the resulting NA cells themselves and
# raise their own warning naming the cell type, so the user sees one message
# rather than two describing the same samples.
.dist_by_sample_quiet <- function(...) {
  withCallingHandlers(
    calculate_distance_to_type_by_sample(...),
    warning = function(w) {
      if (grepl("No target cells in sample", conditionMessage(w))) {
        invokeRestart("muffleWarning")
      }
    }
  )
}


#' Detect and measure overlap between per-sample coordinate frames
#'
#' Reports whether tissue sections occupy the same coordinate region, which is
#' the condition under which a pooled nearest-neighbour search returns
#' neighbours from a different sample.
#'
#' @param coords Numeric matrix (n x 2) of spatial coordinates.
#' @param sample_ids Vector of length n giving the sample each cell belongs to.
#' @param target_mask Optional logical vector of length n marking the cells a
#'   distance would be measured TO (in RIPPLE, the query cells). When supplied,
#'   the returned \code{cross_sample_fraction} measures how much the overlap
#'   actually costs. Without it, severity cannot be judged and any overlap is
#'   treated as severe.
#' @param warn Logical. Emit a warning when any sample pair overlaps. Overlap
#'   is a fact about the coordinates, so there is no threshold below which it
#'   stops being true and none is applied here.
#' @param severe_fraction Numeric. Fraction of cells whose nearest target cell
#'   falls in another sample, above which the returned \code{severe} flag is
#'   set. This labels the result and is recorded in the run's QC output; it does
#'   NOT decide whether the warning fires.
#' @param max_cells Integer. Cap on the number of cells used to estimate
#'   \code{cross_sample_fraction}.
#'
#' @details
#' Two quantities are reported, and they are not interchangeable.
#'
#' \code{n_overlapping_pairs} counts pairs of samples whose bounding boxes
#' intersect. This is the direct measurement, and any intersection is a real
#' overlap.
#'
#' \code{ratio} is the sum of per-sample bounding-box areas over the global
#' bounding-box area. It is a ONE-SIDED test: a ratio above 1 proves at least
#' two sections overlap, by the pigeonhole principle, but a ratio below 1
#' proves nothing at all. Ten sections tiled with wide gaps, two of them
#' perfectly superimposed, give a ratio near 0.02 while that pair overlaps
#' completely. Do not gate on the ratio.
#'
#' \code{cross_sample_fraction} is the severity, and the only one of the three
#' that speaks to consequence: the fraction of cells whose nearest target cell
#' lies in a different sample. It sizes the problem, it does not decide whether
#' to mention it. The estimate uses a deterministically thinned
#' subsample, so it never touches the RNG stream and cannot perturb a seeded
#' permutation downstream.
#'
#' Rows with an \code{NA} coordinate or an \code{NA} sample label are dropped
#' before the boxes are measured, since neither has a well-defined extent. If
#' fewer than two samples remain the check is a no-op.
#'
#' @return Invisibly, a list with \code{ratio}, \code{sum_sample_area},
#'   \code{global_area}, \code{n_overlapping_pairs}, \code{n_pairs},
#'   \code{overlaps}, \code{cross_sample_fraction} (\code{NA} when
#'   \code{target_mask} is absent), \code{n_cells_checked} and \code{severe}.
#'
#' @export
check_coordinate_frames <- function(coords, sample_ids, target_mask = NULL,
                                    warn = TRUE, severe_fraction = 0.05,
                                    max_cells = 10000L) {
  # Drop rows this check cannot speak to. Without this, a single NA coordinate
  # or NA sample label makes every min/max NA, and the overlap comparison then
  # returns NA, which errors out of the `if` below.
  ok <- !is.na(sample_ids) & !is.na(coords[, 1]) & !is.na(coords[, 2])
  if (!all(ok)) {
    coords <- coords[ok, , drop = FALSE]
    sample_ids <- sample_ids[ok]
    if (!is.null(target_mask)) target_mask <- target_mask[ok]
  }

  samples <- unique(sample_ids)
  if (length(samples) < 2) {
    return(invisible(list(
      ratio = NA_real_, sum_sample_area = NA_real_, global_area = NA_real_,
      n_overlapping_pairs = 0L, n_pairs = 0L, overlaps = FALSE,
      cross_sample_fraction = NA_real_, n_cells_checked = 0L, severe = FALSE
    )))
  }

  box <- do.call(rbind, lapply(samples, function(s) {
    i <- sample_ids == s
    c(
      xmin = min(coords[i, 1]), xmax = max(coords[i, 1]),
      ymin = min(coords[i, 2]), ymax = max(coords[i, 2])
    )
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
  # boundary? Estimated on a deterministically thinned subsample, so this
  # never consumes random numbers and cannot shift a seeded permutation.
  cross <- NA_real_
  n_checked <- 0L
  if (!is.null(target_mask) && n_ov > 0) {
    tm <- !is.na(target_mask) & target_mask
    tgt <- which(tm)
    if (length(tgt) > 0) {
      n <- nrow(coords)
      idx <- if (n > max_cells) {
        unique(as.integer(seq.int(1L, n, length.out = max_cells)))
      } else {
        seq_len(n)
      }
      nn <- RANN::nn2(coords[tgt, , drop = FALSE],
        coords[idx, , drop = FALSE],
        k = 1
      )
      owner <- sample_ids[tgt][as.vector(nn$nn.idx)]
      cross <- mean(owner != sample_ids[idx])
      n_checked <- length(idx)
    }
  }

  # Any bounding-box intersection is a real overlap, so an unmeasurable
  # severity is treated as severe rather than waved through.
  severe <- n_ov > 0 && (is.na(cross) || cross > severe_fraction)

  res <- list(
    ratio = ratio, sum_sample_area = sum(areas), global_area = global,
    n_overlapping_pairs = n_ov, n_pairs = n_pairs, overlaps = n_ov > 0,
    cross_sample_fraction = cross, n_cells_checked = n_checked,
    severe = severe
  )

  # ANY bounding-box intersection warns. Overlap is a fact about the
  # coordinates, and the pairwise intersection is the direct measurement of it,
  # so there is no threshold below which it stops being true. The measured
  # fraction goes in the message to say how much it costs, and `severe` is
  # still returned and recorded, but neither gates whether the user is told.
  if (warn && n_ov > 0) {
    warning(
      n_ov, " of ", n_pairs, " sample pairs have overlapping coordinate ",
      "bounding boxes, so the sections share a coordinate frame",
      if (is.na(cross)) {
        ""
      } else {
        paste0(" and ", round(100 * cross, 1), "% of cells have their nearest ",
          "target cell in a DIFFERENT sample"
        )
      },
      ". RIPPLE partitions its own distance search by sample, so these ",
      "results are unaffected, but any pooled nearest-neighbour analysis on ",
      "these coordinates returns cross-sample neighbours. That includes most ",
      "other spatial tools and any hand-rolled kNN.",
      call. = FALSE
    )
  }

  invisible(res)
}


#' Get neighbor cell types for a set of query cells
#'
#' Retrieves the cell type labels of all neighbors for specified query cells,
#' based on a precomputed kNN graph.
#'
#' @param cell_types Character vector of cell type labels for all cells.
#' @param query_indices Integer vector of indices for query cells.
#' @param knn_result List. Result from \code{\link{build_knn_graph}}, containing
#'   \code{indices} and \code{distances} matrices.
#'
#' @return A \code{data.table} with columns \code{query_cell}, \code{neighbor_cell},
#'   and \code{neighbor_type}.
#'
#' @importFrom data.table rbindlist data.table
#' @export
get_neighbor_cell_types <- function(cell_types, query_indices, knn_result) {
  results <- data.table::rbindlist(lapply(query_indices, function(i) {
    neighbor_idx <- knn_result$indices[i, ]
    data.table::data.table(
      query_cell = i,
      neighbor_cell = neighbor_idx,
      neighbor_type = cell_types[neighbor_idx]
    )
  }))
  return(results)
}


#' Calculate neighborhood composition for a cell type
#'
#' Computes the proportions of different cell types among the neighbors of
#' all cells of a specified query type.
#'
#' @param cell_types Character vector of cell type labels for all cells.
#' @param query_cell_type Character. Cell type whose neighborhoods to analyze.
#' @param knn_result List. Result from \code{\link{build_knn_graph}}.
#'
#' @return A \code{data.table} with columns \code{query_cell_type},
#'   \code{neighbor_type}, \code{count}, and \code{proportion}.
#'
#' @examples
#' \dontrun{
#' coords <- matrix(runif(200), ncol = 2)
#' types <- sample(c("A", "B", "C"), 100, replace = TRUE)
#' knn <- build_knn_graph(coords, k = 10, sample_ids = rep("s1", 100))
#' comp <- calculate_neighbor_composition(types, "A", knn)
#' }
#'
#' @importFrom data.table data.table setcolorder
#' @export
calculate_neighbor_composition <- function(cell_types, query_cell_type, knn_result) {
  query_idx <- which(cell_types == query_cell_type)

  if (length(query_idx) == 0) {
    warning(sprintf("No cells of type '%s' found", query_cell_type))
    return(data.table::data.table())
  }

  # Get all neighbor types
  neighbor_types <- as.vector(knn_result$indices[query_idx, ])
  neighbor_types <- cell_types[neighbor_types]

  # Count and calculate proportions
  counts <- table(neighbor_types)
  dt <- data.table::data.table(
    neighbor_type = names(counts),
    count = as.integer(counts),
    proportion = as.numeric(counts) / sum(counts)
  )

  dt[, query_cell_type := query_cell_type]
  data.table::setcolorder(dt, c("query_cell_type", "neighbor_type", "count", "proportion"))

  return(dt)
}


#' Check spatial autocorrelation in RIPPLE model residuals
#'
#' Computes Moran's I on Poisson GLM residuals for one or more genes of
#' interest within a specific target cell type. High Moran's I indicates
#' that nearby cells have correlated residuals, which means the GLM's
#' independence assumption is violated and per-sample p-values may be
#' anti-conservative.
#'
#' This diagnostic assesses residual spatial structure for selected genes.
#' It does not correct autocorrelation or recalibrate the Wald p-values.
#' Use the same input subset, query-neighbor count and distance cap as in
#' the main analysis. Requested genes are refitted without repeating the
#' cross-sample expression filter; each fit requires five expressing cells.
#'
#' @param input A Seurat, SCE, or SpatialExperiment object (or path to
#'   an \code{.rds} file containing one).
#' @param genes Character vector of gene names to check.
#' @param celltype_column Cell type column name.
#' @param target_celltype Which cell type to assess (the target, not the query).
#' @param query_celltype Query cell type (needed for distance calculation).
#' @param sample_column Sample/replicate column name (default: \code{"sample_id"}).
#' @param k Number of nearest neighbors for the spatial weights matrix
#'   (default: 20).
#' @param max_distance_um Distance cap in micrometers (default: 200, or
#'   \code{ripple_config()}). More distant target cells are retained at the cap,
#'   matching \code{run_ripple()}; use the same value as the main analysis.
#' @param k_neighbors Number of nearest query cells whose distances are averaged
#'   (default: 1, or \code{ripple_config()}). Match the main analysis. This is
#'   separate from \code{k}, which controls the residual spatial weights.
#' @param min_cells_per_sample Minimum target cells per sample (default: 30, or
#'   \code{ripple_config()}). At least \code{k + 1} usable cells are also needed.
#' @param x_column X coordinate column (default: NULL, auto-detect).
#' @param y_column Y coordinate column (default: NULL, auto-detect).
#' @param verbose Print progress (default: TRUE).
#'
#' @return A \code{data.table} with columns:
#' \describe{
#'   \item{gene}{Gene name.}
#'   \item{sample_id}{Sample identifier.}
#'   \item{morans_i}{Observed Moran's I statistic.}
#'   \item{morans_expected}{Expected Moran's I under no autocorrelation.}
#'   \item{morans_pvalue}{P-value from \code{spdep::moran.test()}.}
#'   \item{interpretation}{One of "none", "weak", "moderate", "strong".}
#'   \item{n_cells}{Number of cells used.}
#' }
#'
#' @details
#' For each gene and sample, the function:
#' \enumerate{
#'   \item Selects target cells and calculates mean distance to the nearest
#'     \code{k_neighbors} query cells within the same sample. Distances are
#'     capped at \code{max_distance_um}, without excluding more distant cells.
#'   \item Fits the same Poisson GLM as \code{\link{fit_poisson}}.
#'   \item Extracts deviance residuals.
#'   \item Builds a k-nearest-neighbor spatial weights matrix.
#'   \item Computes Moran's I via \code{spdep::moran.test()}.
#' }
#'
#' Interpretation thresholds:
#' \itemize{
#'   \item \code{|I| < 0.05}: "none".
#'   \item \code{0.05 <= |I| < 0.15}: "weak".
#'   \item \code{0.15 <= |I| < 0.30}: "moderate".
#'   \item \code{|I| >= 0.30}: "strong".
#' }
#' These descriptive labels do not establish independence or calibrated
#' inference. Samples with insufficient cells or failed fits/tests are reported
#' with missing statistics and an explanatory \code{interpretation} value.
#'
#' @examples
#' \dontrun{
#' autocor <- check_spatial_autocorrelation(
#'   input           = my_spe,
#'   genes           = c("MIF", "CD74", "PDCD1"),
#'   celltype_column = "cell_type",
#'   target_celltype = "CD8_T",
#'   query_celltype  = "tumor",
#'   sample_column   = "patient"
#' )
#' autocor[order(-morans_i)]
#' }
#'
#' @importFrom data.table data.table rbindlist
#' @export
check_spatial_autocorrelation <- function(input,
                                          genes,
                                          celltype_column,
                                          target_celltype,
                                          query_celltype,
                                          sample_column = "sample_id",
                                          k = 20,
                                          max_distance_um = 200,
                                          x_column = NULL,
                                          y_column = NULL,
                                          verbose = TRUE,
                                          k_neighbors = 1,
                                          min_cells_per_sample = 30) {
  if (missing(sample_column)) sample_column <- .resolve(NULL, "sample_column", "sample_id")
  if (missing(k_neighbors)) k_neighbors <- .resolve(NULL, "k_neighbors", 1L)
  if (missing(max_distance_um)) max_distance_um <- .resolve(NULL, "max_distance_um", 200)
  if (missing(min_cells_per_sample)) min_cells_per_sample <- .resolve(NULL, "min_cells_per_sample", 30L)
  for (nm in c("k", "k_neighbors", "min_cells_per_sample")) {
    value <- get(nm)
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value < 1 || value != floor(value)) {
      stop(nm, " must be a positive integer.", call. = FALSE)
    }
  }
  if (!is.numeric(max_distance_um) || length(max_distance_um) != 1L ||
      is.na(max_distance_um) || max_distance_um <= 0) {
    stop("max_distance_um must be positive (Inf disables capping).", call. = FALSE)
  }
  if (!requireNamespace("spdep", quietly = TRUE)) {
    stop(
      "Package 'spdep' is required for spatial autocorrelation testing.\n",
      "Install with: install.packages('spdep')\n",
      "On macOS inside a conda env, install via conda instead so the sf ",
      "system libraries (GDAL, GEOS, PROJ) get resolved:\n",
      "  conda install -c conda-forge r-spdep",
      call. = FALSE
    )
  }

  .msg <- function(...) if (isTRUE(verbose)) message(...)

  # Load data
  data <- .resolve_input(input, require_expr = FALSE, verbose = verbose)
  count_matrix <- data$counts
  cell_data <- data$meta
  rm(data)

  # Resolve coordinates

  coord_cols <- get_coord_columns(cell_data, x_col = x_column, y_col = y_column)
  coords <- as.matrix(cell_data[, ..coord_cols])

  # Validate columns
  if (!celltype_column %in% names(cell_data)) {
    stop("celltype_column '", celltype_column, "' not found in metadata.",
      call. = FALSE
    )
  }
  if (!sample_column %in% names(cell_data)) {
    stop("sample_column '", sample_column, "' not found in metadata.",
      call. = FALSE
    )
  }

  # Calculate distances to query (NA-safe mask; see calculate_distance_to_type)
  celltypes_all <- cell_data[[celltype_column]]
  query_mask <- !is.na(celltypes_all) & celltypes_all == query_celltype
  if (sum(query_mask) < 1) {
    stop("No query cells found for '", query_celltype, "'.", call. = FALSE)
  }
  # Partitioned by sample. The Moran's I weights below are already built per
  # sample; computing the distance pooled would have undercut the very
  # diagnostic this function provides.
  sample_ids_all <- cell_data[[sample_column]]
  cell_data[, dist_to_query := calculate_distance_to_type_by_sample(
    coords, sample_ids_all, query_mask, k = k_neighbors
  )]

  # Match the main analysis: retain distant target cells at the distance cap.
  # Samples without query cells have undefined distance and cannot be fitted.
  target_mask <- !is.na(celltypes_all) &
    celltypes_all == target_celltype &
    is.finite(cell_data$dist_to_query)
  stopifnot(!anyNA(target_mask))
  target_data <- cell_data[target_mask]
  target_data[, dist_to_query := pmin(dist_to_query, max_distance_um)]
  if (nrow(target_data) == 0) {
    stop("No target cells with defined within-sample query distances.", call. = FALSE)
  }
  target_barcodes <- target_data$barcode
  target_coords <- coords[target_mask, , drop = FALSE]

  .msg(
    "Target cells (", target_celltype, ", distance capped at ", max_distance_um,
    " um; query k=", k_neighbors, "): ", nrow(target_data)
  )

  # Total counts for offset
  target_counts <- count_matrix[, target_barcodes, drop = FALSE]
  total_counts <- Matrix::colSums(target_counts)

  # Validate genes exist
  available_genes <- intersect(genes, rownames(target_counts))
  missing_genes <- setdiff(genes, rownames(target_counts))
  if (length(missing_genes) > 0) {
    .msg("Genes not found in data: ", paste(missing_genes, collapse = ", "))
  }
  if (length(available_genes) == 0) {
    stop("None of the specified genes found in the count matrix.", call. = FALSE)
  }

  samples <- unique(target_data[[sample_column]])
  .msg("Samples: ", length(samples))
  .msg("Genes to check: ", paste(available_genes, collapse = ", "))

  # Per gene x sample: fit GLM, extract residuals, compute Moran's I
  results <- data.table::rbindlist(lapply(available_genes, function(g) {
    .msg("  Gene: ", g)

    data.table::rbindlist(lapply(samples, function(s) {
      samp_idx <- which(target_data[[sample_column]] == s)
      if (length(samp_idx) < max(k + 1, min_cells_per_sample)) {
        return(data.table::data.table(
          gene = g, sample_id = s,
          morans_i = NA_real_, morans_expected = NA_real_,
          morans_pvalue = NA_real_, interpretation = "insufficient_cells",
          n_cells = length(samp_idx)
        ))
      }

      samp_counts <- as.numeric(target_counts[g, target_barcodes[samp_idx]])
      samp_dist <- target_data$dist_to_query[samp_idx]
      samp_total <- total_counts[target_barcodes[samp_idx]]
      samp_coords <- target_coords[samp_idx, , drop = FALSE]

      # Fit Poisson GLM (same as fit_poisson but we need the full model)
      valid <- is.finite(samp_counts) & is.finite(samp_dist) &
        is.finite(samp_total) & samp_total > 0
      if (sum(valid) < max(k + 1, min_cells_per_sample) || sum(samp_counts[valid] > 0) < 5) {
        return(data.table::data.table(
          gene = g, sample_id = s,
          morans_i = NA_real_, morans_expected = NA_real_,
          morans_pvalue = NA_real_, interpretation = "insufficient_cells",
          n_cells = sum(valid)
        ))
      }

      y <- samp_counts[valid]
      d <- samp_dist[valid]
      log_total <- log(samp_total[valid])
      xy <- samp_coords[valid, , drop = FALSE]

      fit <- tryCatch(
        suppressWarnings(stats::glm(y ~ d + offset(log_total),
          family = stats::poisson()
        )),
        error = function(e) NULL
      )

      if (is.null(fit) || !fit$converged ||
          !"d" %in% rownames(summary(fit)$coefficients)) {
        return(data.table::data.table(
          gene = g, sample_id = s,
          morans_i = NA_real_, morans_expected = NA_real_,
          morans_pvalue = NA_real_, interpretation = "glm_failed",
          n_cells = length(y)
        ))
      }

      # Deviance residuals
      resid <- stats::residuals(fit, type = "deviance")

      # Build spatial weights (k-NN)
      effective_k <- min(k, nrow(xy) - 1)
      knn <- spdep::knearneigh(xy, k = effective_k)
      nb <- spdep::knn2nb(knn)
      lw <- spdep::nb2listw(nb, style = "W")

      # Moran's I test
      mt <- tryCatch(
        spdep::moran.test(resid, lw, alternative = "two.sided"),
        error = function(e) NULL
      )

      if (is.null(mt)) {
        return(data.table::data.table(
          gene = g, sample_id = s,
          morans_i = NA_real_, morans_expected = NA_real_,
          morans_pvalue = NA_real_, interpretation = "moran_failed",
          n_cells = length(y)
        ))
      }

      mi <- mt$estimate["Moran I statistic"]
      me <- mt$estimate["Expectation"]
      mp <- mt$p.value

      interp <- if (abs(mi) < 0.05) {
        "none"
      } else if (abs(mi) < 0.15) {
        "weak"
      } else if (abs(mi) < 0.30) {
        "moderate"
      } else {
        "strong"
      }

      data.table::data.table(
        gene = g, sample_id = s,
        morans_i = unname(mi), morans_expected = unname(me),
        morans_pvalue = mp, interpretation = interp,
        n_cells = length(y)
      )
    }))
  }))

  .msg("\nSummary:")
  .msg("  Total tests: ", nrow(results[!is.na(morans_i)]))
  if (nrow(results[!is.na(morans_i)]) > 0) {
    .msg("  Median Moran's I: ", round(stats::median(results$morans_i, na.rm = TRUE), 4))
    interp_tab <- table(results$interpretation)
    .msg("  Interpretation: ", paste(names(interp_tab), interp_tab,
      sep = "=", collapse = ", "
    ))
  }

  results
}
