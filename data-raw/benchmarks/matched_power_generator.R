# Matched-expression power design. No production inference changes.
source('data-raw/benchmarks/benchmark_helpers.R')
generate_power_matched <- function(iteration, beta, n_samples=10L) {
  # The same seed gives identical layouts for every effect and sample count.
  spe <- generate_benchmark_data(n_samples=10L,n_gradient_neg=5L,
    n_background=46L,beta=0,seed=9400000L+iteration)
  rownames(spe)[51] <- 'COMPOSITION_BALANCER'
  meta <- as.data.frame(SummarizedExperiment::colData(spe))
  xy <- SpatialExperiment::spatialCoords(spe)
  mu <- matrix(3,nrow=51,ncol=ncol(spe))
  distance <- numeric(ncol(spe))
  for(s in unique(meta$sample_id)) {
    idx <- which(meta$sample_id==s)
    query <- which(meta$sample_id==s & meta$cell_type=='Tumor')
    distance[idx] <- pmin(RANN::nn2(xy[query,,drop=FALSE],xy[idx,,drop=FALSE],k=1)$nn.dists[,1],200)
    target <- which(meta$sample_id==s & meta$cell_type=='T_cell')
    shape <- exp(beta*distance[target])
    # Exact expected mean of 3 counts per target cell, separately per sample.
    mu[1:5,target] <- rep(3*shape/mean(shape),each=5)
  }
  # Fixed measured total ensures the fitted offset is the true exposure.
  # The balancing feature represents other transcripts. It is a known non-null
  # in signal simulations, included in BH but excluded from the 45 null genes.
  library_size <- 500L
  mu[51,] <- library_size-colSums(mu[1:50,,drop=FALSE])
  stopifnot(all(mu>0),max(abs(colSums(mu)-library_size))<1e-10)
  set.seed(9500000L+iteration)
  y <- vapply(seq_len(ncol(spe)),function(i)
    as.integer(rmultinom(1,library_size,mu[,i]/library_size)),integer(51))
  dimnames(y) <- dimnames(spe)
  SummarizedExperiment::assay(spe,'counts') <- methods::as(y,'CsparseMatrix')
  keep <- meta$sample_id %in% paste0('sample_',seq_len(n_samples))
  list(spe=spe[,keep],mu=mu[,keep,drop=FALSE],distance=distance[keep],
       library_size=library_size)
}
