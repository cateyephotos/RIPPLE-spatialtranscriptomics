# Run from the repository root: Rscript data-raw/benchmarks/bench_power_matched.R 3
# Repeat for N=5 and N=10; add pilot for a single-dataset check.
suppressPackageStartupMessages({library(data.table);devtools::load_all(quiet=TRUE)})
source('data-raw/benchmarks/matched_power_generator.R')
args <- commandArgs(TRUE)
n_samp <- as.integer(args[1]); pilot <- length(args)>1 && args[2]=='pilot'
stopifnot(length(n_samp) == 1L, n_samp %in% c(3L, 5L, 10L))
base <- 'data-raw/benchmarks/results/power_matched'
job_dir <- file.path(base,if(pilot) 'pilot' else paste0('n',n_samp))
dir.create(job_dir,recursive=TRUE,showWarnings=FALSE)
betas <- if(pilot) -.002 else -c(.0005,.001,.002,.005,.01,.02)
iterations <- if(pilot) 1L else 30L
rows <- list();diagnostics <- list();warnings_seen <- character()
for(iter in seq_len(iterations)) for(b in betas) {
  key <- sprintf('iter%02d_beta%.4f',iter,abs(b))
  cached <- file.path(job_dir,paste0(key,'.rds'))
  if(file.exists(cached)) {
    ans <- readRDS(cached)
  } else {
    sim <- generate_power_matched(iter,b,n_samp)
    spe <- sim$spe
    meta <- as.data.frame(SummarizedExperiment::colData(spe))
    target <- which(meta$cell_type=='T_cell')
    sample_diagnostics <- rbindlist(lapply(unique(meta$sample_id),function(s) {
      idx <- which(meta$cell_type=='T_cell' & meta$sample_id==s)
      y <- as.matrix(SummarizedExperiment::assay(spe)[1:5,idx,drop=FALSE])
      expected <- sim$mu[1,idx]
      stopifnot(abs(mean(expected)-3)<1e-10,
        max(abs(diff(log(expected[order(sim$distance[idx])]))-
          b*diff(sort(sim$distance[idx]))))<1e-10)
      data.table(sample=s,expected_mean=mean(expected),observed_mean=mean(y),
        min_expressing=min(rowSums(y>0)),max_expected=max(expected),
        minimum_expected=min(expected))
    }))
    run_dir <- file.path(job_dir,paste0(key,'_pipeline'))
    res <- withCallingHandlers(run_ripple(input=spe,query_celltype='Tumor',
      target_celltypes='T_cell',celltype_column='cell_type',sample_column='sample_id',
      output_dir=run_dir,analysis_name='bench',verbose=FALSE,n_permutations=0,
      max_distance_um=200,k_neighbors=1,sign_consistency=1,
      min_expr_pct=.01,min_expr_floor=25,priority_genes=character(),organism='human'),
      warning=function(w) {warnings_seen <<- unique(c(warnings_seen,conditionMessage(w)));invokeRestart('muffleWarning')})
    paths <- list.files(run_dir,pattern='^coef_per_sample.csv$',recursive=TRUE,full.names=TRUE)
    stopifnot(length(paths)==1L)
    coef <- fread(paths)
    scores <- rbindlist(lapply(c(1,.75),function(gate) {
      agg <- coef[,as.list(compute_fisher_pval(pval,coef,sign_threshold=gate)),by=gene]
      agg[,fisher_fdr:=p.adjust(fisher_pval,'BH')]
      if(gate==1) {
        chk <- merge(agg[,.(gene,fisher_fdr)],res[,.(gene,fisher_fdr)],by='gene')
        stopifnot(nrow(chk)==nrow(res),max(abs(chk$fisher_fdr.x-chk$fisher_fdr.y))<1e-12)
      }
      grad <- agg[startsWith(gene,'GRAD_NEG_')]
      bg <- agg[startsWith(gene,'BG_')]
      tp <- sum(grad$fisher_fdr<.05,na.rm=TRUE)
      fp <- sum(bg$fisher_fdr<.05,na.rm=TRUE)
      data.table(beta=b,n_samples=n_samp,iteration=iter,gate=gate,
        tp=tp,fn=5L-tp,fp=fp,n_grad_tested=nrow(grad),n_grad_filtered=5L-nrow(grad),
        n_bg_tested=nrow(bg),sensitivity=tp/5,
        conditional_sensitivity=tp/max(nrow(grad),1),
        mean_fitted_beta=mean(grad$median_coef),n_tested=nrow(agg))
    }))
    ans <- list(scores=scores,diagnostics=sample_diagnostics,coefs=coef,
                strict_results=res,warnings=warnings_seen)
    saveRDS(ans,cached)
  }
  rows[[length(rows)+1L]] <- ans$scores
  diagnostics[[length(diagnostics)+1L]] <- cbind(beta=b,n_samples=n_samp,iteration=iter,ans$diagnostics)
  fwrite(rbindlist(rows),file.path(job_dir,'per_run.csv'))
  fwrite(rbindlist(diagnostics),file.path(job_dir,'diagnostics.csv'))
  cat(sprintf('N=%d iteration=%d/%d beta=%.4f complete\n',n_samp,iter,iterations,b))
}
saveRDS(list(per_run=rbindlist(rows),diagnostics=rbindlist(diagnostics),
  warnings=warnings_seen,session=capture.output(sessionInfo())),file.path(job_dir,'results.rds'))
writeLines('Complete',file.path(job_dir,'COMPLETE'))
