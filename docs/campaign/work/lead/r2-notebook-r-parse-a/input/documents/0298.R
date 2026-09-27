compute_pr <- function(yvec, pvec) {
  porder <- order(pvec, decreasing = TRUE)
  yvecs  <- yvec[porder]
  pvecs  <- pvec[porder]
  p      <- sum(yvecs)
  n      <- length(yvecs) - p
  tp     <- cumsum(yvecs)
  tpr    <- tp/p
  pp     <- 1:length(yvecs) 
  prec   <- tp/pp
  
  dup_pred  <- rev(duplicated(pvecs))
  dup_stats <- duplicated(tpr) & duplicated(prec)
  dups <- dup_pred | dup_stats
  
  prec <- c(NaN, prec[!dups])
  tpr <- c(0, tpr[!dups])
  
  prc_data <- data.frame(x = tpr,
                         y = prec)
  prc_data
}