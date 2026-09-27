mk_folds <- function(y, nfold = 10, random = FALSE) {
  n <- length(y)
  nos_g <- table(y)
  G <- length(nos_g)
  nfold <- min(nfold, n)

  reduced.rep <- TRUE
  while (reduced.rep) {
    if (!random) {
      folds <- rep(1:nfold, length = n)
    } else {
      folds <- sample(rep(1:nfold, length = n))
    }
    ## check any fold has reduced class representation
    reduced.rep <- FALSE
    for (i in 1:nfold) {
      G_ifold <- length(unique(y[folds != i]))
      if (G_ifold < G) {
        reduced.rep <- TRUE
        random <- TRUE
        break
      }
    }
  }

  ## create fold list
  foldlist <- rep(list(""), nfold)
  for (i in 1:nfold) {
    foldlist[[i]] <- which(folds == i)
  }

  foldlist
}