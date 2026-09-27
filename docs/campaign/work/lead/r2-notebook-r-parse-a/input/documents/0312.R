make.parents.of.interactions <-
  function(mf, data) {
    modelterms <- terms(mf, data = data)
    termnamesX <- attr(modelterms, "term.labels")
    p <- length(termnamesX)
    interactions <- grep(":", termnamesX)
    parents <- diag(p)
    colnames(parents) <- termnamesX
    rownames(parents) <- termnamesX
    for (j in interactions) {
      term <- interactions[j]
      main <- unlist(strsplit(termnamesX[j], ":", fixed = TRUE))
      parents.of.term <- main
      for (i in 2:length(main)) {
        parents.of.term <- c(
          parents.of.term,
          utils::combn(main, i, FUN = paste0, collapse = ":")
        )
      }
      parents[j, parents.of.term] <- 1
    }

    X <- model.matrix(modelterms, data = data)

    loc <- attr(X, "assign")[-1] # drop intercept

    parents <- parents[loc, loc]
    parents <- rbind(0, parents)
    parents <- cbind(0, parents)
    parents[1, 1] <- 1
    rownames(parents) <- colnames(X)
    colnames(parents) <- colnames(X)

    return(parents)
  }