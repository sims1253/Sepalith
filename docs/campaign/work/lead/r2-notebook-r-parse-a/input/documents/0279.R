c.BFodds <-
  function(..., recursive = FALSE) {
    z <- list(...)
    if (length(z) == 1) {
      return(z[[1]])
    }
    correctClass <- unlist(lapply(z, function(object) {
      inherits(object, "BFodds")
    }))
    if (!all(correctClass)) {
      stop("Cannot concatenate odds with non-odds object.")
    }

    denoms <- lapply(z, function(object) {
      object@denominator
    })
    samedenom <- unlist(lapply(
      denoms[-1],
      function(el, cmp) {
        el %same% cmp
      },
      cmp = denoms[[1]]
    ))
    if (!all(samedenom)) {
      stop("Cannot concatenate odds objects with different denominator models.")
    }

    logodds <- lapply(z, function(object) {
      object@logodds
    })
    df_rownames <- unlist(lapply(z, function(object) {
      rownames(object@logodds)
    }))
    df_rownames <- make.unique(df_rownames, sep = " #")
    logodds <- do.call("rbind", logodds)
    rownames(logodds) <- df_rownames

    ### Grab the Bayes factors
    is.prior <- seq_along(z) * NA
    for (i in seq_along(is.prior)) {
      is.prior[i] <- is.null(z[[i]]@bayesFactor)
    }
    if (!any(is.prior)) {
      bfs <- lapply(z, function(object) {
        object@bayesFactor
      })
      bfs <- do.call("c", bfs)
      bf <- BFodds(bfs, logodds = logodds, bayesFactor = bfs)
    } else if (all(is.prior)) {
      numerators <- unlist(
        lapply(z, function(object) {
          object@numerator
        }),
        recursive = FALSE,
        use.names = FALSE
      )

      bf <- new(
        "BFodds",
        numerator = numerators,
        denominator = z[[1]]@denominator,
        logodds = logodds,
        bayesFactor = NULL,
        version = BFInfo(FALSE)
      )
    } else {
      stop("Cannot concatenate prior odds with posterior odds.")
    }

    return(bf)
  }