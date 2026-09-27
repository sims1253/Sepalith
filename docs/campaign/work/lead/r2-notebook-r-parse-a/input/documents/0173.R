mcfa <- function(
  Y,
  g,
  q,
  itmax = 50,
  nkmeans = 20,
  nrandom = 20,
  tol = 1.e-5,
  initClust = NULL,
  initMethod = 'eigenA',
  convMeas = 'diff',
  errorMsg = FALSE,
  verbose = FALSE,
  ...
) {
  if (!is.matrix(Y)) {
    Y <- as.matrix(Y)
  }
  p <- ncol(Y)
  if (is.null(p)) {
    stop("the data must have more than one variable")
  }
  #return(ERRMSG <- "Error: the data must have more than one variable")
  n <- nrow(Y)
  if (p < q) {
    stop(
      "the number of factors must not be greater
    than the number of variables"
    )
  }

  startClust <- Starts(Y, g, initClust, nkmeans, nrandom)
  maxinit <- ncol(startClust)
  if (is.null(maxinit)) {
    maxinit <- 1
  }

  ERRMSG <- NULL
  maxLOGL <- -Inf
  for (ii in seq_len(maxinit)) {
    if (min(table(startClust[, ii]) == 1)) {
      when <- paste("At start", ii)
      what <- "Initial partition was not used as it has a cluster of one
            sample."
      ERRMSG <- rbind(ERRMSG, cbind(when, what))
      next
    }
    startModel <- try(init.est.para.mcfa(Y, g, q, startClust[, ii], initMethod))
    if (class(startModel)[1] == "try-error") {
      when <- paste("At start", ii)
      what <- "Failed to estimate initial parameters"
      ERRMSG <- rbind(ERRMSG, cbind(when, what))
      next
    }
    estModel <- est.mcfa(
      Y,
      g,
      q,
      itmax,
      tol,
      startModel$pivec,
      startModel$A,
      startModel$xi,
      startModel$omega,
      startModel$D,
      convMeas
    )
    if ((class(estModel)[1] == "mcfa")) {
      if (estModel$logL > maxLOGL) {
        Hmodel <- estModel
        maxLOGL <- Hmodel$logL
      }
      if (verbose) {
        message(sprintf(
          "g = %i, q = %i, initialization %i logL %8.4f,
                    maxlogL = %8.4f \n",
          g,
          q,
          ii,
          estModel$logL,
          maxLOGL
        ))
      }
    }
    if (class(estModel)[1] == "error") {
      when <- paste("At start", ii)
      what <- estModel
      ERRMSG <- rbind(ERRMSG, cbind(when, what))
    }
  }

  if (!exists("Hmodel")) {
    stop("Error: Failed to Estimate a Model. See Error Messages.")
    return(ERRMSG)
  }

  CH <- chol(t(Hmodel$A) %*% Hmodel$A)
  Hmodel$A <- Hmodel$A %*% solve(CH)
  Hmodel$xi <- CH %*% Hmodel$xi
  for (i in seq_len(g)) {
    Hmodel$omega[,, i] <- CH %*% Hmodel$omega[,, i] %*% t(CH)
  }
  d <- (g - 1) + p + q * (p + g) + g * q * (q + 1) / 2 - q * q
  Hmodel$tau <- do.call('tau.mcfa', c(list(Y = Y), Hmodel))
  Hmodel$BIC <- -2 * Hmodel$logL + d * log(n)
  Hmodel$clust <- apply(Hmodel$tau, 1, which.max)
  Hmodel <- append(
    Hmodel,
    do.call('factor.scores.mcfa', c(list(Y = Y), Hmodel))
  )
  Hmodel$call <- match.call()
  if (errorMsg) {
    Hmodel$ERRMSG <- ERRMSG
  }
  class(Hmodel) <- "mcfa"
  return(Hmodel)
}