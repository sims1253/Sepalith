testProportionality <- function(object, parm = NULL, transform = "km") {
  .checkInterface(object$cyclopsData, testOnly = TRUE)

  if (object$cyclopsData$modelType != "cox") {
    stop("Proportionality test for only Cox models are currently implemented")
  }

  nCovariates <- getNumberOfCovariates(object$cyclopsData)
  if (nCovariates != 1) {
    stop("Only single-covariate models are currently implemented")
  }

  times <- getTimeVector(object$cyclopsData)
  y <- getYVector(object$cyclopsData)
  survY <- survival::Surv(time = times, event = y)
  if (is.character(transform)) {
    tname <- transform
    ttimes <- switch(
      transform,
      identity = times,
      rank = rank(times),
      log = log(times),
      km = {
        temp <- survival::survfitKM(
          factor(rep(1L, nrow(survY))),
          survY,
          se.fit = FALSE
        )
        indx <- findInterval(times, temp$time, left.open = TRUE)
        1 - c(1, temp$surv)[indx + 1]
      },
      stop("Unrecognized transform")
    )
  } else {
    tname <- deparse(substitute(transform))
    if (length(tname) > 1) {
      tname <- "user"
    }
    ttimes <- transform(times)
  }
  transformedTimes <- ttimes - mean(ttimes[y == 1])

  res <- .cyclopsTestProportionality(object$interface, NULL, transformedTimes)
  nCovariates <- 1 # TODO Remove
  res$hessian <- matrix(res$hessian, nrow = (nCovariates + 1))

  if (any(abs(res$gradient[1:nCovariates]) > 1E-5)) {
    stop(
      "Internal state of Cyclops 'object' is not at its mode: ",
      res$gradient
    )
  }

  u <- c(rep(0, nCovariates), res$gradient[nCovariates + 1])
  test <- drop(solve(res$hessian, u) %*% u)
  df <- 1

  tbl <- cbind(test, df, pchisq(test, df, lower.tail = FALSE))

  names <- as.character(getCovariateIds(object$cyclopsData)[1])
  if (!is.null(object$cyclopsData$coefficientNames)) {
    names <- object$coefficientNames[1]
  }

  dimnames(tbl) <- list(names, c("chisq", "df", "p"))
  res$table <- tbl

  class(res) <- "cyclopsZph"

  return(res)
}