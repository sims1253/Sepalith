#' Fit a ridge regression
#'
#' Fits a linear model with an L2 penalty on all coefficients except the
#' intercept. The result is a small S3 object with broom-style tidiers.
#'
#' @param formula A two-sided model formula.
#' @param data A data frame containing the variables in `formula`.
#' @param lambda A non-negative penalty strength.
#' @return An object of class `ts_ridge`.
#' @export
fit_ridge <- function(formula, data, lambda = 1) {
  check_formula(formula)
  if (!is.numeric(lambda) || length(lambda) != 1L || is.na(lambda) || lambda < 0) {
    stop("`lambda` must be a single non-negative number.", call. = FALSE)
  }

  mf <- stats::model.frame(formula, data = data)
  y <- stats::model.response(mf)
  X <- stats::model.matrix(formula, mf)

  # Leave the intercept unpenalised so the fit is not pulled towards zero.
  penalty <- diag(lambda, ncol(X))
  if ("(Intercept)" %in% colnames(X)) {
    penalty[1L, 1L] <- 0
  }

  xtx <- crossprod(X)
  xtx_inv <- solve(xtx + penalty)
  beta <- drop(xtx_inv %*% crossprod(X, y))
  fitted <- drop(X %*% beta)
  resid <- y - fitted
  # Effective degrees of freedom: trace of the ridge hat matrix.
  edf <- sum(diag(xtx_inv %*% xtx))
  sigma2 <- sum(resid^2) / (length(y) - edf)

  structure(
    list(
      coefficients = beta,
      fitted.values = fitted,
      residuals = resid,
      vcov = sigma2 * xtx_inv %*% xtx %*% xtx_inv,
      sigma = sqrt(sigma2),
      edf = edf,
      lambda = lambda,
      nobs = length(y),
      terms = attr(mf, "terms"),
      xlevels = stats::.getXlevels(attr(mf, "terms"), mf),
      call = match.call()
    ),
    class = "ts_ridge"
  )
}

#' Tidy a ridge regression
#'
#' Returns one row per coefficient with its standard error, a t statistic
#' based on the effective residual degrees of freedom, and optional intervals.
#'
#' @param x An object of class `ts_ridge`.
#' @param conf.int Whether to add confidence interval columns.
#' @param conf.level Confidence level for the interval.
#' @param ... Unused, for generic consistency.
#' @return A data frame with columns `term`, `estimate`, `std.error`,
#'   `statistic`, `p.value` and optionally `conf.low`, `conf.high`.
#' @importFrom generics tidy
#' @export
tidy.ts_ridge <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  est <- x$coefficients
  se <- sqrt(diag(x$vcov))

  out <- data.frame(
    term = names(est),
    estimate = unname(est),
    std.error = unname(se),
    statistic = unname(est / se),
    stringsAsFactors = FALSE
  )

  # Ridge estimates are biased, so these tests are only approximate.
  df_resid <- x$nobs - x$edf
  out$p.value <- 2 * stats::pt(abs(out$statistic), df = df_resid, lower.tail = FALSE)
  if (conf.int) {
    crit <- stats::qt(1 - (1 - conf.level) / 2, df = df_resid)
    out$conf.low <- out$estimate - crit * out$std.error
    out$conf.high <- out$estimate + crit * out$std.error
  }

  rownames(out) <- NULL
  out
}

#' Glance at a ridge regression
#'
#' Summarises the fit in a single row: variance explained, residual scale,
#' penalty and effective degrees of freedom.
#'
#' @param x An object of class `ts_ridge`.
#' @param ... Unused, for generic consistency.
#' @return A one-row data frame.
#' @importFrom generics glance
#' @export
glance.ts_ridge <- function(x, ...) {
  y <- x$fitted.values + x$residuals
  rss <- sum(x$residuals^2)
  tss <- sum((y - mean(y))^2)

  data.frame(
    r.squared = 1 - rss / tss,
    sigma = x$sigma,
    lambda = x$lambda,
    edf = x$edf,
    df.residual = x$nobs - x$edf,
    nobs = x$nobs
  )
}

#' Augment data with ridge predictions
#'
#' Adds fitted values and residuals to the original data, or predictions to
#' new data. The original data must be the complete cases used in the fit.
#' Rows of `newdata` with missing predictors get a missing prediction.
#'
#' @param x An object of class `ts_ridge`.
#' @param data The data frame used to fit `x`.
#' @param newdata Optional data frame to predict for instead of `data`.
#' @param ... Unused, for generic consistency.
#' @return The input data frame with `.fitted` and, for training data,
#'   `.resid` and `.std.resid` columns.
#' @importFrom generics augment
#' @export
augment.ts_ridge <- function(x, data = NULL, newdata = NULL, ...) {
  if (!is.null(newdata)) {
    # Keep every row of newdata and reuse the training factor levels.
    rhs <- stats::delete.response(x$terms)
    mf <- stats::model.frame(rhs, newdata, na.action = stats::na.pass, xlev = x$xlevels)
    X <- stats::model.matrix(rhs, mf)
    out <- newdata
    out$.fitted <- drop(X %*% x$coefficients)
    return(out)
  }

  if (is.null(data)) {
    stop("Supply either `data` or `newdata`.", call. = FALSE)
  }
  if (nrow(data) != x$nobs) {
    stop("`data` must have the same rows as the fitted model.", call. = FALSE)
  }

  out <- data
  out$.fitted <- x$fitted.values
  out$.resid <- x$residuals
  out$.std.resid <- x$residuals / x$sigma
  out
}

#' Tidy a likelihood-ratio test
#'
#' Returns the test as a one-row data frame in the usual broom layout.
#'
#' @param x An object of class `ts_lr_test`.
#' @param ... Unused, for generic consistency.
#' @return A one-row data frame.
#' @importFrom generics tidy
#' @export
tidy.ts_lr_test <- function(x, ...) {
  data.frame(
    statistic = x$statistic,
    df = x$df,
    p.value = x$p.value,
    method = "Likelihood-ratio test",
    stringsAsFactors = FALSE
  )
}

#' Glance at a likelihood-ratio test
#'
#' Returns the sample size and the decision at the conventional 5% level.
#'
#' @param x An object of class `ts_lr_test`.
#' @param ... Unused, for generic consistency.
#' @return A one-row data frame.
#' @export
glance.ts_lr_test <- function(x, ...) {
  data.frame(
    nobs = x$nobs,
    df = x$df,
    reject = x$p.value < 0.05,
    stringsAsFactors = FALSE
  )
}

#' Tidy a model comparison table
#'
#' Drops the custom class and renames the criterion column after the
#' criterion that was used.
#'
#' @param x An object of class `ts_ic_table`.
#' @param ... Unused, for generic consistency.
#' @return A plain data frame.
#' @export
tidy.ts_ic_table <- function(x, ...) {
  criterion <- attr(x, "criterion")

  # Strip the class so the result prints like any other data frame.
  out <- x
  class(out) <- "data.frame"
  attr(out, "criterion") <- NULL
  names(out)[names(out) == "ic"] <- tolower(criterion)
  names(out)[names(out) == "logLik"] <- "loglik"
  out
}
