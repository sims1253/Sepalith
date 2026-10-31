#' Log-likelihood of a sample under a parametric family
#'
#' Parameters outside their valid range give a log-likelihood of `-Inf`.
#'
#' @param x A numeric vector of observations.
#' @param family One of `"normal"`, `"lognormal"`, `"exponential"`,
#'   `"gamma"`, `"poisson"` or `"uniform"`.
#' @param par A named numeric vector of parameters, named as in the
#'   `estimate` element of [fit.dist()].
#' @keywords internal
#' @return A single number.
#' @noRd
dist.loglik <- function(x, family = c("normal", "lognormal", "exponential", "gamma",
                                      "poisson", "uniform"), par) {
  family <- match.arg(family)

  # Invalid parameters short-circuit to a log-likelihood of -Inf
  dens <- switch(family,
    normal = {
      if (par[["sd"]] <= 0) return(-Inf)
      stats::dnorm(x, mean = par[["mean"]], sd = par[["sd"]], log = TRUE)
    },
    lognormal = {
      if (par[["sdlog"]] <= 0) return(-Inf)
      stats::dlnorm(x, meanlog = par[["meanlog"]], sdlog = par[["sdlog"]], log = TRUE)
    },
    exponential = {
      if (par[["rate"]] <= 0) return(-Inf)
      stats::dexp(x, rate = par[["rate"]], log = TRUE)
    },
    gamma = {
      if (par[["shape"]] <= 0 || par[["rate"]] <= 0) return(-Inf)
      stats::dgamma(x, shape = par[["shape"]], rate = par[["rate"]], log = TRUE)
    },
    poisson = {
      if (par[["lambda"]] < 0) return(-Inf)
      stats::dpois(x, lambda = par[["lambda"]], log = TRUE)
    },
    uniform = {
      if (par[["max"]] <= par[["min"]]) return(-Inf)
      stats::dunif(x, min = par[["min"]], max = par[["max"]], log = TRUE)
    }
  )

  sum(dens)
}

#' Fit a parametric distribution
#'
#' @description
#' Closed-form maximum likelihood estimates, except for the gamma family,
#' which uses the method of moments.
#'
#' @param x A numeric vector of observations.
#' @param family One of `"normal"`, `"lognormal"`, `"exponential"`,
#'   `"gamma"`, `"poisson"` or `"uniform"`.
#' @return A list of class `"dist.fit"` with elements `family`, `estimate`,
#'   `loglik`, `n` and `k` (the number of parameters).
#' @export
fit.dist <- function(x, family = c("normal", "lognormal", "exponential", "gamma",
                                   "poisson", "uniform")) {
  family <- match.arg(family)
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < 2L) {
    stop("At least two non-missing observations are required.", call. = FALSE)
  }

  estimate <- switch(family,
    normal = {
      m <- mean(x)
      c(mean = m, sd = sqrt(mean((x - m)^2)))
    },
    lognormal = {
      if (any(x <= 0)) stop("`x` must be positive for this family.", call. = FALSE)
      lx <- log(x)
      c(meanlog = mean(lx), sdlog = sqrt(mean((lx - mean(lx))^2)))
    },
    exponential = {
      if (any(x < 0)) stop("`x` must be non-negative for this family.", call. = FALSE)
      c(rate = 1 / mean(x))
    },
    gamma = {
      if (any(x <= 0)) stop("`x` must be positive for this family.", call. = FALSE)
      m <- mean(x)
      v <- stats::var(x)
      c(shape = m^2 / v, rate = m / v)
    },
    poisson = {
      if (any(x < 0)) stop("`x` must be non-negative for this family.", call. = FALSE)
      c(lambda = mean(x))
    },
    uniform = {
      c(min = min(x), max = max(x))
    }
  )

  structure(
    list(
      family = family,
      estimate = estimate,
      loglik = dist.loglik(x, family, estimate),
      n = n,
      k = length(estimate)
    ),
    class = "dist.fit"
  )
}

#' Information criteria for a distribution fit
#'
#' @param fit A `"dist.fit"` object from [fit.dist()].
#' @export
#' @return A named numeric vector with elements `aic`, `aicc`, `bic` and
#'   `hqc` (Hannan-Quinn).
dist.ic <- function(fit) {
  deviance <- -2 * fit$loglik
  k <- fit$k
  n <- fit$n
  c(
    aic = deviance + 2 * k,
    aicc = deviance + 2 * k + 2 * k * (k + 1) / (n - k - 1),
    bic = deviance + log(n) * k,
    hqc = deviance + 2 * log(log(n)) * k
  )
}

#' Compare several distribution fits on one sample
#'
#' @param x A numeric vector of observations.
#' @param families Character vector of families to try; families whose
#'   support excludes the data are dropped with a warning.
#' @export
#' @return A data frame with columns `family`, `loglik`, `aic`, `bic` and
#'   `delta.aic`, ordered by `aic`.
compare.fits <- function(x, families = c("normal", "lognormal", "gamma", "exponential")) {
  rows <- list()
  for (fam in families) {
    fit <- tryCatch(fit.dist(x, fam), error = function(e) {
      warning(sprintf("Family `%s` skipped: %s", fam, conditionMessage(e)), call. = FALSE)
      NULL
    })
    if (is.null(fit)) {
      next
    }
    ic <- dist.ic(fit)
    rows[[fam]] <- data.frame(family = fam, loglik = fit$loglik, aic = ic[["aic"]],
                              bic = ic[["bic"]], stringsAsFactors = FALSE)
  }

  # Best fit first
  out <- do.call(rbind, rows)
  out <- out[order(out$aic), , drop = FALSE]
  out$delta.aic <- out$aic - out$aic[1L]
  rownames(out) <- NULL
  out
}

#' Quantiles of a fitted distribution
#'
#' @param p Numeric vector of probabilities.
#' @param fit A `"dist.fit"` object from [fit.dist()].
#' @keywords internal
dist.quantile <- function(p, fit) {
  par <- fit$estimate
  switch(fit$family,
    normal = {
      stats::qnorm(p, mean = par[["mean"]], sd = par[["sd"]])
    },
    lognormal = {
      stats::qlnorm(p, meanlog = par[["meanlog"]], sdlog = par[["sdlog"]])
    },
    exponential = {
      stats::qexp(p, rate = par[["rate"]])
    },
    gamma = {
      stats::qgamma(p, shape = par[["shape"]], rate = par[["rate"]])
    },
    poisson = {
      stats::qpois(p, lambda = par[["lambda"]])
    },
    uniform = {
      stats::qunif(p, min = par[["min"]], max = par[["max"]])
    }
  )
}

#' Points for a quantile-quantile plot against a fitted distribution
#'
#' @param x A numeric vector of observations.
#' @param fit A `"dist.fit"` object from [fit.dist()].
#' @param a Offset in the plotting positions `(i - a) / (n + 1 - 2 * a)`.
#' @export
#' @return A data frame with columns `theoretical`, `sample` and
#'   `residual`, sorted by `sample`.
qq.points <- function(x, fit, a = 0.5) {
  x <- sort(x[!is.na(x)])
  n <- length(x)
  pp <- (seq_len(n) - a) / (n + 1 - 2 * a)
  theoretical <- dist.quantile(pp, fit)

  data.frame(
    theoretical = theoretical,
    sample = x,
    residual = x - theoretical
  )
}
