#' Validate the arguments of a binomial interval
#'
#' @param x Number of successes.
#' @param n Number of trials.
#' @param conf.level Confidence level in `(0, 1)`.
#' @keywords internal
#' @noRd
ci.check <- function(x, n, conf.level) {
  # Validate `x`
  if (length(x) != 1L || is.na(x) || x < 0 || x != round(x)) {
    stop("`x` must be a single non-negative whole number.", call. = FALSE)
  }
  # Validate `n`
  if (length(n) != 1L || is.na(n) || n < 1 || n != round(n)) {
    stop("`n` must be a single positive whole number.", call. = FALSE)
  }
  if (x > n) {
    stop("`x` must not exceed `n`.", call. = FALSE)
  }
  # Validate `conf.level`
  if (length(conf.level) != 1L || is.na(conf.level) || conf.level <= 0 || conf.level >= 1) {
    stop("`conf.level` must be a single number in (0, 1).", call. = FALSE)
  }
  invisible(TRUE)
}

#' Confidence interval for a binomial proportion
#'
#' @param x Number of successes.
#' @param n Number of trials.
#' @param conf.level Confidence level in `(0, 1)`.
#' @param method One of `"wald"`, `"wilson"`, `"agresti"` (Agresti-Coull),
#'   `"clopper"` (Clopper-Pearson) or `"jeffreys"`.
#' @return A named numeric vector with elements `estimate`, `lower` and
#'   `upper`, with bounds clipped to `[0, 1]`.
#' @family interval methods
#' @export
ci.prop <- function(x, n, conf.level = 0.95,
                    method = c("wald", "wilson", "agresti", "clopper", "jeffreys")) {
  method <- match.arg(method)
  ci.check(x, n, conf.level)
  alpha <- 1 - conf.level
  z <- stats::qnorm(1 - alpha / 2)
  p <- x / n

  bounds <- switch(method,
    wald = {
      half <- z * sqrt(p * (1 - p) / n)
      c(p - half, p + half)
    },
    wilson = {
      center <- (x + z^2 / 2) / (n + z^2)
      half <- z * sqrt(n) / (n + z^2) * sqrt(p * (1 - p) + z^2 / (4 * n))
      c(center - half, center + half)
    },
    agresti = {
      n.tilde <- n + z^2
      p.tilde <- (x + z^2 / 2) / n.tilde
      half <- z * sqrt(p.tilde * (1 - p.tilde) / n.tilde)
      c(p.tilde - half, p.tilde + half)
    },
    clopper = {
      lower <- if (x == 0) 0 else stats::qbeta(alpha / 2, x, n - x + 1)
      upper <- if (x == n) 1 else stats::qbeta(1 - alpha / 2, x + 1, n - x)
      c(lower, upper)
    },
    jeffreys = {
      lower <- if (x == 0) 0 else stats::qbeta(alpha / 2, x + 0.5, n - x + 0.5)
      upper <- if (x == n) 1 else stats::qbeta(1 - alpha / 2, x + 0.5, n - x + 0.5)
      c(lower, upper)
    }
  )
  c(estimate = p, lower = max(0, bounds[1]), upper = min(1, bounds[2]))
}

#' Confidence interval for a Poisson rate
#'
#' @param count Observed number of events.
#' @param exposure Total exposure (person-time), positive.
#' @param conf.level Confidence level in `(0, 1)`.
#' @param method One of `"wald"`, `"score"`, `"exact"` or `"byar"`.
#' @return A named numeric vector with elements `estimate`, `lower` and
#'   `upper`, expressed per unit of exposure.
#' @family interval methods
#' @export
ci.rate <- function(count, exposure, conf.level = 0.95,
                    method = c("wald", "score", "exact", "byar")) {
  method <- match.arg(method)
  # Validate `count`
  if (length(count) != 1L || is.na(count) || count < 0 || count != round(count)) {
    stop("`count` must be a single non-negative whole number.", call. = FALSE)
  }
  # Validate `exposure`
  if (length(exposure) != 1L || is.na(exposure) || exposure <= 0) {
    stop("`exposure` must be a single positive number.", call. = FALSE)
  }
  alpha <- 1 - conf.level
  z <- stats::qnorm(1 - alpha / 2)

  bounds <- switch(method,
    wald = {
      half <- z * sqrt(count)
      c(count - half, count + half)
    },
    score = {
      half <- z * sqrt(count + z^2 / 4)
      c(count + z^2 / 2 - half, count + z^2 / 2 + half)
    },
    exact = {
      lower <- if (count == 0) 0 else stats::qchisq(alpha / 2, 2 * count) / 2
      upper <- stats::qchisq(1 - alpha / 2, 2 * (count + 1)) / 2
      c(lower, upper)
    },
    byar = {
      lower <- if (count == 0) 0 else
        count * (1 - 1 / (9 * count) - z / (3 * sqrt(count)))^3
      upper <- (count + 1) * (1 - 1 / (9 * (count + 1)) + z / (3 * sqrt(count + 1)))^3
      c(lower, upper)
    }
  )
  c(estimate = count / exposure, lower = max(0, bounds[1]) / exposure,
    upper = bounds[2] / exposure)
}

#' Confidence interval for a difference of two proportions
#'
#' @param x1,n1 Successes and trials in the first sample.
#' @param x2,n2 Successes and trials in the second sample.
#' @param conf.level Confidence level in `(0, 1)`.
#' @param method One of `"wald"`, `"agresti.caffo"`, `"hauck.anderson"` or
#'   `"newcombe"`.
#' @return A named numeric vector with elements `estimate`, `lower` and
#'   `upper` for `p1 - p2`.
#' @family interval methods
#' @export
ci.diff.prop <- function(x1, n1, x2, n2, conf.level = 0.95,
                         method = c("wald", "agresti.caffo", "hauck.anderson", "newcombe")) {
  method <- match.arg(method)
  ci.check(x1, n1, conf.level)
  ci.check(x2, n2, conf.level)
  z <- stats::qnorm(1 - (1 - conf.level) / 2)
  p1 <- x1 / n1
  p2 <- x2 / n2
  d <- p1 - p2

  bounds <- switch(method,
    wald = {
      half <- z * sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
      c(d - half, d + half)
    },
    agresti.caffo = {
      q1 <- (x1 + 1) / (n1 + 2)
      q2 <- (x2 + 1) / (n2 + 2)
      half <- z * sqrt(q1 * (1 - q1) / (n1 + 2) + q2 * (1 - q2) / (n2 + 2))
      c(q1 - q2 - half, q1 - q2 + half)
    },
    hauck.anderson = {
      cc <- 1 / (2 * min(n1, n2))
      half <- z * sqrt(p1 * (1 - p1) / (n1 - 1) + p2 * (1 - p2) / (n2 - 1)) + cc
      c(d - half, d + half)
    },
    newcombe = {
      w1 <- ci.prop(x1, n1, conf.level, method = "wilson")
      w2 <- ci.prop(x2, n2, conf.level, method = "wilson")
      lower <- d - sqrt((p1 - w1[["lower"]])^2 + (w2[["upper"]] - p2)^2)
      upper <- d + sqrt((w1[["upper"]] - p1)^2 + (p2 - w2[["lower"]])^2)
      c(lower, upper)
    }
  )
  c(estimate = d, lower = max(-1, bounds[1]), upper = min(1, bounds[2]))
}

#' Compare binomial interval methods on one sample
#'
#' @param x Number of successes.
#' @param n Number of trials.
#' @param methods Character vector of methods accepted by [ci.prop()].
#' @param conf.level Confidence level in `(0, 1)`.
#' @return A data frame with columns `method`, `lower`, `upper` and `width`,
#'   ordered by `width`.
#' @keywords internal
ci.compare <- function(x, n, methods = c("wald", "wilson", "agresti", "clopper", "jeffreys"),
                       conf.level = 0.95) {
  lower <- upper <- numeric(length(methods))
  for (i in seq_along(methods)) {
    ci <- ci.prop(x, n, conf.level = conf.level, method = methods[i])
    lower[i] <- ci[["lower"]]
    upper[i] <- ci[["upper"]]
  }

  # Narrowest interval first
  out <- data.frame(method = methods, lower = lower, upper = upper,
                    width = upper - lower, stringsAsFactors = FALSE)
  out <- out[order(out$width), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Exact coverage probability of a binomial interval
#'
#' @description
#' Sums the binomial probabilities of every outcome `0:n` whose interval
#' contains the true proportion `p`.
#'
#' @param p True proportion in `[0, 1]`.
#' @param n Number of trials.
#' @param method A method accepted by [ci.prop()].
#' @param conf.level Confidence level in `(0, 1)`.
#' @export
ci.coverage <- function(p, n, method = "wilson", conf.level = 0.95) {
  outcomes <- 0:n
  covers <- vapply(outcomes, function(k) {
    ci <- ci.prop(k, n, conf.level = conf.level, method = method)
    ci[["lower"]] <= p && p <= ci[["upper"]]
  }, logical(1))
  sum(stats::dbinom(outcomes[covers], n, p))
}
