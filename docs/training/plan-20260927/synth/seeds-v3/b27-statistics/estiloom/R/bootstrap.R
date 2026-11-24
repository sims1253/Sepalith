#' Bootstrap replicates of a statistic
#'
#' @param x A numeric vector.
#' @param stat A function of one vector returning a single number.
#' @param R Number of bootstrap resamples.
#' @param seed Optional random seed for reproducible resampling.
#' @rdname boot.reps
#' @keywords internal
boot.reps <- function(x, stat = mean, R = 999L, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  x <- x[!is.na(x)]
  n <- length(x)
  vapply(seq_len(R), function(i) stat(x[sample.int(n, n, replace = TRUE)]), numeric(1))
}

#' @rdname boot.reps
#' @keywords internal
jack.reps <- function(x, stat = mean) {
  x <- x[!is.na(x)]
  if (length(x) < 2L) {
    stop("`x` must have at least 2 non-missing values.", call. = FALSE)
  }
  vapply(seq_along(x), function(i) stat(x[-i]), numeric(1))
}

#' Bootstrap bias and standard error
#'
#' @inheritParams boot.reps
#' @return A list with elements `estimate`, `bias`, `se` and `corrected`
#'   (the bias-corrected estimate).
#' @export
boot.bias.se <- function(x, stat = mean, R = 999L, seed = NULL) {
  estimate <- stat(x[!is.na(x)])
  reps <- boot.reps(x, stat = stat, R = R, seed = seed)
  bias <- mean(reps) - estimate

  list(
    estimate = estimate,
    bias = bias,
    se = stats::sd(reps),
    corrected = estimate - bias
  )
}

#' Bootstrap confidence interval
#'
#' @inheritParams boot.reps
#' @param conf.level Confidence level in `(0, 1)`.
#' @param type One of `"percentile"`, `"basic"`, `"normal"` or `"bca"`
#'   (bias-corrected and accelerated).
#' @return A named numeric vector with elements `estimate`, `lower` and
#'   `upper`.
#' @export
boot.ci <- function(x, stat = mean, R = 999L, conf.level = 0.95,
                    type = c("percentile", "basic", "normal", "bca"), seed = NULL) {
  type <- match.arg(type)
  if (conf.level <= 0 || conf.level >= 1) {
    stop("`conf.level` must be in (0, 1).", call. = FALSE)
  }
  if (R < 10L) {
    stop("`R` must be at least 10.", call. = FALSE)
  }

  estimate <- stat(x[!is.na(x)])
  reps <- boot.reps(x, stat = stat, R = R, seed = seed)
  alpha <- 1 - conf.level
  probs <- c(alpha / 2, 1 - alpha / 2)

  bounds <- switch(type,
    percentile = {
      stats::quantile(reps, probs, names = FALSE)
    },
    basic = {
      q <- stats::quantile(reps, probs, names = FALSE)
      c(2 * estimate - q[2], 2 * estimate - q[1])
    },
    normal = {
      z <- stats::qnorm(probs[2])
      bias <- mean(reps) - estimate
      c(estimate - bias - z * stats::sd(reps), estimate - bias + z * stats::sd(reps))
    },
    bca = {
      z0 <- stats::qnorm(mean(reps < estimate))
      jack <- jack.reps(x, stat = stat)
      d <- mean(jack) - jack
      accel <- sum(d^3) / (6 * sum(d^2)^1.5)
      zq <- stats::qnorm(probs)
      adj <- stats::pnorm(z0 + (z0 + zq) / (1 - accel * (z0 + zq)))
      stats::quantile(reps, adj, names = FALSE)
    }
  )
  c(estimate = estimate, lower = bounds[1], upper = bounds[2])
}

#' Bootstrap intervals for several columns
#'
#' @param data A data frame.
#' @param cols Character vector of numeric column names.
#' @inheritParams boot.ci
#' @return A data frame with columns `column`, `n`, `estimate`, `lower`
#'   and `upper`.
#' @export
boot.columns <- function(data, cols = names(data), stat = mean, R = 999L,
                         conf.level = 0.95, type = "percentile", seed = NULL) {
  n <- estimate <- lower <- upper <- rep(NA_real_, length(cols))

  for (i in seq_along(cols)) {
    x <- data[[cols[i]]]
    if (!is.numeric(x)) {
      stop(sprintf("Column `%s` must be numeric.", cols[i]), call. = FALSE)
    }
    n[i] <- sum(!is.na(x))
    if (n[i] < 2) {
      next
    }
    # Reuse the seed per column so results do not depend on column order
    ci <- boot.ci(x, stat = stat, R = R, conf.level = conf.level, type = type, seed = seed)
    estimate[i] <- ci[["estimate"]]
    lower[i] <- ci[["lower"]]
    upper[i] <- ci[["upper"]]
  }

  data.frame(column = cols, n = as.integer(n), estimate = estimate,
             lower = lower, upper = upper, stringsAsFactors = FALSE)
}

#' Bootstrap interval for a difference between two samples
#'
#' @description
#' Resamples `x` and `y` independently and applies `stat` to each.
#'
#' @param x,y Numeric vectors.
#' @param stat A function of one vector returning a single number.
#' @param R Number of bootstrap resamples.
#' @param conf.level Confidence level in `(0, 1)`.
#' @param seed Optional random seed for reproducible resampling.
#' @return A list with elements `estimate.x`, `estimate.y`, `se.x`, `se.y`,
#'   `estimate`, `lower`, `upper` and `prop.positive`, where `estimate` is
#'   `stat(x) - stat(y)`, the bounds are percentile bounds and
#'   `prop.positive` is the share of replicates with a positive difference.
#' @export
boot.diff <- function(x, y, stat = mean, R = 999L, conf.level = 0.95, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]

  reps.x <- reps.y <- numeric(R)
  for (r in seq_len(R)) {
    # Resample `x`
    bx <- x[sample.int(length(x), length(x), replace = TRUE)]
    reps.x[r] <- stat(bx)
    # Resample `y`
    by <- y[sample.int(length(y), length(y), replace = TRUE)]
    reps.y[r] <- stat(by)
  }

  # Percentile bounds of the replicate differences
  diffs <- reps.x - reps.y
  alpha <- 1 - conf.level
  q <- stats::quantile(diffs, c(alpha / 2, 1 - alpha / 2), names = FALSE)

  list(
    estimate.x = stat(x),
    estimate.y = stat(y),
    se.x = stats::sd(reps.x),
    se.y = stats::sd(reps.y),
    estimate = stat(x) - stat(y),
    lower = q[1],
    upper = q[2],
    prop.positive = mean(diffs > 0)
  )
}
