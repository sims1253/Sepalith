#' Parameter names of the supported distributions
#'
#' @return A named list mapping each distribution to the names of its
#'   parameters.
#' @keywords internal
dist.param.names <- function() {
  list(
    normal = c("mean", "sd"),
    lognormal = c("meanlog", "sdlog"),
    gamma = c("shape", "rate"),
    beta = c("shape1", "shape2"),
    uniform = c("min", "max"),
    poisson = c("lambda")
  )
}

#' Check distribution parameters
#'
#' @param dist Distribution name, one of the names of [dist.param.names()].
#' @param params A named list of parameter values.
#' @return `params`, restricted to the parameters of `dist`, invisibly.
#' @keywords internal
check.dist.params <- function(dist = c("normal", "lognormal", "gamma", "beta",
                                       "uniform", "poisson"), params) {
  dist <- match.arg(dist)
  required <- dist.param.names()[[dist]]
  absent <- setdiff(required, names(params))
  if (length(absent) > 0L) {
    stop(sprintf("The %s distribution needs parameter `%s`.", dist, absent[[1L]]),
      call. = FALSE)
  }
  need <- function(ok, name, what) {
    if (!isTRUE(ok)) {
      stop(sprintf("Parameter `%s` of the %s distribution must be %s.", name, dist, what),
        call. = FALSE)
    }
  }

  switch(dist,
    normal = {
      need(is.finite(params$mean), "mean", "finite")
      need(params$sd > 0, "sd", "positive")
    },
    lognormal = {
      need(is.finite(params$meanlog), "meanlog", "finite")
      need(params$sdlog > 0, "sdlog", "positive")
    },
    gamma = {
      need(params$shape > 0, "shape", "positive")
      need(params$rate > 0, "rate", "positive")
    },
    beta = {
      need(params$shape1 > 0, "shape1", "positive")
      need(params$shape2 > 0, "shape2", "positive")
    },
    uniform = {
      need(is.finite(params$min), "min", "finite")
      need(is.finite(params$max), "max", "finite")
    },
    poisson = {
      need(params$lambda > 0, "lambda", "positive")
    }
  )
  invisible(params[required])
}

#' Draw a random sample
#'
#' @param n Sample size.
#' @param dist One of `"normal"`, `"lognormal"`, `"gamma"`, `"beta"`,
#'   `"uniform"` or `"poisson"`.
#' @param params A named list of parameter values, see [dist.param.names()].
#' @return A numeric vector of length `n`.
#' @family samplers
#' @export
draw.sample <- function(n, dist = c("normal", "lognormal", "gamma", "beta", "uniform",
                                    "poisson"), params = list()) {
  dist <- match.arg(dist)
  p <- check.dist.params(dist, params)

  switch(dist,
    normal = {
      stats::rnorm(n, mean = p$mean, sd = p$sd)
    },
    lognormal = {
      stats::rlnorm(n, meanlog = p$meanlog, sdlog = p$sdlog)
    },
    gamma = {
      stats::rgamma(n, shape = p$shape, rate = p$rate)
    },
    beta = {
      stats::rbeta(n, shape1 = p$shape1, shape2 = p$shape2)
    },
    uniform = {
      stats::runif(n, min = p$min, max = p$max)
    },
    poisson = {
      stats::rpois(n, lambda = p$lambda)
    }
  )
}

#' Theoretical moments of a distribution
#'
#' @inheritParams draw.sample
#' @return A single number.
#' @rdname dist.moments
#' @family samplers
#' @export
dist.mean <- function(dist = c("normal", "lognormal", "gamma", "beta", "uniform",
                               "poisson"), params = list()) {
  dist <- match.arg(dist)
  p <- check.dist.params(dist, params)

  switch(dist,
    normal = {
      p$mean
    },
    lognormal = {
      exp(p$meanlog + p$sdlog^2 / 2)
    },
    gamma = {
      p$shape / p$rate
    },
    beta = {
      total <- p$shape1 + p$shape2
      p$shape1 / total
    },
    uniform = {
      (p$min + p$max) / 2
    },
    poisson = {
      p$lambda
    }
  )
}

#' @rdname dist.moments
#' @export
dist.var <- function(dist = c("normal", "lognormal", "gamma", "beta", "uniform",
                              "poisson"), params = list()) {
  dist <- match.arg(dist)
  p <- check.dist.params(dist, params)

  switch(dist,
    normal = {
      p$sd^2
    },
    lognormal = {
      (exp(p$sdlog^2) - 1) * exp(2 * p$meanlog + p$sdlog^2)
    },
    gamma = {
      p$shape / p$rate^2
    },
    beta = {
      total <- p$shape1 + p$shape2
      p$shape1 * p$shape2 / (total^2 * (total + 1))
    },
    uniform = {
      (p$max - p$min)^2 / 12
    },
    poisson = {
      p$lambda
    }
  )
}

#' Fit a distribution by the method of moments
#'
#' @param x A numeric sample.
#' @param dist One of `"normal"`, `"lognormal"`, `"gamma"`, `"beta"`,
#'   `"uniform"` or `"poisson"`.
#' @return A named list of parameter values that can be passed to
#'   [draw.sample()].
#' @family samplers
#' @export
fit.moments <- function(x, dist = c("normal", "lognormal", "gamma", "beta", "uniform",
                                    "poisson")) {
  dist <- match.arg(dist)
  x <- x[!is.na(x)]
  if (length(x) < 2L) {
    stop("`x` needs at least two non-missing values.", call. = FALSE)
  }
  m <- mean(x)
  v <- stats::var(x)

  switch(dist,
    normal = {
      list(mean = m, sd = sqrt(v))
    },
    lognormal = {
      sdlog2 <- log(1 + v / m^2)
      list(meanlog = log(m) - sdlog2 / 2, sdlog = sqrt(sdlog2))
    },
    gamma = {
      list(shape = m^2 / v, rate = m / v)
    },
    beta = {
      common <- m * (1 - m) / v - 1
      list(shape1 = m * common, shape2 = (1 - m) * common)
    },
    uniform = {
      half <- sqrt(3 * v)
      list(min = m - half, max = m + half)
    },
    poisson = {
      list(lambda = m)
    }
  )
}
