#' Check parameter ranges
#'
#' @param ranges A named list of length-two numeric vectors `c(lower,
#'   upper)`, one per parameter.
#' @return `ranges`, invisibly.
#' @keywords internal
check.ranges <- function(ranges) {
  if (!is.list(ranges) || is.null(names(ranges)) || any(names(ranges) == "")) {
    stop("`ranges` must be a named list.", call. = FALSE)
  }
  for (name in names(ranges)) {
    r <- ranges[[name]]
    if (!is.numeric(r) || length(r) != 2L || anyNA(r)) {
      stop(sprintf("Range of `%s` must be two non-missing numbers.", name), call. = FALSE)
    }
    if (r[[1L]] >= r[[2L]]) {
      stop(sprintf("Range of `%s` must be increasing.", name), call. = FALSE)
    }
  }
  invisible(ranges)
}

#' Map design points between the unit cube and parameter ranges
#'
#' @param design A data frame with one column per parameter.
#' @param ranges A named list of `c(lower, upper)` ranges, see
#'   [check.ranges()].
#' @return A data frame of the same shape as `design`.
#' @rdname design.scaling
#' @export
design.to.ranges <- function(design, ranges) {
  check.ranges(ranges)
  out <- design
  for (name in names(ranges)) {
    lower <- ranges[[name]][[1L]]
    upper <- ranges[[name]][[2L]]
    out[[name]] <- lower + design[[name]] * (upper - lower)
  }
  out
}

#' @rdname design.scaling
#' @export
design.to.unit <- function(design, ranges) {
  check.ranges(ranges)
  out <- design
  for (name in names(ranges)) {
    lower <- ranges[[name]][[1L]]
    upper <- ranges[[name]][[2L]]
    out[[name]] <- (design[[name]] - lower) / (upper - lower)
  }
  out
}

#' Generate an experimental design
#'
#' @param n Number of design points; for `"grid"` the number of levels per
#'   parameter.
#' @param ranges A named list of `c(lower, upper)` ranges, see
#'   [check.ranges()].
#' @param type `"grid"` (full factorial), `"random"` (uniform),
#'   `"lhs"` (Latin hypercube) or `"oat"` (one parameter at a time around
#'   the centre).
#' @return A data frame with one column per parameter, in parameter units.
#' @export
design.points <- function(n, ranges, type = c("lhs", "random", "grid", "oat")) {
  type <- match.arg(type)
  check.ranges(ranges)
  k <- length(ranges)
  params <- names(ranges)

  unit <- switch(type,
    grid = {
      levels <- seq(0, 1, length.out = n)
      expand.grid(rep(list(levels), k), KEEP.OUT.ATTRS = FALSE)
    },
    random = {
      draws <- matrix(stats::runif(n * k), nrow = n, ncol = k)
      as.data.frame(draws)
    },
    lhs = {
      draws <- vapply(seq_len(k), function(j) (sample.int(n) - stats::runif(n)) / n,
        numeric(n))
      as.data.frame(matrix(draws, nrow = n, ncol = k))
    },
    oat = {
      levels <- seq(0, 1, length.out = n)
      blocks <- lapply(seq_len(k), function(j) {
        draws <- matrix(0.5, nrow = n, ncol = k)
        draws[, j] <- levels
        draws
      })
      as.data.frame(do.call(rbind, blocks))
    }
  )
  names(unit) <- params
  design.to.ranges(unit, ranges)
}

#' One-at-a-time effects
#'
#' Moves each parameter from the lower to the upper end of its range while
#' the others stay at `base`, and reports the change in the model output.
#'
#' @param fn A function of a named list of parameters returning one number.
#' @param ranges A named list of `c(lower, upper)` ranges, see
#'   [check.ranges()].
#' @param base A named list of baseline values for all parameters.
#' @return A data frame with columns `parameter`, `low`, `high` and
#'   `effect` (`high - low`), sorted by decreasing absolute effect.
#' @export
oat.effects <- function(fn, ranges, base) {
  check.ranges(ranges)
  absent <- setdiff(names(ranges), names(base))
  if (length(absent) > 0L) {
    stop(sprintf("`base` has no value for `%s`.", absent[[1L]]), call. = FALSE)
  }
  rows <- list()
  for (name in names(ranges)) {
    # Lower end of the range
    low <- base
    low[[name]] <- ranges[[name]][[1L]]
    low.value <- fn(low)
    # Upper end of the range
    high <- base
    high[[name]] <- ranges[[name]][[2L]]
    high.value <- fn(high)
    rows[[name]] <- data.frame(parameter = name, low = low.value, high = high.value)
  }

  out <- do.call(rbind, rows)
  out$effect <- out$high - out$low
  out <- out[order(-abs(out$effect)), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Sensitivity of an outcome to design parameters
#'
#' @param design A data frame with one column per parameter.
#' @param outcome Numeric vector of model outputs, one per design row.
#' @param method `"pearson"` or `"spearman"` correlation, or `"prcc"`
#'   (partial rank correlation, controlling for the other parameters).
#' @return A named numeric vector with one coefficient per parameter.
#' @export
sensitivity.coef <- function(design, outcome, method = c("prcc", "pearson", "spearman")) {
  method <- match.arg(method)
  if (nrow(design) != length(outcome)) {
    stop("`outcome` must have one value per row of `design`.", call. = FALSE)
  }
  params <- names(design)

  coefs <- switch(method,
    pearson = {
      vapply(params, function(p) stats::cor(design[[p]], outcome), numeric(1))
    },
    spearman = {
      vapply(params, function(p) {
        stats::cor(design[[p]], outcome, method = "spearman")
      }, numeric(1))
    },
    prcc = {
      ranked <- as.data.frame(lapply(design, rank))
      y <- rank(outcome)
      vapply(params, function(p) {
        others <- ranked[setdiff(params, p)]
        rx <- stats::residuals(stats::lm(ranked[[p]] ~ ., data = others))
        ry <- stats::residuals(stats::lm(y ~ ., data = others))
        stats::cor(rx, ry)
      }, numeric(1))
    }
  )
  coefs
}
