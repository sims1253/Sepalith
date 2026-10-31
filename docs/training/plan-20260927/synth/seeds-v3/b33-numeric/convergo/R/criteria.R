#' Convergence criteria for iterative methods
#'
#' @param abs.tol Absolute tolerance on the step norm.
#' @param rel.tol Relative tolerance on the step norm, scaled by the norm of
#'   the current iterate.
#' @param f.tol Tolerance on the residual or function value norm.
#' @param max.iter Maximum number of iterations.
#' @param stall.iter Number of iterations without improvement of the residual
#'   after which the method is stopped.
#' @return An object of class `conv_criteria`: a list with elements
#'   `abs.tol`, `rel.tol`, `f.tol`, `max.iter` and `stall.iter`.
#' @export
conv.criteria <- function(abs.tol = 1e-10, rel.tol = 1e-8, f.tol = 1e-10,
                          max.iter = 500L, stall.iter = 25L) {
  values <- list(abs.tol = abs.tol, rel.tol = rel.tol, f.tol = f.tol,
                 max.iter = max.iter, stall.iter = stall.iter)

  # Validate `abs.tol`
  if (!is.numeric(abs.tol) || length(abs.tol) != 1L || !(abs.tol >= 0)) {
    stop(sprintf("`%s` must be a single non-negative number.", "abs.tol"), call. = FALSE)
  }

  # Validate `rel.tol`
  if (!is.numeric(rel.tol) || length(rel.tol) != 1L || !(rel.tol >= 0)) {
    stop(sprintf("`%s` must be a single non-negative number.", "rel.tol"), call. = FALSE)
  }

  # Validate `f.tol`
  if (!is.numeric(f.tol) || length(f.tol) != 1L || !(f.tol >= 0)) {
    stop(sprintf("`%s` must be a single non-negative number.", "f.tol"), call. = FALSE)
  }

  # Validate `max.iter`
  if (!is.numeric(max.iter) || length(max.iter) != 1L || !(max.iter >= 1)) {
    stop(sprintf("`%s` must be a single positive whole number.", "max.iter"), call. = FALSE)
  }

  # Validate `stall.iter`
  if (!is.numeric(stall.iter) || length(stall.iter) != 1L || !(stall.iter >= 1)) {
    stop(sprintf("`%s` must be a single positive whole number.", "stall.iter"), call. = FALSE)
  }
  values$max.iter <- as.integer(max.iter)
  values$stall.iter <- as.integer(stall.iter)
  structure(values, class = "conv_criteria")
}

#' Test an iteration state against the criteria
#'
#' @description
#' Runs every check in parallel so the caller can report which criteria
#' held. The method has `converged` when the step criteria (`abs` or `rel`)
#' and the residual criterion `f` hold together; it has `stopped` when it
#' converged, diverged (non-finite residual), hit `max.iter` or stalled.
#'
#' @param state A list with elements `iter`, `step` (step norm), `x.norm`
#'   (norm of the current iterate), `f.norm` (residual norm) and `stall`
#'   (iterations since the residual last improved).
#' @param criteria An object from [conv.criteria()].
#' @return A list with elements `checks` (named logical vector with
#'   elements `abs`, `rel`, `f`, `diverged`, `iter` and `stall`),
#'   `converged`, `stopped` and `reason`.
#' @export
conv.test <- function(state, criteria = conv.criteria()) {
  checks <- c(
    abs = state$step <= criteria$abs.tol,
    rel = state$step <= criteria$rel.tol * max(state$x.norm, .Machine$double.eps),
    f = is.finite(state$f.norm) && state$f.norm <= criteria$f.tol,
    diverged = !is.finite(state$f.norm),
    iter = state$iter >= criteria$max.iter,
    stall = state$stall >= criteria$stall.iter
  )
  converged <- (checks[["abs"]] || checks[["rel"]]) && checks[["f"]]
  reason <- if (converged) {
    "converged"
  } else if (checks[["diverged"]]) {
    "diverged"
  } else if (checks[["iter"]]) {
    "max.iter"
  } else if (checks[["stall"]]) {
    "stalled"
  } else {
    "running"
  }
  list(checks = checks, converged = converged, stopped = reason != "running",
       reason = reason)
}

#' Iteration traces
#'
#' A trace stores one row of named diagnostics per iteration.
#'
#' @param fields Character vector naming the diagnostics to record.
#' @return An object of class `conv_trace` with elements `fields`, `iter`
#'   and `values` (a matrix with one column per field).
#' @export
trace.new <- function(fields) {
  if (!is.character(fields) || anyDuplicated(fields) || "iter" %in% fields) {
    stop("`fields` must be unique names other than \"iter\".", call. = FALSE)
  }
  structure(list(fields = fields, iter = integer(0),
                 values = matrix(numeric(0), 0L, length(fields),
                                 dimnames = list(NULL, fields))),
            class = "conv_trace")
}

#' @rdname trace.new
#' @param trace An object of class `conv_trace`.
#' @param iter The iteration number.
#' @param values A named numeric vector with one element per field.
#' @export
trace.add <- function(trace, iter, values) {
  stopifnot(inherits(trace, "conv_trace"))
  missing.fields <- setdiff(trace$fields, names(values))
  if (length(missing.fields) > 0L) {
    stop(sprintf("`values` lacks the fields: %s.", paste(missing.fields, collapse = ", ")),
         call. = FALSE)
  }
  trace$iter <- c(trace$iter, as.integer(iter))
  trace$values <- rbind(trace$values, values[trace$fields])
  trace
}

#' Accelerate a convergent sequence
#'
#' @param x A numeric sequence of iterates.
#' @param method One of `"aitken"`, `"cesaro"`, `"richardson"` or `"wynn"`.
#' @return The accelerated sequence, shorter than `x` by the number of
#'   terms each method consumes.
#' @export
accelerate <- function(x, method = c("aitken", "cesaro", "richardson", "wynn")) {
  method <- match.arg(method)
  n <- length(x)
  switch(method,
    aitken = {
      d1 <- diff(x)
      d2 <- diff(x, differences = 2L)
      x[seq_len(n - 2L)] - d1[seq_len(n - 2L)]^2 / d2
    },
    cesaro = {
      cumsum(x) / seq_len(n)
    },
    richardson = {
      k <- seq_len(n - 1L)
      (k + 1) * x[k + 1L] - k * x[k]
    },
    wynn = {
      # Two columns of the epsilon table give the Shanks transform e_1
      eps1 <- 1 / diff(x)
      x[2:(n - 1L)] + 1 / diff(eps1)
    }
  )
}

#' Fixed-point iteration
#'
#' @param g A function mapping a numeric vector to one of the same length.
#' @param x0 Starting value.
#' @param damping Weight of the new iterate, in `(0, 1]`.
#' @param criteria An object from [conv.criteria()].
#' @return A list with elements `x`, `iter`, `reason` and `trace` (a data
#'   frame with columns `iter`, `step` and `residual`).
#' @export
fixed.point <- function(g, x0, damping = 1, criteria = conv.criteria()) {
  # Validate `x0`
  if (!is.numeric(x0) || length(x0) < 1L || anyNA(x0)) {
    stop(sprintf("`%s` must be a non-empty numeric vector.", "x0"), call. = FALSE)
  }

  # Validate `damping`
  if (!is.numeric(damping) || length(damping) != 1L || !(damping > 0 && damping <= 1)) {
    stop(sprintf("`%s` must be a single number in (0, 1].", "damping"), call. = FALSE)
  }
  x <- x0
  trace <- trace.new(c("step", "residual"))
  best <- Inf
  stall <- 0L
  for (iter in seq_len(criteria$max.iter)) {
    gx <- g(x)
    x.new <- (1 - damping) * x + damping * gx
    step <- sqrt(sum((x.new - x)^2))
    residual <- sqrt(sum((g(x.new) - x.new)^2))
    stall <- if (residual < best) 0L else stall + 1L
    best <- min(best, residual)
    x <- x.new
    trace <- trace.add(trace, iter, c(step = step, residual = residual))
    test <- conv.test(list(iter = iter, step = step, x.norm = sqrt(sum(x^2)),
                           f.norm = residual, stall = stall), criteria)
    if (test$stopped) {
      break
    }
  }
  list(x = x, iter = iter, reason = test$reason,
       trace = data.frame(iter = trace$iter, trace$values))
}
