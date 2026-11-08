#' Build root-finding tolerances
#'
#' @param x_tol Absolute tolerance on the change in `x` between iterations.
#' @param f_tol Absolute tolerance on `|f(x)|`.
#' @param max_iter Maximum number of iterations.
#' @param max_step Largest step an open method may take in one iteration.
#' @param damping Factor applied to the step of an open method, in `(0, 1]`.
#' @return A list with elements `x_tol`, `f_tol`, `max_iter`, `max_step` and
#'   `damping`.
#' @export
root_control <- function(x_tol = 1e-10, f_tol = 1e-12, max_iter = 100L,
                         max_step = Inf, damping = 1) {
  # Validate `x_tol`
  if (!is.numeric(x_tol) || length(x_tol) != 1L || !(x_tol > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "x_tol"), call. = FALSE)
  }

  # Validate `f_tol`
  if (!is.numeric(f_tol) || length(f_tol) != 1L || !(f_tol > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "f_tol"), call. = FALSE)
  }

  # Validate `max_iter`
  if (!is.numeric(max_iter) || length(max_iter) != 1L || !(max_iter >= 1)) {
    stop(sprintf("`%s` must be a single positive number.", "max_iter"), call. = FALSE)
  }

  # Validate `max_step`
  if (!is.numeric(max_step) || length(max_step) != 1L || !(max_step > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "max_step"), call. = FALSE)
  }

  # Validate `damping`
  if (!is.numeric(damping) || length(damping) != 1L || !(damping > 0 && damping <= 1)) {
    stop(sprintf("`%s` must be a single number in (0, 1].", "damping"), call. = FALSE)
  }
  return(list(x_tol = x_tol, f_tol = f_tol, max_iter = as.integer(max_iter),
              max_step = max_step, damping = damping))
}

#' Check that an interval brackets a root
#'
#' @param f A function of one argument.
#' @param lower Lower end of the bracket.
#' @param upper Upper end of the bracket.
#' @return A list with elements `lower`, `upper`, `f_lower` and `f_upper`.
#' @keywords internal
check_bracket <- function(f, lower, upper) {
  # Validate `lower`
  if (!is.numeric(lower) || length(lower) != 1L || !is.finite(lower)) {
    stop(sprintf("`%s` must be a single finite number.", "lower"), call. = FALSE)
  }

  # Validate `upper`
  if (!is.numeric(upper) || length(upper) != 1L || !is.finite(upper)) {
    stop(sprintf("`%s` must be a single finite number.", "upper"), call. = FALSE)
  }
  if (lower >= upper) {
    stop("`lower` must be smaller than `upper`.", call. = FALSE)
  }

  # Evaluate `f` at `lower`
  f_lower <- f(lower)
  if (!is.finite(f_lower)) {
    stop(sprintf("`f(%s)` is not finite.", "lower"), call. = FALSE)
  }

  # Evaluate `f` at `upper`
  f_upper <- f(upper)
  if (!is.finite(f_upper)) {
    stop(sprintf("`f(%s)` is not finite.", "upper"), call. = FALSE)
  }

  if (sign(f_lower) == sign(f_upper)) {
    stop("`f` must change sign between `lower` and `upper`.", call. = FALSE)
  }
  return(list(lower = lower, upper = upper, f_lower = f_lower, f_upper = f_upper))
}

#' Find a root inside a bracket
#'
#' @description
#' All methods keep a sign-changing bracket `[a, b]`; they differ only in how
#' the next trial point `x` is chosen.
#'
#' @param f A function of one argument.
#' @param lower Lower end of the bracket.
#' @param upper Upper end of the bracket.
#' @param method One of `"bisection"`, `"false_position"`, `"illinois"` or
#'   `"ridders"`.
#' @param control A list from [root_control()].
#' @return A list with elements `root`, `f_root`, `iterations`, `converged`
#'   and `method`.
#' @export
find_root <- function(f, lower, upper,
                      method = c("bisection", "false_position", "illinois", "ridders"),
                      control = root_control()) {
  method <- match.arg(method)
  br <- check_bracket(f, lower, upper)
  a <- br$lower
  b <- br$upper
  fa <- br$f_lower
  fb <- br$f_upper
  side <- 0L
  converged <- FALSE

  for (iter in seq_len(control$max_iter)) {
    switch(method,
      bisection = {
        x <- (a + b) / 2
      },
      false_position = {
        x <- (a * fb - b * fa) / (fb - fa)
      },
      illinois = {
        # Halve the value at an end point that was kept twice in a row
        if (side <= -2L) fb <- fb / 2
        if (side >= 2L) fa <- fa / 2
        x <- (a * fb - b * fa) / (fb - fa)
      },
      ridders = {
        m <- (a + b) / 2
        fm <- f(m)
        x <- m + (m - a) * sign(fa - fb) * fm / sqrt(fm^2 - fa * fb)
      }
    )

    fx <- f(x)
    if (sign(fx) == sign(fa)) {
      a <- x
      fa <- fx
      side <- min(side, 0L) - 1L
    } else {
      b <- x
      fb <- fx
      side <- max(side, 0L) + 1L
    }
    if (abs(b - a) < control$x_tol || abs(fx) < control$f_tol) {
      converged <- TRUE
      break
    }
  }
  return(list(root = x, f_root = fx, iterations = iter, converged = converged,
              method = method))
}

#' Open root-finding iterations
#'
#' @description
#' Iterates from a starting value without keeping a bracket. Convergence is
#' declared when both the step and the residual criteria hold.
#'
#' @param f A function of one argument.
#' @param x0 Starting value.
#' @param df Derivative of `f`; needed only for `method = "newton"`.
#' @param method One of `"newton"`, `"secant"` or `"steffensen"`.
#' @param control A list from [root_control()].
#' @return A list with elements `root`, `f_root`, `iterations`, `converged`
#'   and `method`.
#' @export
iterate_root <- function(f, x0, df = NULL, method = c("newton", "secant", "steffensen"),
                         control = root_control()) {
  method <- match.arg(method)
  if (method == "newton" && !is.function(df)) {
    stop("`df` must be a function when `method = \"newton\"`.", call. = FALSE)
  }
  x <- x0
  fx <- f(x)
  x_prev <- x0 + 1e-4 * max(1, abs(x0))
  f_prev <- f(x_prev)
  converged <- FALSE

  for (iter in seq_len(control$max_iter)) {
    switch(method,
      newton = {
        slope <- df(x)
      },
      secant = {
        slope <- (fx - f_prev) / (x - x_prev)
      },
      steffensen = {
        slope <- (f(x + fx) - fx) / fx
      }
    )
    if (!is.finite(slope) || slope == 0) {
      break
    }

    step <- control$damping * fx / slope
    step <- sign(step) * min(abs(step), control$max_step)
    x_prev <- x
    f_prev <- fx
    x <- x - step
    fx <- f(x)
    checks <- c(step = abs(step) < control$x_tol, value = abs(fx) < control$f_tol)
    if (all(checks)) {
      converged <- TRUE
      break
    }
  }
  return(list(root = x, f_root = fx, iterations = iter, converged = converged,
              method = method))
}

#' Grow an interval until it brackets a root
#'
#' @param f A function of one argument.
#' @param x0 Centre of the initial interval.
#' @param step Initial half-width of the interval.
#' @param grow Factor by which each end moves further out.
#' @param max_iter Maximum number of expansions.
#' @return A numeric vector `c(lower, upper)`.
#' @export
expand_bracket <- function(f, x0, step = 0.1, grow = 1.6, max_iter = 50L) {
  # Validate `step`
  if (!is.numeric(step) || length(step) != 1L || !(step > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "step"), call. = FALSE)
  }

  # Validate `grow`
  if (!is.numeric(grow) || length(grow) != 1L || !(grow > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "grow"), call. = FALSE)
  }
  lower <- x0 - step
  upper <- x0 + step
  f_lower <- f(lower)
  f_upper <- f(upper)

  for (iter in seq_len(max_iter)) {
    if (sign(f_lower) != sign(f_upper)) {
      return(c(lower = lower, upper = upper))
    }
    # Move whichever end has the smaller absolute value
    if (abs(f_lower) < abs(f_upper)) {
      lower <- lower - grow * (upper - lower)
      f_lower <- f(lower)
    } else {
      upper <- upper + grow * (upper - lower)
      f_upper <- f(upper)
    }
  }
  stop(sprintf("No sign change found after %d expansions.", max_iter), call. = FALSE)
}

#' Find all roots on a grid
#'
#' @param f A vectorised function of one argument.
#' @param lower Lower end of the search range.
#' @param upper Upper end of the search range.
#' @param n_grid Number of grid intervals scanned for sign changes.
#' @param method A bracketing method passed to [find_root()].
#' @return A numeric vector of roots in increasing order.
#' @export
find_all_roots <- function(f, lower, upper, n_grid = 100L, method = "illinois") {
  grid <- seq(lower, upper, length.out = n_grid + 1L)
  values <- f(grid)
  roots <- grid[values == 0]

  changes <- which(sign(values[-1]) * sign(values[-length(values)]) < 0)
  for (i in changes) {
    fit <- find_root(f, grid[i], grid[i + 1L], method = method)
    roots <- c(roots, fit$root)
  }
  return(sort(roots))
}
