#' Butcher tableau of an explicit Runge-Kutta method
#'
#' @param method One of `"euler"`, `"midpoint"`, `"heun"`, `"ralston"` or
#'   `"rk4"`.
#' @return A list with elements `a` (the stage matrix), `b` (the weights),
#'   `c` (the nodes) and `order`.
#' @export
ode.tableau <- function(method = c("euler", "midpoint", "heun", "ralston", "rk4")) {
  method <- match.arg(method)
  switch(method,
    euler = list(
      a = matrix(0, 1, 1),
      b = 1,
      c = 0,
      order = 1L
    ),
    midpoint = list(
      a = matrix(c(0, 1 / 2, 0, 0), 2, 2),
      b = c(0, 1),
      c = c(0, 1 / 2),
      order = 2L
    ),
    heun = list(
      a = matrix(c(0, 1, 0, 0), 2, 2),
      b = c(1 / 2, 1 / 2),
      c = c(0, 1),
      order = 2L
    ),
    ralston = list(
      a = matrix(c(0, 2 / 3, 0, 0), 2, 2),
      b = c(1 / 4, 3 / 4),
      c = c(0, 2 / 3),
      order = 2L
    ),
    rk4 = list(
      a = rbind(c(0, 0, 0, 0), c(1 / 2, 0, 0, 0), c(0, 1 / 2, 0, 0), c(0, 0, 1, 0)),
      b = c(1, 2, 2, 1) / 6,
      c = c(0, 1 / 2, 1 / 2, 1),
      order = 4L
    )
  )
}

#' One explicit Runge-Kutta step
#'
#' @description
#' Each method evaluates the stages of the matching [ode.tableau()] and
#' combines them with its weights.
#'
#' @param f The right-hand side, a function `f(t, y)` returning a vector the
#'   length of `y`.
#' @param t The current time.
#' @param y The current state.
#' @param h The step size.
#' @param method One of `"euler"`, `"midpoint"`, `"heun"`, `"ralston"` or
#'   `"rk4"`.
#' @return The state at time `t + h`.
#' @export
ode.step <- function(f, t, y, h, method = c("euler", "midpoint", "heun", "ralston", "rk4")) {
  method <- match.arg(method)
  k1 <- f(t, y)
  switch(method,
    euler = {
      y + h * k1
    },
    midpoint = {
      k2 <- f(t + h / 2, y + h / 2 * k1)
      y + h * k2
    },
    heun = {
      k2 <- f(t + h, y + h * k1)
      y + h * (k1 + k2) / 2
    },
    ralston = {
      k2 <- f(t + 2 * h / 3, y + 2 * h / 3 * k1)
      y + h * (k1 + 3 * k2) / 4
    },
    rk4 = {
      k2 <- f(t + h / 2, y + h / 2 * k1)
      k3 <- f(t + h / 2, y + h / 2 * k2)
      k4 <- f(t + h, y + h * k3)
      y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    }
  )
}

#' Stability function of an explicit Runge-Kutta method
#'
#' @param z Complex or real values of `h * lambda`.
#' @param method A method understood by [ode.tableau()].
#' @return The complex amplification factors
#'   `R(z) = 1 + z * b' (I - z a)^(-1) 1`, so that the method applied to
#'   `y' = lambda y` gives `y[n + 1] = R(h lambda) y[n]`.
#' @export
ode.stability <- function(z, method = "rk4") {
  tab <- ode.tableau(method)
  s <- length(tab$b)
  vapply(as.complex(z), function(zi) {
    1 + zi * sum(tab$b * solve(diag(s) - zi * tab$a, rep(1, s)))
  }, complex(1))
}

#' Integrate an ODE on a fixed grid
#'
#' @inheritParams ode.step
#' @param y0 The initial state.
#' @param times Increasing vector of output times; the first is the initial
#'   time.
#' @param substeps Number of steps taken between consecutive output times.
#' @return A matrix with a `time` column followed by one column per state
#'   component.
#' @export
ode.solve <- function(f, y0, times, method = "rk4", substeps = 1L) {
  # Validate `y0`
  if (!is.numeric(y0) || length(y0) < 1L || anyNA(y0)) {
    stop(sprintf("`%s` must be a non-empty numeric vector.", "y0"), call. = FALSE)
  }

  # Validate `times`
  if (!is.numeric(times) || length(times) < 2L || is.unsorted(times, strictly = TRUE)) {
    stop(sprintf("`%s` must be a strictly increasing numeric vector.", "times"), call. = FALSE)
  }

  # Validate `substeps`
  if (!is.numeric(substeps) || length(substeps) != 1L || !(substeps >= 1)) {
    stop(sprintf("`%s` must be a single positive whole number.", "substeps"), call. = FALSE)
  }

  labels <- names(y0)
  if (is.null(labels)) {
    labels <- paste0("y", seq_along(y0))
  }
  out <- matrix(NA_real_, length(times), length(y0) + 1L,
                dimnames = list(NULL, c("time", labels)))
  out[1, ] <- c(times[1], y0)
  y <- y0
  for (i in seq_along(times)[-1]) {
    h <- (times[i] - times[i - 1L]) / substeps
    for (j in seq_len(substeps)) {
      y <- ode.step(f, times[i - 1L] + (j - 1L) * h, y, h, method)
    }
    out[i, ] <- c(times[i], y)
  }
  out
}

#' Adaptive Heun-Euler integration
#'
#' @description
#' Each step computes the Euler and Heun solutions; their difference is the
#' error estimate. A rejected step shrinks `h`, an accepted step may grow
#' it, both by the factor `safety * (tol / err)^(1 / 2)` clamped to
#' `[shrink, grow]`.
#'
#' @inheritParams ode.solve
#' @param t.span A vector `c(start, end)`.
#' @param h0 Initial step size.
#' @param tol Absolute error tolerance per step.
#' @param safety,shrink,grow Step-size control constants.
#' @param max.steps Maximum number of attempted steps.
#' @return A list with elements `t`, `y` (a matrix with one row per accepted
#'   time), `accepted` and `rejected`.
#' @export
ode.adaptive <- function(f, y0, t.span, h0 = 0.1, tol = 1e-6, safety = 0.9,
                         shrink = 0.2, grow = 5, max.steps = 10000L) {
  t <- t.span[1]
  y <- y0
  h <- h0
  times <- t
  states <- list(y0)
  accepted <- 0L
  rejected <- 0L

  for (step in seq_len(max.steps)) {
    if (t >= t.span[2]) break
    h <- min(h, t.span[2] - t)
    k1 <- f(t, y)
    k2 <- f(t + h, y + h * k1)
    y.low <- y + h * k1
    y.high <- y + h * (k1 + k2) / 2
    err <- max(abs(y.high - y.low))
    factor <- if (err == 0) grow else safety * (tol / err)^(1 / 2)

    if (err <= tol) {
      t <- t + h
      y <- y.high
      times <- c(times, t)
      states[[length(states) + 1L]] <- y
      accepted <- accepted + 1L
      h <- h * min(grow, max(1, factor))
    } else {
      rejected <- rejected + 1L
      h <- h * max(shrink, min(1, factor))
    }
  }
  list(t = times, y = do.call(rbind, states), accepted = accepted, rejected = rejected)
}

#' Empirical order of convergence of a method
#'
#' @inheritParams ode.solve
#' @param t.end The final time; integration starts at zero.
#' @param exact The exact state at `t.end`.
#' @param steps Increasing vector of step counts.
#' @return A data frame with columns `steps`, `h`, `error` and `order` (the
#'   observed order between consecutive rows; `NA` in the first row).
#' @export
ode.order <- function(f, y0, t.end, exact, method = "rk4", steps = c(10L, 20L, 40L, 80L)) {
  h <- t.end / steps
  error <- vapply(steps, function(n) {
    sol <- ode.solve(f, y0, c(0, t.end), method = method, substeps = n)
    max(abs(sol[2L, -1L] - exact))
  }, numeric(1))
  order <- c(NA_real_, log(error[-length(error)] / error[-1]) / log(h[-length(h)] / h[-1]))
  data.frame(steps = steps, h = h, error = error, order = order)
}
