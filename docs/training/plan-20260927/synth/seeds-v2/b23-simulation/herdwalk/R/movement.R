#' Boundary helpers
#'
#' Put coordinates that left the arena back inside. `wrap_coord()` treats
#' the arena as a torus, `reflect_coord()` mirrors at the walls as if the
#' agent bounced off them, and `clamp_coord()` stops agents at the wall they
#' tried to cross.
#'
#' @param coord Numeric vector of coordinates along one axis.
#' @param limit Size of the arena along that axis.
#' @return A numeric vector of coordinates inside `[0, limit]`.
#' @keywords internal
wrap_coord <- function(coord, limit) {
  coord %% limit
}

#' @rdname wrap_coord
reflect_coord <- function(coord, limit) {
  # Fold onto one period of the mirrored arena, then flip the far half.
  period <- 2 * limit
  folded <- coord %% period
  ifelse(folded > limit, period - folded, folded)
}

#' @rdname wrap_coord
clamp_coord <- function(coord, limit) {
  pmin(pmax(coord, 0), limit)
}

#' Apply a boundary rule to a population
#'
#' Puts every agent back inside the arena using one of the supported
#' boundary conditions.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population with corrected `x` and `y` columns.
#' @keywords internal
apply_boundary <- function(pop, boundary = c("wrap", "reflect", "clamp")) {
  boundary <- match.arg(boundary)
  arena <- attr(pop, "arena")
  fix <- switch(boundary,
    wrap = wrap_coord,
    reflect = reflect_coord,
    clamp = clamp_coord
  )

  pop$x <- fix(pop$x, arena[["width"]])
  pop$y <- fix(pop$y, arena[["height"]])
  pop
}

#' Move agents by a Gaussian random walk
#'
#' Each agent takes an independent step with normally distributed x and y
#' displacements. Removed agents do not move.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param step_sd Standard deviation of the step along each axis.
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population after one movement step.
#' @export
move_random <- function(pop, step_sd = 1, boundary = "wrap") {
  mobile <- pop$state != "R"
  n_mobile <- sum(mobile)
  if (n_mobile == 0) {
    return(pop)
  }

  pop$x[mobile] <- pop$x[mobile] + stats::rnorm(n_mobile, sd = step_sd)
  pop$y[mobile] <- pop$y[mobile] + stats::rnorm(n_mobile, sd = step_sd)
  apply_boundary(pop, boundary)
}

#' Move agents by a Levy flight
#'
#' Step lengths follow a Pareto distribution and headings are uniform, so
#' most moves are short with occasional long jumps. Removed agents do not
#' move.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param alpha Tail exponent of the step-length distribution.
#' @param min_step Smallest possible step length.
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population after one movement step.
#' @export
move_levy <- function(pop, alpha = 1.5, min_step = 0.5, boundary = "wrap") {
  mobile <- pop$state != "R"
  n_mobile <- sum(mobile)
  if (n_mobile == 0) {
    return(pop)
  }

  # Inverse-CDF draw from a Pareto(min_step, alpha) distribution.
  lengths <- min_step * stats::runif(n_mobile)^(-1 / alpha)
  heading <- stats::runif(n_mobile, 0, 2 * pi)

  pop$x[mobile] <- pop$x[mobile] + lengths * cos(heading)
  pop$y[mobile] <- pop$y[mobile] + lengths * sin(heading)
  apply_boundary(pop, boundary)
}

#' Move agents toward an attractor
#'
#' Agents drift a fixed fraction of the way toward a point of interest,
#' such as a market or a water hole, plus a small amount of noise.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param target Numeric vector of length two giving the attractor position.
#' @param pull Fraction of the remaining distance covered per step.
#' @param step_sd Standard deviation of the step along each axis.
#' @param boundary Boundary condition: `"wrap"`, `"reflect"` or `"clamp"`.
#' @return The population after one movement step.
#' @export
move_toward <- function(pop, target, pull = 0.1, step_sd = 0.5, boundary = "clamp") {
  stopifnot(length(target) == 2, pull >= 0, pull <= 1)
  mobile <- pop$state != "R"
  n_mobile <- sum(mobile)
  if (n_mobile == 0) {
    return(pop)
  }

  dx <- target[[1]] - pop$x[mobile]
  dy <- target[[2]] - pop$y[mobile]
  pop$x[mobile] <- pop$x[mobile] + pull * dx + stats::rnorm(n_mobile, sd = step_sd)
  pop$y[mobile] <- pop$y[mobile] + pull * dy + stats::rnorm(n_mobile, sd = step_sd)
  apply_boundary(pop, boundary)
}

#' Mean squared displacement over a trajectory
#'
#' Given a list of population snapshots, computes the mean squared distance
#' of each agent from its starting position at every step. Agents absent
#' from a snapshot are dropped from that step's average.
#'
#' @param snapshots A list of population data frames, one per time step.
#' @return A data frame with columns `step` and `msd`.
#' @export
mean_squared_displacement <- function(snapshots) {
  if (length(snapshots) == 0) {
    stop("`snapshots` must contain at least one population.", call. = FALSE)
  }

  origin <- snapshots[[1]]
  msd <- numeric(length(snapshots))
  for (i in seq_along(snapshots)) {
    current <- snapshots[[i]]
    # Match by id: agents may have been added or removed along the way.
    idx <- match(current$id, origin$id)
    ok <- !is.na(idx)
    if (!any(ok)) {
      msd[i] <- NA_real_
      next
    }

    dx <- current$x[ok] - origin$x[idx[ok]]
    dy <- current$y[ok] - origin$y[idx[ok]]
    msd[i] <- mean(dx^2 + dy^2)
  }

  data.frame(step = seq_along(snapshots) - 1L, msd = msd)
}
