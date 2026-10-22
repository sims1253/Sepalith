#' Poisson offspring distribution
#'
#' Describes a branching process in which each individual has a Poisson
#' number of offspring.
#'
#' @param mean Mean number of offspring per individual.
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.poisson <- function(mean) {
  stopifnot(mean >= 0)
  return(structure(list(
    name = "poisson",
    mean = mean,
    r = function(n) stats::rpois(n, mean),
    pgf = function(s) exp(mean * (s - 1))
  ), class = "offspring"))
}

#' Negative binomial offspring distribution
#'
#' Describes a branching process with overdispersed offspring counts, the
#' usual model for superspreading.
#'
#' @param mean Mean number of offspring per individual.
#' @param size Dispersion parameter; smaller values mean more variation.
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.negbin <- function(mean, size) {
  stopifnot(mean >= 0, size > 0)
  return(structure(list(
    name = "negbin",
    mean = mean,
    r = function(n) stats::rnbinom(n, size = size, mu = mean),
    pgf = function(s) (1 + mean / size * (1 - s))^(-size)
  ), class = "offspring"))
}

#' Geometric offspring distribution
#'
#' Describes a branching process with geometric offspring counts on
#' `0, 1, 2, ...`.
#'
#' @param mean Mean number of offspring per individual.
#' @return A list of class `offspring` with a sampler `r` and a probability
#'   generating function `pgf`.
#' @family offspring distributions
#' @export
offspring.geometric <- function(mean) {
  stopifnot(mean >= 0)
  p <- 1 / (1 + mean)
  return(structure(list(
    name = "geometric",
    mean = mean,
    r = function(n) stats::rgeom(n, p),
    pgf = function(s) p / (1 - (1 - p) * s)
  ), class = "offspring"))
}

#' Print an offspring distribution
#'
#' @param x An `offspring` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.offspring <- function(x, ...) {
  regime <- if (x$mean < 1) {
    "subcritical"
  } else if (x$mean > 1) {
    "supercritical"
  } else {
    "critical"
  }
  cat("<offspring> ", x$name, " with mean ", format(x$mean), " (", regime, ")\n",
    sep = "")
  return(invisible(x))
}

#' Simulate generation sizes of a Galton-Watson process
#'
#' Starts from a number of founders and draws offspring for every
#' individual in every generation. Stops at extinction or once the
#' population exceeds a cap, which is treated as a major outbreak.
#'
#' @param offspring An `offspring` object, e.g. from [offspring.poisson()].
#' @param generations Maximum number of generations.
#' @param founders Number of individuals in generation zero.
#' @param cap Population size above which simulation stops.
#' @return A list with elements `sizes` (generation sizes), `extinct`,
#'   `capped` and `total`.
#' @family branching processes
#' @export
bp.simulate <- function(offspring, generations = 50, founders = 1, cap = 1e5) {
  stopifnot(inherits(offspring, "offspring"))
  sizes <- rep(NA_real_, generations + 1)
  sizes[1] <- founders
  extinct <- FALSE
  capped <- FALSE
  for (g in seq_len(generations)) {
    current <- sizes[g]
    if (current == 0) {
      extinct <- TRUE
      break
    }
    if (current > cap) {
      capped <- TRUE
      break
    }
    sizes[g + 1] <- sum(offspring$r(current))
  }
  sizes <- sizes[!is.na(sizes)]
  if (!extinct && !capped && utils::tail(sizes, 1) == 0) {
    extinct <- TRUE
  }
  return(list(
    sizes = sizes,
    extinct = extinct,
    capped = capped,
    total = sum(sizes)
  ))
}

#' Extinction probability of a branching process
#'
#' Finds the smallest fixed point of the probability generating function on
#' `[0, 1]` by iterating `q <- pgf(q)` from zero.
#'
#' @param offspring An `offspring` object, e.g. from [offspring.poisson()].
#' @param founders Number of individuals in generation zero.
#' @param tol Convergence tolerance.
#' @param max.iter Maximum number of iterations.
#' @return The probability that the process eventually dies out.
#' @family branching processes
#' @export
bp.extinction.prob <- function(offspring, founders = 1, tol = 1e-10, max.iter = 10000) {
  stopifnot(inherits(offspring, "offspring"))
  if (offspring$mean <= 1) {
    return(1)
  }
  q <- 0
  for (i in seq_len(max.iter)) {
    q.new <- offspring$pgf(q)
    if (abs(q.new - q) < tol) {
      q <- q.new
      break
    }
    q <- q.new
  }
  return(q^founders)
}

#' Compare simulated and theoretical extinction
#'
#' Runs many branching processes and contrasts the share that went extinct
#' with the extinction probability from [bp.extinction.prob()].
#'
#' @param offspring An `offspring` object, e.g. from [offspring.poisson()].
#' @param reps Number of simulated processes.
#' @param founders Number of individuals in generation zero.
#' @param generations Maximum number of generations.
#' @return A list with elements `simulated`, `theoretical`, `se` and
#'   `totals` (total progeny of the extinct runs).
#' @family branching processes
#' @export
bp.check.extinction <- function(offspring, reps = 1000, founders = 1, generations = 100) {
  extinct <- logical(reps)
  totals <- rep(NA_real_, reps)
  for (r in seq_len(reps)) {
    run <- bp.simulate(offspring, generations = generations, founders = founders)
    extinct[r] <- run$extinct
    if (run$extinct) {
      totals[r] <- run$total
    }
  }
  p.hat <- mean(extinct)
  return(list(
    simulated = p.hat,
    theoretical = bp.extinction.prob(offspring, founders = founders),
    se = sqrt(p.hat * (1 - p.hat) / reps),
    totals = totals[!is.na(totals)]
  ))
}
