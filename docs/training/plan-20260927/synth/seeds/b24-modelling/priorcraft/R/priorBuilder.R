#' Construct a prior specification
#'
#' Low-level constructor shared by the family-specific builders. Stores the
#' distribution family, its parameters and the model parameter it applies to.
#'
#' @param family Name of the distribution family.
#' @param params Named numeric vector of distribution parameters.
#' @param class Parameter class, as used by brms (`"b"`, `"Intercept"`, `"sd"`,
#'   `"sigma"`, ...).
#' @param coef Optional coefficient name within `class`.
#' @return An object of class `pcPrior`.
#' @keywords internal
newPrior <- function(family, params, class = "b", coef = NULL) {
  if (!is.numeric(params) || is.null(names(params))) {
    stop("`params` must be a named numeric vector.", call. = FALSE)
  }
  if (anyNA(params)) {
    stop("`params` must not contain missing values.", call. = FALSE)
  }
  prior <- list(
    family = family,
    params = params,
    class = class,
    coef = if (is.null(coef)) "" else coef
  )
  class(prior) <- "pcPrior"
  return(prior)
}

#' Normal prior
#'
#' Builds a normal prior with the given mean and standard deviation.
#'
#' @param mean Location of the distribution.
#' @param sd Scale of the distribution; must be positive.
#' @param class Parameter class, as used by brms (`"b"`, `"Intercept"`, `"sd"`,
#'   `"sigma"`, ...).
#' @param coef Optional coefficient name within `class`.
#' @return An object of class `pcPrior`.
#' @family prior builders
#' @export
normalPrior <- function(mean = 0, sd = 1, class = "b", coef = NULL) {
  if (sd <= 0) {
    stop("`sd` must be positive.", call. = FALSE)
  }
  return(newPrior("normal", c(mean = mean, sd = sd), class = class, coef = coef))
}

#' Student-t prior
#'
#' Builds a Student-t prior with the given degrees of freedom, location and
#' scale.
#'
#' @param df Degrees of freedom; must be positive.
#' @param location Location of the distribution.
#' @param scale Scale of the distribution; must be positive.
#' @param class Parameter class, as used by brms (`"b"`, `"Intercept"`, `"sd"`,
#'   `"sigma"`, ...).
#' @param coef Optional coefficient name within `class`.
#' @return An object of class `pcPrior`.
#' @family prior builders
#' @export
studentPrior <- function(df = 3, location = 0, scale = 2.5, class = "b", coef = NULL) {
  if (df <= 0) {
    stop("`df` must be positive.", call. = FALSE)
  }
  if (scale <= 0) {
    stop("`scale` must be positive.", call. = FALSE)
  }
  params <- c(df = df, location = location, scale = scale)
  return(newPrior("student_t", params, class = class, coef = coef))
}

#' Exponential prior
#'
#' Builds an exponential prior with the given rate, typically used for
#' standard deviations.
#'
#' @param rate Rate of the distribution; must be positive.
#' @param class Parameter class, as used by brms (`"b"`, `"Intercept"`, `"sd"`,
#'   `"sigma"`, ...).
#' @param coef Optional coefficient name within `class`.
#' @return An object of class `pcPrior`.
#' @family prior builders
#' @export
exponentialPrior <- function(rate = 1, class = "sd", coef = NULL) {
  if (rate <= 0) {
    stop("`rate` must be positive.", call. = FALSE)
  }
  return(newPrior("exponential", c(rate = rate), class = class, coef = coef))
}

#' Stan code for a prior
#'
#' Renders a prior as the distribution statement used in Stan and in
#' [brms::set_prior()].
#'
#' @param prior An object of class `pcPrior`.
#' @param digits Number of significant digits for parameters.
#' @return A single string such as `"normal(0, 2.5)"`.
#' @export
priorToStan <- function(prior, digits = 4) {
  stopifnot(inherits(prior, "pcPrior"))
  p <- signif(prior$params, digits)
  stanCode <- switch(prior$family,
    normal = sprintf("normal(%s, %s)", p[["mean"]], p[["sd"]]),
    student_t = sprintf("student_t(%s, %s, %s)", p[["df"]], p[["location"]], p[["scale"]]),
    exponential = sprintf("exponential(%s)", p[["rate"]]),
    stop(sprintf("Unsupported prior family `%s`.", prior$family), call. = FALSE)
  )
  return(stanCode)
}

#' Convert priors for brms
#'
#' Turns a list of prior specifications into a single `brmsprior` object.
#'
#' @param priors A list of `pcPrior` objects.
#' @return A `brmsprior` data frame.
#' @export
toBrmsPrior <- function(priors) {
  if (inherits(priors, "pcPrior")) {
    priors <- list(priors)
  }
  converted <- priors %>%
    purrr::map(function(prior) {
      brms::set_prior(priorToStan(prior), class = prior$class, coef = prior$coef)
    })
  return(Reduce(`+`, converted))
}

#' Convert a prior for rstanarm
#'
#' Translates a single prior specification into the matching rstanarm prior
#' constructor. rstanarm sets priors per argument, so `class` is not used.
#'
#' @param prior An object of class `pcPrior`.
#' @param autoscale Passed to the rstanarm constructor.
#' @return An rstanarm prior list.
#' @export
toRstanarmPrior <- function(prior, autoscale = FALSE) {
  stopifnot(inherits(prior, "pcPrior"))
  p <- prior$params
  out <- switch(prior$family,
    normal = rstanarm::normal(
      location = p[["mean"]],
      scale = p[["sd"]],
      autoscale = autoscale
    ),
    student_t = rstanarm::student_t(
      df = p[["df"]],
      location = p[["location"]],
      scale = p[["scale"]],
      autoscale = autoscale
    ),
    exponential = rstanarm::exponential(
      rate = p[["rate"]],
      autoscale = autoscale
    ),
    stop(sprintf("Unsupported prior family `%s`.", prior$family), call. = FALSE)
  )
  return(out)
}

#' Rescale a prior to the data
#'
#' Multiplies the scale parameter of a prior by the standard deviation of the
#' response, so that a unit prior becomes weakly informative on the data scale.
#'
#' @param prior An object of class `pcPrior`.
#' @param y Numeric response vector.
#' @return The rescaled `pcPrior`.
#' @export
scalePriorToData <- function(prior, y) {
  stopifnot(inherits(prior, "pcPrior"))
  s <- stats::sd(y, na.rm = TRUE)
  if (!is.finite(s) || s == 0) {
    warning("Response has no spread; prior left unchanged.", call. = FALSE)
    return(prior)
  }
  if (prior$family == "normal") {
    prior$params[["sd"]] <- prior$params[["sd"]] * s
  } else if (prior$family == "student_t") {
    prior$params[["scale"]] <- prior$params[["scale"]] * s
  } else if (prior$family == "exponential") {
    prior$params[["rate"]] <- prior$params[["rate"]] / s
  }
  return(prior)
}

#' Tabulate a set of priors
#'
#' Lays out a list of priors as a tibble for reporting, one row per prior.
#'
#' @param priors A list of `pcPrior` objects.
#' @return A tibble with columns `class`, `coef`, `family` and `prior`.
#' @importFrom rlang .data
#' @export
priorTable <- function(priors) {
  if (inherits(priors, "pcPrior")) {
    priors <- list(priors)
  }
  table <- priors %>%
    purrr::map_dfr(function(prior) {
      tibble::tibble(
        class = prior$class,
        coef = prior$coef,
        family = prior$family,
        prior = priorToStan(prior)
      )
    }) %>%
    dplyr::arrange(.data$class, .data$coef)
  return(table)
}
