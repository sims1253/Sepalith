#' Format an argument name for messages
#'
#' @param name Argument name as a string.
#' @return The name wrapped in backticks.
#' @keywords internal
#' @noRd
fmt_arg <- function(name) {
  paste0("`", name, "`")
}

#' Check a single number
#'
#' @description
#' Fails unless `x` is a finite numeric scalar inside `[lower, upper]`.
#'
#' @param x Value to check.
#' @param name Argument name used in error messages.
#' @param lower,upper Inclusive bounds.
#' @param allow_na Whether a single `NA` is accepted.
#' @return `x`, invisibly.
#' @export
check_number <- function(x, name = "x", lower = -Inf, upper = Inf, allow_na = FALSE) {
  if (!is.numeric(x) && !identical(x, NA) || length(x) != 1L) {
    stop(sprintf("%s must be a single number.", fmt_arg(name)), call. = FALSE)
  }
  if (allow_na && is.na(x)) {
    return(invisible(x))
  }
  if (is.na(x) || !is.finite(x)) {
    stop(sprintf("%s must be finite, not %s.", fmt_arg(name), format(x)), call. = FALSE)
  }
  if (x < lower) {
    stop(sprintf("%s must be at least %s.", fmt_arg(name), format(lower)), call. = FALSE)
  }
  if (x > upper) {
    stop(sprintf("%s must be at most %s.", fmt_arg(name), format(upper)), call. = FALSE)
  }
  return(invisible(x))
}

#' Check a single count
#'
#' @description
#' Fails unless `x` is a whole, non-negative number no larger than `upper`.
#'
#' @param x Value to check.
#' @param name Argument name used in error messages.
#' @param upper Inclusive upper bound.
#' @param allow_na Whether a single `NA` is accepted.
#' @return `x` coerced to integer, invisibly.
#' @export
check_count <- function(x, name = "x", upper = .Machine$integer.max, allow_na = FALSE) {
  if (!is.numeric(x) && !identical(x, NA) || length(x) != 1L) {
    stop(sprintf("%s must be a single number.", fmt_arg(name)), call. = FALSE)
  }
  if (allow_na && is.na(x)) {
    return(invisible(NA_integer_))
  }
  if (is.na(x) || x != round(x)) {
    stop(sprintf("%s must be a whole number, not %s.", fmt_arg(name), format(x)), call. = FALSE)
  }
  if (x < 0) {
    stop(sprintf("%s must be at least %s.", fmt_arg(name), format(0)), call. = FALSE)
  }
  if (x > upper) {
    stop(sprintf("%s must be at most %s.", fmt_arg(name), format(upper)), call. = FALSE)
  }
  return(invisible(as.integer(x)))
}

#' Check a single string
#'
#' @description
#' Fails unless `x` is a non-missing character scalar.
#'
#' @param x Value to check.
#' @param name Argument name used in error messages.
#' @param allow_empty Whether `""` is accepted.
#' @return `x`, invisibly.
#' @export
check_string <- function(x, name = "x", allow_empty = FALSE) {
  if (!is.character(x) || length(x) != 1L) {
    stop(sprintf("%s must be a single string.", fmt_arg(name)), call. = FALSE)
  }
  if (is.na(x)) {
    stop(sprintf("%s must not be missing.", fmt_arg(name)), call. = FALSE)
  }
  if (!allow_empty && !nzchar(x)) {
    stop(sprintf("%s must not be empty.", fmt_arg(name)), call. = FALSE)
  }
  return(invisible(x))
}

#' Check a single flag
#'
#' @description
#' Fails unless `x` is `TRUE` or `FALSE`.
#'
#' @param x Value to check.
#' @param name Argument name used in error messages.
#' @return `x`, invisibly.
#' @export
check_flag <- function(x, name = "x") {
  if (!is.logical(x) || length(x) != 1L) {
    stop(sprintf("%s must be a single flag.", fmt_arg(name)), call. = FALSE)
  }
  if (is.na(x)) {
    stop(sprintf("%s must not be missing.", fmt_arg(name)), call. = FALSE)
  }
  return(invisible(x))
}

#' Check a scalar of a given kind
#'
#' Dispatches on `type` and reports which kind was expected on failure.
#' `"positive"` and `"probability"` are numbers with fixed bounds,
#' `"choice"` is a string from `choices`.
#'
#' @param x Value to check.
#' @param type One of `"number"`, `"positive"`, `"probability"`, `"count"`,
#'   `"string"`, `"flag"` or `"choice"`.
#' @param name Argument name used in error messages.
#' @param choices Allowed values when `type = "choice"`.
#' @return `x`, invisibly.
#' @export
check_scalar <- function(x,
                         type = c("number", "positive", "probability", "count", "string",
                                  "flag", "choice"),
                         name = "x", choices = NULL) {
  type <- match.arg(type)
  switch(type,
    number = {
      check_number(x, name = name)
    },
    positive = {
      check_number(x, name = name)
      if (x <= 0) {
        stop(sprintf("%s must be positive.", fmt_arg(name)), call. = FALSE)
      }
    },
    probability = {
      check_number(x, name = name, lower = 0, upper = 1)
    },
    count = {
      check_count(x, name = name)
    },
    string = {
      check_string(x, name = name)
    },
    flag = {
      check_flag(x, name = name)
    },
    choice = {
      check_string(x, name = name)
      if (!x %in% choices) {
        stop(sprintf("%s must be one of %s.", fmt_arg(name),
                     paste(fmt_arg(choices), collapse = ", ")), call. = FALSE)
      }
    }
  )
  return(invisible(x))
}

#' Check the arguments of a sampler
#'
#' Validates every tuning argument of the sampler front end before any work
#' is done, so that a bad call fails fast with a message naming the argument.
#'
#' @param n_iter Number of iterations, a positive count.
#' @param burn_in Number of warm-up iterations, a count below `n_iter`.
#' @param thin Thinning interval, a positive count.
#' @param step_size Initial step size, a number in `(0, 1]`.
#' @param seed Random seed, a count.
#' @param verbose Whether to print progress, a flag.
#' @param method Sampler name, one of `"rwm"`, `"hmc"` or `"slice"`.
#' @return A list with elements `n_iter`, `burn_in`, `thin`, `step_size`,
#'   `seed`, `verbose` and `method`.
#' @keywords internal
check_sampler_args <- function(n_iter, burn_in, thin, step_size, seed, verbose, method) {
  # Validate `n_iter`
  n_iter <- check_count(n_iter, name = "n_iter")
  if (n_iter == 0L) {
    stop(sprintf("%s must be at least %s.", fmt_arg("n_iter"), format(1)), call. = FALSE)
  }

  # Validate `burn_in`
  burn_in <- check_count(burn_in, name = "burn_in")
  if (burn_in >= n_iter) {
    stop(sprintf("%s must be smaller than %s.", fmt_arg("burn_in"), fmt_arg("n_iter")),
         call. = FALSE)
  }

  # Validate `thin`
  thin <- check_count(thin, name = "thin")
  if (thin == 0L) {
    stop(sprintf("%s must be at least %s.", fmt_arg("thin"), format(1)), call. = FALSE)
  }

  # Validate `step_size`
  check_number(step_size, name = "step_size", upper = 1)
  if (step_size <= 0) {
    stop(sprintf("%s must be positive.", fmt_arg("step_size")), call. = FALSE)
  }

  # Validate `seed`
  seed <- check_count(seed, name = "seed")

  # Validate `verbose`
  check_flag(verbose, name = "verbose")

  # Validate `method`
  check_scalar(method, type = "choice", name = "method", choices = c("rwm", "hmc", "slice"))

  return(list(
    n_iter = n_iter,
    burn_in = burn_in,
    thin = thin,
    step_size = step_size,
    seed = seed,
    verbose = verbose,
    method = method
  ))
}

#' Check the arguments of an optimiser
#'
#' Validates the control settings of the optimiser front end. Tolerances
#' must be positive, iteration limits must be counts, and `max_time` may be
#' `NA` for no limit.
#'
#' @param max_iter Largest number of iterations, a positive count.
#' @param abs_tol,rel_tol Convergence tolerances, positive numbers.
#' @param learning_rate Initial learning rate, a probability.
#' @param patience Iterations without improvement before stopping, a count
#'   below `max_iter`.
#' @param max_time Time limit in seconds, a positive number or `NA`.
#' @param trace Whether to record the objective, a flag.
#' @return A list with elements `max_iter`, `abs_tol`, `rel_tol`,
#'   `learning_rate`, `patience`, `max_time` and `trace`.
#' @keywords internal
check_optimizer_args <- function(max_iter, abs_tol, rel_tol, learning_rate, patience,
                                 max_time, trace) {
  # Validate `max_iter`
  max_iter <- check_count(max_iter, name = "max_iter")
  if (max_iter == 0L) {
    stop(sprintf("%s must be at least %s.", fmt_arg("max_iter"), format(1)), call. = FALSE)
  }

  # Validate `abs_tol`
  check_scalar(abs_tol, type = "positive", name = "abs_tol")

  # Validate `rel_tol`
  check_scalar(rel_tol, type = "positive", name = "rel_tol")

  # Validate `learning_rate`
  check_scalar(learning_rate, type = "probability", name = "learning_rate")

  # Validate `patience`
  patience <- check_count(patience, name = "patience")
  if (patience >= max_iter) {
    stop(sprintf("%s must be smaller than %s.", fmt_arg("patience"), fmt_arg("max_iter")),
         call. = FALSE)
  }

  # Validate `max_time`
  check_number(max_time, name = "max_time", lower = 0, allow_na = TRUE)

  # Validate `trace`
  check_flag(trace, name = "trace")

  return(list(
    max_iter = max_iter,
    abs_tol = abs_tol,
    rel_tol = rel_tol,
    learning_rate = learning_rate,
    patience = patience,
    max_time = max_time,
    trace = trace
  ))
}
