#' Abort with a formatted argument error
#'
#' @param name Argument name used in error messages.
#' @param template A `sprintf()` template whose first slot takes the name.
#' @param ... Further values for the template.
#' @return Never returns; signals an error of class `vetter_error`.
#' @keywords internal
#' @noRd
abort_arg <- function(name, template, ...) {
  msg <- sprintf(template, paste0("`", name, "`"), ...)
  cond <- structure(
    class = c("vetter_error", "error", "condition"),
    list(message = msg, call = NULL, arg = name)
  )
  stop(cond)
}

#' Check the length of a vector
#'
#' @description
#' Fails when `x` is shorter than `min_len` or longer than `max_len`.
#'
#' @param x Vector to check.
#' @param name Argument name used in error messages.
#' @param min_len,max_len Inclusive length bounds.
#' @return `x`, invisibly.
#' @export
check_length <- function(x, name = "x", min_len = 0L, max_len = Inf) {
  n <- length(x)
  if (n < min_len) {
    abort_arg(name, "%s must have at least %d elements, not %d.", as.integer(min_len), n)
  }
  if (n > max_len) {
    abort_arg(name, "%s must have at most %d elements, not %d.", as.integer(max_len), n)
  }
  invisible(x)
}

#' Check a numeric vector
#'
#' @description
#' Fails unless `x` is numeric, has no missing values when `allow_na` is
#' `FALSE`, and every element lies in `[lower, upper]`.
#'
#' @param x Vector to check.
#' @param name Argument name used in error messages.
#' @param lower,upper Inclusive bounds.
#' @param allow_na Whether missing values are accepted.
#' @return `x`, invisibly.
#' @export
check_numeric <- function(x, name = "x", lower = -Inf, upper = Inf, allow_na = FALSE) {
  if (!is.numeric(x)) {
    abort_arg(name, "%s must be numeric, not %s.", class(x)[1])
  }
  if (!allow_na && anyNA(x)) {
    abort_arg(name, "%s must not contain missing values (found %d).", sum(is.na(x)))
  }
  below <- which(x < lower)
  if (length(below) > 0L) {
    abort_arg(name, "%s must be at least %s; element %d is %s.",
              format(lower), below[1], format(x[below[1]]))
  }
  above <- which(x > upper)
  if (length(above) > 0L) {
    abort_arg(name, "%s must be at most %s; element %d is %s.",
              format(upper), above[1], format(x[above[1]]))
  }
  invisible(x)
}

#' Check a character vector
#'
#' @description
#' Fails unless `x` is character, has no missing values when `allow_na` is
#' `FALSE`, and every element has between `min_chars` and `max_chars`
#' characters.
#'
#' @param x Vector to check.
#' @param name Argument name used in error messages.
#' @param min_chars,max_chars Inclusive bounds on `nchar(x)`.
#' @param allow_na Whether missing values are accepted.
#' @return `x`, invisibly.
#' @export
check_character <- function(x, name = "x", min_chars = 0L, max_chars = Inf,
                            allow_na = FALSE) {
  if (!is.character(x)) {
    abort_arg(name, "%s must be character, not %s.", class(x)[1])
  }
  if (!allow_na && anyNA(x)) {
    abort_arg(name, "%s must not contain missing values (found %d).", sum(is.na(x)))
  }
  width <- nchar(x)
  short <- which(width < min_chars)
  if (length(short) > 0L) {
    abort_arg(name, "%s must have at least %d characters; element %d has %d.",
              as.integer(min_chars), short[1], width[short[1]])
  }
  long <- which(width > max_chars)
  if (length(long) > 0L) {
    abort_arg(name, "%s must have at most %d characters; element %d has %d.",
              as.integer(max_chars), long[1], width[long[1]])
  }
  invisible(x)
}

#' Check that a vector is sorted
#'
#' @param x Vector to check.
#' @param name Argument name used in error messages.
#' @param decreasing Whether the order must be decreasing instead of
#'   increasing.
#' @param strict Whether ties are forbidden.
#' @return `x`, invisibly.
#' @export
check_sorted <- function(x, name = "x", decreasing = FALSE, strict = FALSE) {
  if (length(x) < 2L) {
    return(invisible(x))
  }
  step <- diff(x)
  if (decreasing) {
    bad <- if (strict) which(step >= 0) else which(step > 0)
    if (length(bad) > 0L) {
      abort_arg(name, "%s must be decreasing; element %d breaks the order.", bad[1] + 1L)
    }
  } else {
    bad <- if (strict) which(step <= 0) else which(step < 0)
    if (length(bad) > 0L) {
      abort_arg(name, "%s must be increasing; element %d breaks the order.", bad[1] + 1L)
    }
  }
  invisible(x)
}

#' Check that a vector has no duplicates
#'
#' @param x Vector to check.
#' @param name Argument name used in error messages.
#' @param ignore_na Whether repeated missing values are tolerated.
#' @return `x`, invisibly.
#' @export
check_unique <- function(x, name = "x", ignore_na = TRUE) {
  dup <- duplicated(x)
  if (ignore_na) {
    dup <- dup & !is.na(x)
  }
  if (any(dup)) {
    first <- which(dup)[1]
    abort_arg(name, "%s must be unique; %s appears more than once.", format(x[first]))
  }
  invisible(x)
}

#' Check a vector of a given kind
#'
#' Applies the kind-specific checks named by `type` and then the length
#' check, so callers can validate any vector argument with one call.
#'
#' @param x Vector to check.
#' @param type One of `"numeric"`, `"integer"`, `"character"`, `"logical"`,
#'   `"factor"`, `"date"` or `"id"`.
#' @param name Argument name used in error messages.
#' @param min_len,max_len Inclusive length bounds.
#' @return `x`, invisibly.
#' @export
check_vector <- function(x,
                         type = c("numeric", "integer", "character", "logical", "factor",
                                  "date", "id"),
                         name = "x", min_len = 1L, max_len = Inf) {
  type <- match.arg(type)
  switch(type,
    numeric = {
      check_numeric(x, name = name)
    },
    integer = {
      check_numeric(x, name = name)
      if (any(x != round(x))) {
        abort_arg(name, "%s must contain whole numbers only.")
      }
    },
    character = {
      check_character(x, name = name)
    },
    logical = {
      if (!is.logical(x)) {
        abort_arg(name, "%s must be logical, not %s.", class(x)[1])
      }
    },
    factor = {
      if (!is.factor(x)) {
        abort_arg(name, "%s must be factor, not %s.", class(x)[1])
      }
    },
    date = {
      if (!inherits(x, "Date")) {
        abort_arg(name, "%s must be Date, not %s.", class(x)[1])
      }
    },
    id = {
      check_character(x, name = name, min_chars = 1L)
      check_unique(x, name = name)
    }
  )
  check_length(x, name = name, min_len = min_len, max_len = max_len)
}

#' Check paired data vectors
#'
#' @description
#' Validates the `x` and `y` vectors of a paired analysis and the optional
#' `weights`, then checks that all three have the same length.
#'
#' @param x,y Numeric vectors of paired observations.
#' @param weights Non-negative weights, or `NULL` for equal weights.
#' @param allow_na Whether missing values are accepted in `x` and `y`.
#' @return A list with elements `x`, `y`, `weights` and `n`; `weights` is
#'   filled with ones when it was `NULL`.
#' @export
check_paired <- function(x, y, weights = NULL, allow_na = FALSE) {
  # Validate `x`
  check_numeric(x, name = "x", allow_na = allow_na)
  check_length(x, name = "x", min_len = 2L)
  if (all(is.na(x))) {
    abort_arg("x", "%s must contain at least one observed value.")
  }

  # Validate `y`
  check_numeric(y, name = "y", allow_na = allow_na)
  check_length(y, name = "y", min_len = 2L)
  if (all(is.na(y))) {
    abort_arg("y", "%s must contain at least one observed value.")
  }

  # Validate `weights`
  if (is.null(weights)) {
    weights <- rep(1, length(x))
  }
  check_numeric(weights, name = "weights", lower = 0)
  if (sum(weights) == 0) {
    abort_arg("weights", "%s must not all be zero.")
  }

  # Validate the lengths
  if (length(y) != length(x)) {
    abort_arg("y", "%s must have the same length as `x` (%d), not %d.", length(x), length(y))
  }
  if (length(weights) != length(x)) {
    abort_arg("weights", "%s must have the same length as `x` (%d), not %d.",
              length(x), length(weights))
  }
  list(x = x, y = y, weights = weights, n = length(x))
}
