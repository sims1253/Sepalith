#' Signal a tallyhook argument error
#'
#' @description Wraps [cli::cli_abort()] so that every argument error carries
#'   the `tallyhook_error_arg` class plus a more specific subclass.
#' @param message A cli-formatted error message.
#' @param subclass A string naming the specific error, appended to `tallyhook_error_`.
#' @param arg The name of the argument being checked, used in error messages.
#' @param call The execution environment of the caller, used in error messages.
#' @param ... Passed to [cli::cli_abort()].
#' @noRd
abort_arg <- function(message, subclass, arg, call = rlang::caller_env(), ...) {
  classes <- c(
    paste0("tallyhook_error_", subclass),
    "tallyhook_error_arg",
    "tallyhook_error"
  )

  # Interpolate in the checker's frame, where `x`, `min` and friends live
  cli::cli_abort(
    message,
    class = classes,
    arg = arg,
    call = call,
    ...,
    .envir = parent.frame()
  )
}

#' Check that an argument is a single string
#'
#' @description Validates that `x` is a character vector of length one that
#'   is not `NA`.
#' @param x The object to check.
#' @param allow_null Whether `NULL` is accepted as a valid value.
#' @param arg The name of the argument being checked, used in error messages.
#' @param call The execution environment of the caller, used in error messages.
#' @return `x`, invisibly, if the check passes.
#' @keywords internal
check_string <- function(x,
                         allow_null = FALSE,
                         arg = rlang::caller_arg(x),
                         call = rlang::caller_env()) {
  if (allow_null && is.null(x)) {
    return(invisible(NULL))
  }
  if (!is.character(x) || length(x) != 1L || is.na(x)) {
    abort_arg(
      "{.arg {arg}} must be a single string, not {.obj_type_friendly {x}}.",
      subclass = "string",
      arg = arg,
      call = call
    )
  }
  return(invisible(x))
}

#' Check that an argument is a single logical flag
#'
#' @description Validates that `x` is `TRUE` or `FALSE`.
#' @param x The object to check.
#' @param allow_null Whether `NULL` is accepted as a valid value.
#' @param arg The name of the argument being checked, used in error messages.
#' @param call The execution environment of the caller, used in error messages.
#' @return `x`, invisibly, if the check passes.
#' @keywords internal
check_flag <- function(x,
                       allow_null = FALSE,
                       arg = rlang::caller_arg(x),
                       call = rlang::caller_env()) {
  if (allow_null && is.null(x)) {
    return(invisible(NULL))
  }
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    abort_arg(
      "{.arg {arg}} must be {.code TRUE} or {.code FALSE}, not {.obj_type_friendly {x}}.",
      subclass = "flag",
      arg = arg,
      call = call
    )
  }
  return(invisible(x))
}

#' Check that an argument is a count
#'
#' @description Validates that `x` is a single non-negative whole number.
#'   Doubles such as `3` are accepted as long as they have no fractional
#'   part.
#' @param x The object to check.
#' @param allow_zero Whether zero counts as a valid value.
#' @param arg The name of the argument being checked, used in error messages.
#' @param call The execution environment of the caller, used in error messages.
#' @return `x` coerced to integer, invisibly, if the check passes.
#' @keywords internal
check_count <- function(x,
                        allow_zero = TRUE,
                        arg = rlang::caller_arg(x),
                        call = rlang::caller_env()) {
  # Inf passes the whole-number test below but has no integer form
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    abort_arg(
      "{.arg {arg}} must be a single finite number, not {.obj_type_friendly {x}}.",
      subclass = "count",
      arg = arg,
      call = call
    )
  }
  if (x != trunc(x)) {
    abort_arg(
      "{.arg {arg}} must be a whole number, not {x}.",
      subclass = "count",
      arg = arg,
      call = call
    )
  }

  lower <- if (allow_zero) 0 else 1
  if (x < lower) {
    abort_arg(
      "{.arg {arg}} must be at least {lower}, not {x}.",
      subclass = "count",
      arg = arg,
      call = call
    )
  }
  return(invisible(as.integer(x)))
}

#' Check that a number lies within a range
#'
#' @description Validates that `x` is a single finite number between `min`
#'   and `max`, both bounds inclusive.
#' @param x The object to check.
#' @param min,max The inclusive lower and upper bounds.
#' @param arg The name of the argument being checked, used in error messages.
#' @param call The execution environment of the caller, used in error messages.
#' @return `x`, invisibly, if the check passes.
#' @keywords internal
check_number_between <- function(x,
                                 min = -Inf,
                                 max = Inf,
                                 arg = rlang::caller_arg(x),
                                 call = rlang::caller_env()) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x)) {
    abort_arg(
      "{.arg {arg}} must be a single finite number, not {.obj_type_friendly {x}}.",
      subclass = "number",
      arg = arg,
      call = call
    )
  }
  if (x < min || x > max) {
    abort_arg(
      c(
        "{.arg {arg}} must be between {min} and {max}.",
        "x" = "You supplied {x}."
      ),
      subclass = "number",
      arg = arg,
      call = call
    )
  }
  return(invisible(x))
}

#' Check that a value is one of a set of choices
#'
#' @description A stricter alternative to [match.arg()] that never partially
#'   matches and produces a classed error listing the valid choices.
#' @param x The object to check.
#' @param choices A character vector of allowed values.
#' @param arg The name of the argument being checked, used in error messages.
#' @param call The execution environment of the caller, used in error messages.
#' @return `x`, invisibly, if the check passes.
#' @keywords internal
check_choice <- function(x,
                         choices,
                         arg = rlang::caller_arg(x),
                         call = rlang::caller_env()) {
  # Pass arg and call through so errors still name the original caller
  check_string(x, arg = arg, call = call)
  if (!x %in% choices) {
    abort_arg(
      c(
        "{.arg {arg}} must be one of {.or {.val {choices}}}.",
        "x" = "You supplied {.val {x}}."
      ),
      subclass = "choice",
      arg = arg,
      call = call
    )
  }
  return(invisible(x))
}

#' Check that a data frame has the required columns
#'
#' @description Validates that `data` is a data frame and that every name in
#'   `cols` is present. Missing columns are listed in the error message.
#' @param data A data frame of ledger entries.
#' @param cols A character vector of column names that must be present.
#' @param arg The name of the argument being checked, used in error messages.
#' @param call The execution environment of the caller, used in error messages.
#' @return `data`, invisibly, if the check passes.
#' @keywords internal
check_has_columns <- function(data,
                              cols,
                              arg = rlang::caller_arg(data),
                              call = rlang::caller_env()) {
  if (!is.data.frame(data)) {
    abort_arg(
      "{.arg {arg}} must be a data frame, not {.obj_type_friendly {data}}.",
      subclass = "columns",
      arg = arg,
      call = call
    )
  }

  # Report every missing column at once rather than the first one
  missing_cols <- setdiff(cols, names(data))
  if (length(missing_cols) > 0L) {
    abort_arg(
      c(
        "{.arg {arg}} is missing {length(missing_cols)} required column{?s}.",
        "x" = "Missing: {.field {missing_cols}}."
      ),
      subclass = "columns",
      arg = arg,
      call = call,
      missing = missing_cols
    )
  }
  return(invisible(data))
}
