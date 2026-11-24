#' Raise all collected argument problems at once
#'
#' @param problems A named character vector; names are argument names and
#'   values are problem descriptions.
#' @param fn Name of the calling function, used in the header line.
#' @return `TRUE`, invisibly, when `problems` is empty; otherwise an error.
#' @keywords internal
#' @noRd
report_problems = function(problems, fn) {
  if (length(problems) == 0L) {
    return(invisible(TRUE))
  }
  lines = sprintf("* `%s` %s", names(problems), unname(problems))
  header = sprintf("Invalid arguments to `%s()`:", fn)
  stop(paste(c(header, lines), collapse = "\n"), call. = FALSE)
}

#' Is a value a single finite number
#'
#' @param x Value to test.
#' @return `TRUE` or `FALSE`.
#' @keywords internal
#' @noRd
is_number = function(x) {
  is.numeric(x) && length(x) == 1L && !is.na(x) && is.finite(x)
}

#' Check the arguments of a rolling window
#'
#' @description
#' Collects every problem with the window specification and reports them
#' together instead of stopping at the first one.
#'
#' @param start,end Window limits, single numbers with `start < end`.
#' @param step Distance between window positions, a positive number.
#' @param align Alignment of the window, one of `"left"`, `"center"` or
#'   `"right"`.
#' @param min_obs Smallest number of observations per window, a positive
#'   number.
#' @return `TRUE`, invisibly.
#' @export
check_window_args = function(start, end, step, align, min_obs = 1) {
  problems = character()

  # Check `start`
  if (!is_number(start)) {
    problems[["start"]] = "must be a single finite number."
  }

  # Check `end`
  if (!is_number(end)) {
    problems[["end"]] = "must be a single finite number."
  } else if (is_number(start) && end <= start) {
    problems[["end"]] = "must be greater than `start`."
  }

  # Check `step`
  if (!is_number(step)) {
    problems[["step"]] = "must be a single finite number."
  } else if (step <= 0) {
    problems[["step"]] = "must be positive."
  }

  # Check `align`
  if (!is.character(align) || length(align) != 1L) {
    problems[["align"]] = "must be a single string."
  } else if (!align %in% c("left", "center", "right")) {
    problems[["align"]] = "must be one of `left`, `center`, `right`."
  }

  # Check `min_obs`
  if (!is_number(min_obs)) {
    problems[["min_obs"]] = "must be a single finite number."
  } else if (min_obs < 1) {
    problems[["min_obs"]] = "must be at least 1."
  }
  report_problems(problems, "check_window_args")
}

#' Check plot dimensions
#'
#' @description
#' Validates the width and height of an output device together with its
#' resolution and unit.
#'
#' @param width,height Device size, positive numbers.
#' @param dpi Resolution in dots per inch, a positive number.
#' @param units One of `"in"`, `"cm"` or `"px"`.
#' @return A list with elements `width`, `height`, `dpi` and `units`, with
#'   `width` and `height` converted to inches.
#' @export
check_plot_size = function(width, height, dpi = 300, units = "in") {
  problems = character()
  if (!is_number(width) || width <= 0) {
    problems[["width"]] = "must be a positive number."
  }
  if (!is_number(height) || height <= 0) {
    problems[["height"]] = "must be a positive number."
  }
  if (!is_number(dpi) || dpi <= 0) {
    problems[["dpi"]] = "must be a positive number."
  }
  if (!identical(units, "in") && !identical(units, "cm") && !identical(units, "px")) {
    problems[["units"]] = "must be one of `in`, `cm`, `px`."
  }
  report_problems(problems, "check_plot_size")

  scale = switch(units, "in" = 1, cm = 1 / 2.54, px = 1 / dpi)
  list(width = width * scale, height = height * scale, dpi = dpi, units = "in")
}

#' Check a pair of bounds
#'
#' @description
#' Validates a lower and an upper bound that may each be `NULL` (unbounded)
#' and checks that they are ordered when both are given.
#'
#' @param lower,upper Bounds, single numbers or `NULL`.
#' @param name Prefix used for the argument names in messages.
#' @return A numeric vector `c(lower, upper)` with `NULL` replaced by `-Inf`
#'   and `Inf`.
#' @export
check_bounds_pair = function(lower = NULL, upper = NULL, name = "") {
  problems = character()
  lower_name = paste0(name, "lower")
  upper_name = paste0(name, "upper")
  if (is.null(lower)) {
    lower = -Inf
  } else if (!is.numeric(lower) || length(lower) != 1L || is.na(lower)) {
    problems[[lower_name]] = "must be a single number or `NULL`."
  }
  if (is.null(upper)) {
    upper = Inf
  } else if (!is.numeric(upper) || length(upper) != 1L || is.na(upper)) {
    problems[[upper_name]] = "must be a single number or `NULL`."
  }
  if (length(problems) == 0L && lower > upper) {
    problems[[upper_name]] = sprintf("must not be smaller than `%s`.", lower_name)
  }
  report_problems(problems, "check_bounds_pair")
  c(lower, upper)
}

#' Check a named option list
#'
#' @description
#' Fails when `x` is not a fully named list, lacks any of the `required`
#' names, or contains names that are neither `required` nor `optional`.
#'
#' @param x List of options.
#' @param required Names that must be present.
#' @param optional Names that may be present.
#' @return `x`, invisibly.
#' @export
check_named_list = function(x, required = character(), optional = character()) {
  if (!is.list(x)) {
    stop("`x` must be a list.", call. = FALSE)
  }
  nms = names(x)
  if (length(x) > 0L && (is.null(nms) || any(!nzchar(nms)))) {
    stop("`x` must be a fully named list.", call. = FALSE)
  }
  missing_names = setdiff(required, nms)
  if (length(missing_names) > 0L) {
    stop(sprintf("`x` is missing required element(s): %s.",
                 paste0("`", missing_names, "`", collapse = ", ")), call. = FALSE)
  }
  unknown_names = setdiff(nms, c(required, optional))
  if (length(unknown_names) > 0L) {
    stop(sprintf("`x` has unknown element(s): %s.",
                 paste0("`", unknown_names, "`", collapse = ", ")), call. = FALSE)
  }
  invisible(x)
}

#' Check the arguments of a delimited file import
#'
#' @description
#' Validates every argument of the importer and reports all problems at
#' once.
#'
#' @param path Path to an existing file.
#' @param sep Field separator, a single character.
#' @param header Whether the first line holds column names, a flag.
#' @param skip Number of lines to skip, a non-negative whole number.
#' @param n_max Maximum number of rows, a positive whole number or `Inf`.
#' @param encoding File encoding, one of `"UTF-8"`, `"latin1"` or
#'   `"unknown"`.
#' @return `TRUE`, invisibly.
#' @keywords internal
check_import_args = function(path, sep, header, skip, n_max, encoding) {
  problems = character()

  # Check `path`
  if (!is.character(path) || length(path) != 1L || is.na(path)) {
    problems[["path"]] = "must be a single string."
  } else if (!file.exists(path)) {
    problems[["path"]] = sprintf("points to a file that does not exist: %s", path)
  }

  # Check `sep`
  if (!is.character(sep) || length(sep) != 1L || is.na(sep)) {
    problems[["sep"]] = "must be a single string."
  } else if (nchar(sep) != 1L) {
    problems[["sep"]] = "must be exactly one character."
  }

  # Check `header`
  if (!is.logical(header) || length(header) != 1L || is.na(header)) {
    problems[["header"]] = "must be `TRUE` or `FALSE`."
  }

  # Check `skip`
  if (!is_number(skip) || skip < 0 || skip != round(skip)) {
    problems[["skip"]] = "must be a non-negative whole number."
  }

  # Check `n_max`
  if (!identical(n_max, Inf) && (!is_number(n_max) || n_max < 1 || n_max != round(n_max))) {
    problems[["n_max"]] = "must be a positive whole number or `Inf`."
  }

  # Check `encoding`
  if (!is.character(encoding) || length(encoding) != 1L) {
    problems[["encoding"]] = "must be a single string."
  } else if (!encoding %in% c("UTF-8", "latin1", "unknown")) {
    problems[["encoding"]] = "must be one of `UTF-8`, `latin1`, `unknown`."
  }
  report_problems(problems, "check_import_args")
}

#' Check axis settings
#'
#' @description
#' Validates the limits, labels and break counts of both axes of a plot and
#' reports all problems at once.
#'
#' @param xlim,ylim Axis limits, increasing numeric vectors of length 2, or
#'   `NULL` for automatic limits.
#' @param xlab,ylab Axis labels, single strings.
#' @param xbreaks,ybreaks Number of axis breaks, whole numbers of at least 2.
#' @return `TRUE`, invisibly.
#' @export
check_axis_args = function(xlim = NULL, ylim = NULL, xlab = "x", ylab = "y",
                           xbreaks = 5, ybreaks = 5) {
  problems = character()

  # Check the x axis
  if (!is.null(xlim)) {
    if (!is.numeric(xlim) || length(xlim) != 2L || anyNA(xlim)) {
      problems[["xlim"]] = "must be a numeric vector of length 2."
    } else if (xlim[2] <= xlim[1]) {
      problems[["xlim"]] = "must be increasing."
    }
  }
  if (!is.character(xlab) || length(xlab) != 1L) {
    problems[["xlab"]] = "must be a single string."
  }
  if (!is_number(xbreaks) || xbreaks < 2 || xbreaks != round(xbreaks)) {
    problems[["xbreaks"]] = "must be a whole number of at least 2."
  }

  # Check the y axis
  if (!is.null(ylim)) {
    if (!is.numeric(ylim) || length(ylim) != 2L || anyNA(ylim)) {
      problems[["ylim"]] = "must be a numeric vector of length 2."
    } else if (ylim[2] <= ylim[1]) {
      problems[["ylim"]] = "must be increasing."
    }
  }
  if (!is.character(ylab) || length(ylab) != 1L) {
    problems[["ylab"]] = "must be a single string."
  }
  if (!is_number(ybreaks) || ybreaks < 2 || ybreaks != round(ybreaks)) {
    problems[["ybreaks"]] = "must be a whole number of at least 2."
  }
  report_problems(problems, "check_axis_args")
}
