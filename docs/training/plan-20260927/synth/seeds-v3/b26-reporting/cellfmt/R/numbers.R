#' Validate the shared arguments of the number formatters
#'
#' Stops with an informative message when one of the common formatting
#' arguments has the wrong type or length.
#'
#' @param x A numeric vector to format.
#' @param digits Number of decimal places to keep.
#' @param big_mark Character used to separate thousands.
#' @param na_string String used in place of missing values.
#' @return `TRUE`, invisibly, when every argument is valid.
#' @keywords internal
check_number_args <- function(x, digits, big_mark, na_string) {
  # Validate `x`
  if (!is.numeric(x)) {
    stop("`x` must be a numeric vector.", call. = FALSE)
  }
  # Validate `digits`
  if (!is.numeric(digits) || length(digits) != 1L || is.na(digits)) {
    stop("`digits` must be a single number.", call. = FALSE)
  }

  if (digits < 0) {
    stop("`digits` must be non-negative.", call. = FALSE)
  }
  # Validate `big_mark`
  if (!is.character(big_mark) || length(big_mark) != 1L || is.na(big_mark)) {
    stop("`big_mark` must be a single string.", call. = FALSE)
  }
  # Validate `na_string`
  if (!is.character(na_string) || length(na_string) != 1L || is.na(na_string)) {
    stop("`na_string` must be a single string.", call. = FALSE)
  }

  return(invisible(TRUE))
}

#' Format numbers with a fixed number of decimals
#'
#' Rounds `x` to `digits` decimals and inserts a thousands separator.
#'
#' @param x A numeric vector to format.
#' @param digits Number of decimal places to keep.
#' @param big_mark Character used to separate thousands.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family number formatters
#' @export
fmt_fixed <- function(x, digits = 2, big_mark = ",", na_string = "") {
  check_number_args(x, digits, big_mark, na_string)
  out <- formatC(x, format = "f", digits = digits, big.mark = big_mark)
  out[is.na(x)] <- na_string
  return(out)
}

#' Format proportions as percentages
#'
#' Multiplies `x` by `scale`, rounds to `digits` decimals and appends a
#' percent sign.
#'
#' @param x A numeric vector to format.
#' @param digits Number of decimal places to keep.
#' @param scale Multiplier applied before formatting; use 1 when `x` already
#'   holds percentages.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family number formatters
#' @export
fmt_percent <- function(x, digits = 1, scale = 100, na_string = "") {
  check_number_args(x, digits, ",", na_string)
  out <- formatC(x * scale, format = "f", digits = digits)
  out <- paste0(out, "%")
  out[is.na(x)] <- na_string
  return(out)
}

#' Format whole numbers
#'
#' Rounds `x` to the nearest integer and inserts a thousands separator.
#'
#' @param x A numeric vector to format.
#' @param big_mark Character used to separate thousands.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family number formatters
#' @export
fmt_integer <- function(x, big_mark = ",", na_string = "") {
  check_number_args(x, 0, big_mark, na_string)
  out <- formatC(round(x), format = "d", big.mark = big_mark)
  out[is.na(x)] <- na_string
  return(out)
}

#' Formatting specification for a number style
#'
#' Collects the [formatC()] settings used by [fmt_number()] for one style.
#'
#' @param style One of `"fixed"`, `"percent"`, `"basis"`, `"scientific"` or
#'   `"integer"`.
#' @param digits Number of decimal places to keep.
#' @return A list with elements `format`, `digits`, `scale`, `prefix` and
#'   `suffix`.
#' @keywords internal
number_style_spec <- function(style = c("fixed", "percent", "basis", "scientific", "integer"),
                              digits = 2) {
  style <- match.arg(style)
  spec <- switch(style,
    fixed = list(
      format = "f",
      digits = digits,
      scale = 1,
      prefix = "",
      suffix = ""
    ),
    percent = list(
      format = "f",
      digits = digits,
      scale = 100,
      prefix = "",
      suffix = "%"
    ),
    basis = list(
      format = "f",
      digits = digits,
      scale = 10000,
      prefix = "",
      suffix = " bp"
    ),
    scientific = list(
      format = "e",
      digits = digits,
      scale = 1,
      prefix = "",
      suffix = ""
    ),
    integer = list(
      format = "d",
      digits = 0,
      scale = 1,
      prefix = "",
      suffix = ""
    )
  )
  return(spec)
}

#' Format numbers in a named style
#'
#' A single entry point for the styles described by [number_style_spec()].
#'
#' @param x A numeric vector to format.
#' @param style One of `"fixed"`, `"percent"`, `"basis"`, `"scientific"` or
#'   `"integer"`.
#' @param digits Number of decimal places to keep.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family number formatters
#' @export
fmt_number <- function(x, style = c("fixed", "percent", "basis", "scientific", "integer"),
                       digits = 2, na_string = "") {
  style <- match.arg(style)
  check_number_args(x, digits, ",", na_string)
  spec <- number_style_spec(style, digits)
  value <- x * spec$scale
  if (spec$format == "d") {
    value <- round(value)
  }
  out <- formatC(value, format = spec$format, digits = spec$digits)
  out <- paste0(spec$prefix, out, spec$suffix)
  out[is.na(x)] <- na_string
  return(out)
}

#' Format numbers with an explicit sign
#'
#' Positive values get `plus`, negative values get `minus`, and exact zeros
#' are written as `zero`. Useful for change columns.
#'
#' @param x A numeric vector to format.
#' @param digits Number of decimal places to keep.
#' @param plus Prefix for positive values.
#' @param minus Prefix for negative values.
#' @param zero String used for values equal to zero.
#' @return A character vector the same length as `x`, with `NA` kept as
#'   `NA_character_`.
#' @family number formatters
#' @export
fmt_signed <- function(x, digits = 2, plus = "+", minus = "-", zero = "0") {
  check_number_args(x, digits, ",", zero)
  out <- character(length(x))
  pos <- !is.na(x) & x > 0
  neg <- !is.na(x) & x < 0
  if (any(pos)) {
    body <- formatC(abs(x[pos]), format = "f", digits = digits)
    out[pos] <- paste0(plus, body)
  }

  if (any(neg)) {
    body <- formatC(abs(x[neg]), format = "f", digits = digits)
    out[neg] <- paste0(minus, body)
  }

  out[!is.na(x) & x == 0] <- zero
  out[is.na(x)] <- NA_character_
  return(out)
}

#' Summary statistics as display strings
#'
#' Computes a handful of descriptive statistics and formats them for a
#' report table.
#'
#' @param x A numeric vector to format.
#' @param digits Number of decimal places to keep.
#' @param big_mark Character used to separate thousands.
#' @return A named character vector with elements `n`, `mean`, `sd`, `min`,
#'   `q1`, `median`, `q3`, `max` and `missing`; `n` and `missing` are written
#'   as integers.
#' @export
summarise_numbers <- function(x, digits = 2, big_mark = ",") {
  check_number_args(x, digits, big_mark, "")
  ok <- x[!is.na(x)]
  values <- c(
    n = length(ok),
    mean = mean(ok),
    sd = stats::sd(ok),
    min = min(ok),
    q1 = stats::quantile(ok, 0.25, names = FALSE),
    median = stats::median(ok),
    q3 = stats::quantile(ok, 0.75, names = FALSE),
    max = max(ok),
    missing = sum(is.na(x))
  )
  out <- fmt_fixed(values, digits = digits, big_mark = big_mark)
  names(out) <- names(values)
  counts <- c("n", "missing")
  out[counts] <- fmt_integer(values[counts], big_mark = big_mark)
  return(out)
}
