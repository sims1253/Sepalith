#' Check that a value is a positive whole number
#'
#' Builds a message suitable for [shiny::validate()] when `value` is missing,
#' non-numeric, or not a whole number greater than zero.
#'
#' @param value The input value to check.
#' @param label A human-readable field name used in the message.
#' @return `NULL` when the value is valid, otherwise a single character message.
#' @family input checks
#' @keywords internal
need_positive_int <- function(value, label) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(sprintf("%s is required.", label))
  }
  if (!is.numeric(value) || value != round(value)) {
    return(sprintf("%s must be a whole number.", label))
  }
  if (value <= 0) {
    return(sprintf("%s must be greater than zero.", label))
  }
  NULL
}

#' Check that a value is a non-negative price
#'
#' Builds a message suitable for [shiny::validate()] when `value` is missing,
#' non-numeric, negative, or has more than two decimal places.
#'
#' @param value The input value to check.
#' @param label A human-readable field name used in the message.
#' @return `NULL` when the value is valid, otherwise a single character message.
#' @family input checks
#' @keywords internal
need_price <- function(value, label) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(sprintf("%s is required.", label))
  }
  if (!is.numeric(value)) {
    return(sprintf("%s must be a number.", label))
  }
  if (value < 0) {
    return(sprintf("%s cannot be negative.", label))
  }
  if (abs(value * 100 - round(value * 100)) > 1e-8) {
    return(sprintf("%s can have at most two decimal places.", label))
  }
  NULL
}

#' Check that a value is a well-formed SKU
#'
#' SKUs are three upper-case letters, a dash and four digits, for example
#' `"BOL-0042"`. Surrounding whitespace and lower case are tolerated.
#'
#' @param value The input value to check.
#' @param label A human-readable field name used in the message.
#' @return `NULL` when the value is valid, otherwise a single character message.
#' @family input checks
#' @keywords internal
need_sku <- function(value, label) {
  if (is.null(value) || length(value) != 1 || is.na(value) || !nzchar(trimws(value))) {
    return(sprintf("%s is required.", label))
  }
  if (!grepl("^[A-Z]{3}-[0-9]{4}$", clean_sku(value))) {
    return(sprintf("%s must look like ABC-1234.", label))
  }
  NULL
}

#' Check that a value is non-empty text
#'
#' Builds a message suitable for [shiny::validate()] when `value` is missing,
#' blank, or longer than `max_chars` characters.
#'
#' @param value The input value to check.
#' @param label A human-readable field name used in the message.
#' @param max_chars Maximum number of characters allowed after trimming.
#' @return `NULL` when the value is valid, otherwise a single character message.
#' @family input checks
#' @keywords internal
need_text <- function(value, label, max_chars = 60) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(sprintf("%s is required.", label))
  }
  value <- trimws(value)
  if (!nzchar(value)) {
    return(sprintf("%s cannot be blank.", label))
  }
  if (nchar(value) > max_chars) {
    return(sprintf("%s must be at most %d characters.", label, max_chars))
  }
  NULL
}

#' Normalise a SKU entered by a user
#'
#' Trims whitespace, upper-cases letters and replaces underscores or spaces
#' with a single dash.
#'
#' @param x A character vector of raw SKU strings.
#' @return A character vector the same length as `x`.
#' @keywords internal
clean_sku <- function(x) {
  x <- toupper(trimws(x))
  gsub("[_ ]+", "-", x)
}

#' Normalise a supplier name entered by a user
#'
#' Trims whitespace, collapses internal runs of spaces and converts the
#' result to title case so that duplicates group together.
#'
#' @param x A character vector of raw supplier names.
#' @return A character vector the same length as `x`.
#' @keywords internal
clean_supplier <- function(x) {
  x <- trimws(gsub("\\s+", " ", x))
  tools::toTitleCase(tolower(x))
}

#' Collect the messages returned by several checks
#'
#' Drops `NULL` results so that only failing checks remain, which makes it
#' easy to show the first problem or list all of them.
#'
#' @param ... Results of `need_*()` calls.
#' @return A character vector of messages, possibly of length zero.
#' @keywords internal
collect_errors <- function(...) {
  results <- list(...)
  failed <- Filter(Negate(is.null), results)
  as.character(unlist(failed, use.names = FALSE))
}
