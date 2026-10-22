#' Check form inputs
#'
#' @description
#' Each check builds a message suitable for [shiny::validate()] or returns
#' `NULL` when the input is fine:
#'
#' * `need_positive_int()`: a whole number greater than zero;
#' * `need_price()`: a non-negative amount with at most two decimals;
#' * `need_sku()`: three letters, a dash and four digits, e.g. `"BOL-0042"`;
#' * `need_text()`: non-blank text of at most `max_chars` characters.
#'
#' @param value The input value to check.
#' @param label A human-readable field name used in the message.
#' @return `NULL` when the value is valid, otherwise a single character message.
#' @keywords internal
need_positive_int <- function(value, label) {
  if (is.null(value) || length(value) != 1 || is.na(value)) {
    return(sprintf("%s is required.", label))
  }
  # checked before the sign so that 2.5 is reported as "not whole"
  if (!is.numeric(value) || value != round(value)) {
    return(sprintf("%s must be a whole number.", label))
  }
  if (value <= 0) {
    return(sprintf("%s must be greater than zero.", label))
  }
  NULL
}

#' @rdname need_positive_int
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

  # compare in cents with a tolerance; 0.1 * 100 is not exactly 10
  if (abs(value * 100 - round(value * 100)) > 1e-8) {
    return(sprintf("%s can have at most two decimal places.", label))
  }
  NULL
}

#' @rdname need_positive_int
need_sku <- function(value, label) {
  if (is.null(value) || length(value) != 1 || is.na(value) || !nzchar(trimws(value))) {
    return(sprintf("%s is required.", label))
  }
  # validate the cleaned form, so " bol_0042" is accepted
  if (!grepl("^[A-Z]{3}-[0-9]{4}$", clean_sku(value))) {
    return(sprintf("%s must look like ABC-1234.", label))
  }
  NULL
}

#' @rdname need_positive_int
#' @param max_chars Maximum number of characters allowed after trimming.
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
#' @noRd
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
#' @noRd
clean_supplier <- function(x) {
  # toTitleCase() leaves all-caps words alone, hence the tolower() below
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
