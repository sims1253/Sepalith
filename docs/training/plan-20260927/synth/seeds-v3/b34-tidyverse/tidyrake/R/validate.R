#' Build a validation problem record
#'
#' @param column Name of the column that failed.
#' @param rule Name of the rule that failed.
#' @param rows Integer row indices that failed the rule.
#' @return A data frame with columns `column`, `rule`, `n` and `rows`, or
#'   `NULL` when `rows` is empty.
#' @keywords internal
problem <- function(column, rule, rows) {
  if (length(rows) == 0L) {
    return(NULL)
  }
  data.frame(
    column = column,
    rule = rule,
    n = length(rows),
    rows = paste(utils::head(rows, 5L), collapse = ","),
    stringsAsFactors = FALSE
  )
}

#' Check an identifier column
#'
#' @param x The column values.
#' @param name The column name used in the problem records.
#' @param allow_missing Skip the `missing` rule?
#' @return A data frame of problems, possibly with zero rows.
#' @keywords internal
check_ids <- function(x, name, allow_missing = FALSE) {
  x <- as.character(x)

  # Rows that break each rule
  missing <- which(is.na(x) | trimws(x) == "")
  duplicate <- which(duplicated(x) & !is.na(x))
  spaced <- which(!is.na(x) & grepl("\\s", x))
  bad_chars <- which(!is.na(x) & grepl("[^A-Za-z0-9_ -]", x))
  if (allow_missing) {
    missing <- integer()
  }

  out <- rbind(
    data.frame(column = character(), rule = character(), n = integer(),
               rows = character(), stringsAsFactors = FALSE),
    problem(name, "missing", missing),
    problem(name, "duplicate", duplicate),
    problem(name, "whitespace", spaced),
    problem(name, "bad_chars", bad_chars)
  )
  rownames(out) <- NULL
  out
}

#' Check an amount column
#'
#' @param x The column values.
#' @param name The column name used in the problem records.
#' @param allow_missing Skip the `missing` rule?
#' @return A data frame of problems, possibly with zero rows.
#' @keywords internal
check_amounts <- function(x, name, allow_missing = FALSE) {
  value <- suppressWarnings(as.numeric(x))

  # Rows that break each rule
  missing <- which(is.na(x))
  not_numeric <- which(!is.na(x) & is.na(value))
  negative <- which(!is.na(value) & value < 0)
  zero <- which(!is.na(value) & value == 0)
  if (allow_missing) {
    missing <- integer()
  }

  out <- rbind(
    data.frame(column = character(), rule = character(), n = integer(),
               rows = character(), stringsAsFactors = FALSE),
    problem(name, "missing", missing),
    problem(name, "not_numeric", not_numeric),
    problem(name, "negative", negative),
    problem(name, "zero", zero)
  )
  rownames(out) <- NULL
  out
}

#' Check a date column
#'
#' @param x The column values.
#' @param name The column name used in the problem records.
#' @param min_date Earliest allowed date.
#' @param max_date Latest allowed date.
#' @param allow_missing Skip the `missing` rule?
#' @return A data frame of problems, possibly with zero rows.
#' @keywords internal
check_dates <- function(x, name, min_date = "2000-01-01", max_date = Sys.Date(),
                        allow_missing = FALSE) {
  value <- as.Date(x, optional = TRUE)

  # Rows that break each rule
  missing <- which(is.na(x))
  not_date <- which(!is.na(x) & is.na(value))
  past <- which(!is.na(value) & value < as.Date(min_date))
  future <- which(!is.na(value) & value > as.Date(max_date))
  if (allow_missing) {
    missing <- integer()
  }

  out <- rbind(
    data.frame(column = character(), rule = character(), n = integer(),
               rows = character(), stringsAsFactors = FALSE),
    problem(name, "missing", missing),
    problem(name, "not_date", not_date),
    problem(name, "past", past),
    problem(name, "future", future)
  )
  rownames(out) <- NULL
  out
}

#' Check a categorical column
#'
#' @param x The column values.
#' @param name The column name used in the problem records.
#' @param levels The allowed values.
#' @param allow_missing Skip the `missing` rule?
#' @return A data frame of problems, possibly with zero rows.
#' @keywords internal
check_levels <- function(x, name, levels, allow_missing = FALSE) {
  x <- as.character(x)

  # Rows that break each rule
  missing <- which(is.na(x))
  unknown <- which(!is.na(x) & !(x %in% levels))
  case_only <- which(!is.na(x) & !(x %in% levels) & tolower(x) %in% tolower(levels))
  padded <- which(!is.na(x) & !(x %in% levels) & trimws(x) %in% levels)
  if (allow_missing) {
    missing <- integer()
  }

  out <- rbind(
    data.frame(column = character(), rule = character(), n = integer(),
               rows = character(), stringsAsFactors = FALSE),
    problem(name, "missing", missing),
    problem(name, "unknown_level", unknown),
    problem(name, "case_mismatch", case_only),
    problem(name, "padded", padded)
  )
  rownames(out) <- NULL
  out
}

#' Validate a table column by column
#'
#' Runs [check_ids()] on `ids`, [check_amounts()] on `amounts`,
#' [check_dates()] on `dates` and [check_levels()] on every element of
#' `levels`, then stacks the problems.
#'
#' @param data A data frame.
#' @param ids Identifier columns.
#' @param amounts Amount columns.
#' @param dates Date columns.
#' @param levels A named list mapping categorical columns to their allowed
#'   values.
#' @param max_date Latest allowed date for `dates`.
#' @return A data frame of problems with columns `column`, `rule`, `n` and
#'   `rows`, with class `"rake_problems"`.
#' @export
validate_table <- function(data, ids = NULL, amounts = NULL, dates = NULL,
                           levels = list(), max_date = Sys.Date()) {
  cols <- c(ids, amounts, dates, names(levels))
  unknown <- setdiff(cols, names(data))
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown columns: %s.", toString(unknown)), call. = FALSE)
  }
  found <- list()

  # Run each checker over its columns
  for (col in ids) {
    found[[length(found) + 1L]] <- check_ids(data[[col]], col)
  }
  for (col in amounts) {
    found[[length(found) + 1L]] <- check_amounts(data[[col]], col)
  }
  for (col in dates) {
    found[[length(found) + 1L]] <- check_dates(data[[col]], col, max_date = max_date)
  }
  for (col in names(levels)) {
    found[[length(found) + 1L]] <- check_levels(data[[col]], col, levels[[col]])
  }

  out <- do.call(rbind, c(list(check_ids(character(), "")), found))
  rownames(out) <- NULL
  class(out) <- c("rake_problems", "data.frame")
  out
}

#' Format validation problems as text
#'
#' @param x A `rake_problems` data frame from [validate_table()].
#' @param ... Unused.
#' @return A character vector with one line per problem.
#' @export
format.rake_problems <- function(x, ...) {
  if (nrow(x) == 0L) {
    return("No problems found.")
  }
  sprintf("%s: %s in %d row(s) [%s]", x$column, x$rule, x$n, x$rows)
}
