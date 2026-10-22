#' Create a check report
#'
#' Wraps the data frame returned by [run.checks()] in a `check_report`
#' object that remembers when and against which database it was run.
#'
#' @param results A data frame of check results.
#' @param database Label of the database that was checked.
#' @return An object of class `check_report`.
#' @export
new_check_report <- function(results, database) {
  needed <- c("check", "table", "column", "failures", "passed")
  missing_cols <- setdiff(needed, names(results))
  if (length(missing_cols) > 0) {
    stop("`results` lacks column(s): ", toString(missing_cols), call. = FALSE)
  }
  structure(
    list(
      results = results,
      database = database,
      run_at = Sys.time()
    ),
    class = "check_report"
  )
}

#' Format a check report
#'
#' Produces one line per check, marking failures with the number of
#' offending rows.
#'
#' @param x A `check_report` object.
#' @param ... Ignored.
#' @return A character vector.
#' @export
format.check_report <- function(x, ...) {
  res <- x$results
  target <- ifelse(is.na(res$column), res$table, paste0(res$table, ".", res$column))
  mark <- ifelse(res$passed, "PASS", "FAIL")
  detail <- ifelse(
    res$passed,
    "",
    ifelse(is.na(res$failures), " (error)", paste0(" (", res$failures, " rows)"))
  )
  sprintf("%s  %-16s %s%s", mark, res$check, target, detail)
}

#' Print a check report
#'
#' Prints a header with the database label and run time, followed by the
#' formatted checks.
#'
#' @param x A `check_report` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.check_report <- function(x, ...) {
  header <- sprintf(
    "<check_report: %s, %s>",
    x$database,
    format(x$run_at, "%Y-%m-%d %H:%M")
  )
  cat(header, "\n", sep = "")
  cat(format(x, ...), sep = "\n")
  invisible(x)
}

#' Summarise a check report
#'
#' Counts passed and failed checks per table.
#'
#' @param object A `check_report` object.
#' @param ... Ignored.
#' @return A data frame with columns `table`, `passed`, `failed`.
#' @importFrom magrittr %>%
#' @export
summary.check_report <- function(object, ...) {
  object$results %>%
    dplyr::group_by(table) %>%
    dplyr::summarise(
      passed = sum(passed),
      failed = sum(!passed),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(failed), table) %>%
    as.data.frame()
}

#' Store a check report in the database
#'
#' Appends the results of a report to the `check_history` table, creating
#' the table on first use.
#'
#' @param con A `DBIConnection` object.
#' @param report A `check_report` object.
#' @param table Name of the history table.
#' @return The number of rows written, invisibly.
#' @export
write_check_history <- function(con, report, table = "check_history") {
  stopifnot(inherits(report, "check_report"))
  rows <- report$results %>%
    dplyr::mutate(
      database = report$database,
      run_at = format(report$run_at, "%Y-%m-%dT%H:%M:%S")
    )
  if (!DBI::dbExistsTable(con, table)) {
    DBI::dbCreateTable(con, table, rows)
  }
  n <- DBI::dbAppendTable(con, table, rows)
  invisible(n)
}

#' Read recent check history
#'
#' Fetches stored check results from the last `days` days, newest first,
#' using a lazy table so that filtering happens in the database.
#'
#' @param con A `DBIConnection` object.
#' @param days Number of days to look back.
#' @param table Name of the history table.
#' @return A tibble of stored check results.
#' @importFrom magrittr %>%
#' @export
read_check_history <- function(con, days = 30, table = "check_history") {
  cutoff <- format(Sys.time() - days * 86400, "%Y-%m-%dT%H:%M:%S")
  dplyr::tbl(con, table) %>%
    dplyr::filter(run_at >= !!cutoff) %>%
    dplyr::arrange(dplyr::desc(run_at)) %>%
    dplyr::collect()
}

#' Find flaky checks
#'
#' Flags checks that have both passed and failed within the stored history,
#' along with their failure rate.
#'
#' @param con A `DBIConnection` object.
#' @param days Number of days to look back.
#' @param table Name of the history table.
#' @return A tibble with columns `check`, `table`, `column`, `runs` and
#'   `fail_rate`, most flaky first.
#' @importFrom magrittr %>%
#' @export
flaky_checks <- function(con, days = 30, table = "check_history") {
  read_check_history(con, days, table) %>%
    dplyr::group_by(check, table, column) %>%
    dplyr::summarise(
      runs = dplyr::n(),
      fail_rate = mean(!as.logical(passed)),
      .groups = "drop"
    ) %>%
    dplyr::filter(fail_rate > 0, fail_rate < 1) %>%
    dplyr::arrange(dplyr::desc(pmin(fail_rate, 1 - fail_rate)), dplyr::desc(runs))
}
