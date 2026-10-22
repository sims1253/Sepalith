#' Build a check result
#'
#' Wraps the outcome of one SQL data quality check in a one-row data frame
#' so that results from different checks can be bound together.
#'
#' @param check Short name of the check.
#' @param table Name of the table checked.
#' @param column Name of the column checked, or `NA`.
#' @param failures Number of offending rows.
#' @keywords internal
check.result = function(check, table, column, failures) {
  failures = as.numeric(failures)
  data.frame(
    check = check,
    table = table,
    column = column,
    failures = failures,
    passed = failures == 0,
    stringsAsFactors = FALSE
  )
}

#' SQL data quality checks
#'
#' Each check counts offending rows entirely on the server and returns a
#' one-row check result. `check.not.null()` counts rows where `column` is
#' `NULL`; `check.unique()` counts values of `column` that occur more than
#' once; `check.accepted.values()` counts rows whose `column` is not one of
#' `allowed`; `check.in.range()` counts rows whose `column` lies outside
#' `[lower, upper]`. Missing values are only counted by `check.not.null()`.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table.
#' @param column Name of the column.
#' @export
#' @return A one-row check result.
check.not.null = function(con, table, column) {
  sql = glue::glue_sql(
    "SELECT COUNT(*) AS n FROM {`table`} WHERE {`column`} IS NULL",
    .con = con
  )
  n = DBI::dbGetQuery(con, sql)$n
  check.result("not_null", table, column, n)
}

#' @rdname check.not.null
#' @export
check.unique = function(con, table, column) {
  # count duplicated values, not duplicated rows
  sql = glue::glue_sql(
    "SELECT COUNT(*) AS n FROM (
       SELECT {`column`} FROM {`table`}
       WHERE {`column`} IS NOT NULL
       GROUP BY {`column`} HAVING COUNT(*) > 1
     ) AS d",
    .con = con
  )
  n = DBI::dbGetQuery(con, sql)$n
  check.result("unique", table, column, n)
}

#' @rdname check.not.null
#' @param allowed A vector of allowed values.
#' @export
check.accepted.values = function(con, table, column, allowed) {
  if (length(allowed) == 0) {
    stop("`allowed` must contain at least one value.", call. = FALSE)
  }

  sql = glue::glue_sql(
    "SELECT COUNT(*) AS n FROM {`table`}
     WHERE {`column`} IS NOT NULL AND {`column`} NOT IN ({allowed*})",
    .con = con
  )
  n = DBI::dbGetQuery(con, sql)$n
  check.result("accepted_values", table, column, n)
}

#' @rdname check.not.null
#' @param lower,upper Inclusive bounds of the range.
#' @export
check.in.range = function(con, table, column, lower, upper) {
  if (lower > upper) {
    stop("`lower` must not exceed `upper`.", call. = FALSE)
  }

  sql = glue::glue_sql(
    "SELECT COUNT(*) AS n FROM {`table`}
     WHERE {`column`} < {lower} OR {`column`} > {upper}",
    .con = con
  )
  n = DBI::dbGetQuery(con, sql)$n
  check.result("in_range", table, column, n)
}

#' Check a foreign key relationship
#'
#' Counts rows of `table` whose `column` has no match in
#' `ref_table.ref_column`, entirely on the server.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table.
#' @param column Name of the column.
#' @param ref_table Name of the referenced table.
#' @param ref_column Name of the referenced column.
#' @export
#' @return A one-row check result.
check.relationship = function(con, table, column, ref_table, ref_column = column) {
  # orphans are child rows whose left join found no parent
  sql = glue::glue_sql(
    "SELECT COUNT(*) AS n FROM {`table`} AS c
     LEFT JOIN {`ref_table`} AS p ON c.{`column`} = p.{`ref_column`}
     WHERE c.{`column`} IS NOT NULL AND p.{`ref_column`} IS NULL",
    .con = con
  )
  n = DBI::dbGetQuery(con, sql)$n
  check.result("relationship", table, column, n)
}

#' Run a list of checks
#'
#' Runs each check described in `spec` and binds the results. Each element
#' of `spec` is a list with a `type` and the arguments that check needs.
#' A check that errors is recorded as failed with `NA` failures.
#'
#' @param con A `DBIConnection` object.
#' @param spec A list of check specifications.
#' @param stop.on.fail Whether to raise an error if any check fails.
#' @export
#' @return A data frame of check results.
run.checks = function(con, spec, stop.on.fail = FALSE) {
  runners = list(
    not_null = check.not.null,
    unique = check.unique,
    accepted_values = check.accepted.values,
    in_range = check.in.range,
    relationship = check.relationship
  )

  # an empty spec still yields a well-formed, zero-row result
  if (length(spec) == 0) {
    return(check.result(character(), character(), character(), numeric()))
  }

  results = vector("list", length(spec))
  for (i in seq_along(spec)) {
    s = spec[[i]]
    runner = runners[[s$type]]
    if (is.null(runner)) {
      stop("Unknown check type `", s$type, "`.", call. = FALSE)
    }

    args = c(list(con = con), s[setdiff(names(s), "type")])
    results[[i]] = tryCatch(
      do.call(runner, args),
      error = function(e) {
        warning("Check ", i, " errored: ", conditionMessage(e), call. = FALSE)
        check.result(s$type, s$table %||% NA_character_, s$column %||% NA_character_, NA)
      }
    )
  }

  # an errored check has passed = NA; count it as a failure
  out = do.call(rbind, results)
  out$passed[is.na(out$passed)] = FALSE
  if (stop.on.fail && !all(out$passed)) {
    stop(sum(!out$passed), " data quality check(s) failed.", call. = FALSE)
  }
  out
}

#' Default value for `NULL`
#'
#' @param x,y Values; `y` is returned when `x` is `NULL`.
#' @noRd
`%||%` = function(x, y) {
  if (is.null(x)) y else x
}
