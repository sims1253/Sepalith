#' Quote a vector of column names
#'
#' Quotes each name with [DBI::dbQuoteIdentifier()] and joins them into a
#' comma-separated list for use in a `SELECT` or `INSERT` clause.
#'
#' @param con A `DBIConnection` object.
#' @param cols A character vector of column names.
#' @return A single `SQL` string.
#' @keywords internal
quote_cols <- function(con, cols) {
  if (length(cols) == 0) {
    stop("`cols` must contain at least one column name.", call. = FALSE)
  }
  quoted <- DBI::dbQuoteIdentifier(con, cols)
  return(DBI::SQL(paste(quoted, collapse = ", ")))
}

#' Quote a possibly schema-qualified table name
#'
#' Splits `"schema.table"` on the first dot and builds a [DBI::Id()] so both
#' parts are quoted separately.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @return An `SQL` string with the quoted table name.
#' @keywords internal
quote_table <- function(con, table) {
  parts <- strsplit(table, ".", fixed = TRUE)[[1]]
  id <- if (length(parts) == 2) {
    DBI::Id(schema = parts[1], table = parts[2])
  } else {
    DBI::Id(table = table)
  }
  return(DBI::dbQuoteIdentifier(con, id))
}

#' Build a SELECT statement
#'
#' Builds a `SELECT` with optional equality filters. Filter values are
#' interpolated safely with [glue::glue_sql()].
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param cols A character vector of column names, or `NULL` for all columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @family sql builders
#' @export
build_select <- function(con, table, cols = NULL, where = list()) {
  tbl <- quote_table(con, table)
  col_sql <- if (is.null(cols)) DBI::SQL("*") else quote_cols(con, cols)
  sql <- glue::glue_sql("SELECT {col_sql} FROM {tbl}", .con = con)
  if (length(where) > 0) {
    sql <- glue::glue_sql("{sql} WHERE {where_clause(con, where)}", .con = con)
  }
  return(sql)
}

#' Build a DELETE statement
#'
#' Builds a `DELETE` restricted by equality filters. An empty `where` is
#' refused so that a typo never wipes a whole table.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @family sql builders
#' @export
build_delete <- function(con, table, where) {
  if (length(where) == 0) {
    stop("Refusing to build a DELETE without a WHERE clause.", call. = FALSE)
  }
  tbl <- quote_table(con, table)
  sql <- glue::glue_sql("DELETE FROM {tbl} WHERE {where_clause(con, where)}", .con = con)
  return(sql)
}

#' Build an UPDATE statement
#'
#' Builds an `UPDATE` setting the columns in `values`, restricted by
#' equality filters.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param values A named list of new values; names are columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @family sql builders
#' @export
build_update <- function(con, table, values, where) {
  if (length(where) == 0) {
    stop("Refusing to build an UPDATE without a WHERE clause.", call. = FALSE)
  }
  tbl <- quote_table(con, table)
  sets <- vapply(names(values), function(col) {
    glue::glue_sql("{`col`} = {values[[col]]}", .con = con)
  }, character(1))
  set_sql <- DBI::SQL(paste(sets, collapse = ", "))
  sql <- glue::glue_sql(
    "UPDATE {tbl} SET {set_sql} WHERE {where_clause(con, where)}",
    .con = con
  )
  return(sql)
}

#' Build a WHERE clause from equality filters
#'
#' Each element becomes `col = value`, or `col IN (...)` when the value has
#' more than one element, or `col IS NULL` when it is `NA`.
#'
#' @param con A `DBIConnection` object.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string without the `WHERE` keyword.
#' @keywords internal
where_clause <- function(con, where) {
  if (is.null(names(where)) || any(!nzchar(names(where)))) {
    stop("Every element of `where` must be named.", call. = FALSE)
  }
  parts <- character(length(where))
  for (i in seq_along(where)) {
    col <- names(where)[i]
    val <- where[[i]]
    parts[i] <- if (length(val) == 1 && is.na(val)) {
      glue::glue_sql("{`col`} IS NULL", .con = con)
    } else if (length(val) > 1) {
      glue::glue_sql("{`col`} IN ({val*})", .con = con)
    } else {
      glue::glue_sql("{`col`} = {val}", .con = con)
    }
  }
  return(DBI::SQL(paste(parts, collapse = " AND ")))
}

#' Build a date-range predicate
#'
#' Returns a half-open interval predicate `col >= from AND col < to`. Either
#' bound may be `NULL` to leave that side open.
#'
#' @param con A `DBIConnection` object.
#' @param col Name of the date or timestamp column.
#' @param from Inclusive lower bound, or `NULL`.
#' @param to Exclusive upper bound, or `NULL`.
#' @return An `SQL` string, or `SQL("1 = 1")` when both bounds are `NULL`.
#' @export
date_range_sql <- function(con, col, from = NULL, to = NULL) {
  parts <- character()
  if (!is.null(from)) {
    from <- as.character(as.Date(from))
    parts <- c(parts, glue::glue_sql("{`col`} >= {from}", .con = con))
  }
  if (!is.null(to)) {
    to <- as.character(as.Date(to))
    parts <- c(parts, glue::glue_sql("{`col`} < {to}", .con = con))
  }
  if (length(parts) == 0) {
    return(DBI::SQL("1 = 1"))
  }
  return(DBI::SQL(paste(parts, collapse = " AND ")))
}
