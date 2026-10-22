#' @title Quote a vector of column names
#' @description Quotes each name with [DBI::dbQuoteIdentifier()] and joins
#'   them into a comma-separated list for use in a `SELECT` or `INSERT`
#'   clause.
#' @param con A `DBIConnection` object.
#' @param cols A character vector of column names.
#' @keywords internal
quote_cols <- function(con, cols) {
  if (length(cols) == 0) {
    stop("`cols` must contain at least one column name.", call. = FALSE)
  }
  quoted <- DBI::dbQuoteIdentifier(con, cols)
  return(DBI::SQL(paste(quoted, collapse = ", ")))
}

#' @title Quote a possibly schema-qualified table name
#' @description Splits `"schema.table"` on the first dot and builds a
#'   [DBI::Id()] so both parts are quoted separately.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @keywords internal
quote_table <- function(con, table) {
  # anything with more than one dot is passed through as a plain name
  parts <- strsplit(table, ".", fixed = TRUE)[[1]]
  id <- if (length(parts) == 2) {
    DBI::Id(schema = parts[1], table = parts[2])
  } else {
    DBI::Id(table = table)
  }
  return(DBI::dbQuoteIdentifier(con, id))
}

#' @title Build a SELECT statement
#' @description Builds a `SELECT` with optional equality filters. Filter
#'   values are interpolated safely with [glue::glue_sql()].
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param cols A character vector of column names, or `NULL` for all columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
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

#' @title Build a DELETE statement
#' @description Builds a `DELETE` restricted by equality filters. An empty
#'   `where` is refused so that a typo never wipes a whole table.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_delete <- function(con, table, where) {
  if (length(where) == 0) {
    stop("Refusing to build a DELETE without a WHERE clause.", call. = FALSE)
  }

  tbl <- quote_table(con, table)
  sql <- glue::glue_sql("DELETE FROM {tbl} WHERE {where_clause(con, where)}", .con = con)
  return(sql)
}

#' @title Build an UPDATE statement
#' @description Builds an `UPDATE` setting the columns in `values`,
#'   restricted by equality filters.
#' @param con A `DBIConnection` object.
#' @param table Name of the table, optionally prefixed with a schema.
#' @param values A named list of new values; names are columns.
#' @param where A named list of equality filters; names are columns.
#' @return An `SQL` string.
#' @export
build_update <- function(con, table, values, where) {
  if (length(where) == 0) {
    stop("Refusing to build an UPDATE without a WHERE clause.", call. = FALSE)
  }
  if (length(values) == 0 || is.null(names(values)) || any(!nzchar(names(values)))) {
    stop("`values` must be a non-empty, fully named list.", call. = FALSE)
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

#' @title Build a WHERE clause from equality filters
#' @description Each element becomes `col = value`, or `col IN (...)` when the
#'   value has more than one element, or `col IS NULL` when it is `NA`. An
#'   empty value is an error rather than a silently dropped filter.
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
    if (length(val) == 0) {
      stop("Filter on `", col, "` has no values.", call. = FALSE)
    }
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

#' @title Build a date-range predicate
#' @description Returns a half-open interval predicate
#'   `col >= from AND col < to`. Either bound may be `NULL` to leave that
#'   side open.
#' @param con A `DBIConnection` object.
#' @param col Name of the date or timestamp column.
#' @param from Inclusive lower bound, or `NULL`.
#' @param to Exclusive upper bound, or `NULL`.
#' @return An `SQL` string, or `SQL("1 = 1")` when both bounds are `NULL`.
#' @export
date_range_sql <- function(con, col, from = NULL, to = NULL) {
  parts <- character()

  # dates go in as ISO strings so every backend compares them the same way
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
