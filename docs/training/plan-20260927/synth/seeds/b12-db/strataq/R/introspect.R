#' List tables in a schema
#'
#' Queries `information_schema.tables` for base tables in `schema`, sorted
#' by name.
#'
#' @param con A `DBIConnection` object.
#' @param schema Name of the schema.
#' @return A character vector of table names.
#' @family introspection
#' @export
listSchemaTables <- function(con, schema = "public") {
  sql <- glue::glue_sql(
    "SELECT table_name FROM information_schema.tables
     WHERE table_schema = {schema} AND table_type = 'BASE TABLE'
     ORDER BY table_name",
    .con = con
  )
  res <- DBI::dbGetQuery(con, sql)
  res$table_name
}

#' List views in a schema
#'
#' Queries `information_schema.tables` for views in `schema`, sorted by
#' name.
#'
#' @param con A `DBIConnection` object.
#' @param schema Name of the schema.
#' @return A character vector of view names.
#' @family introspection
#' @export
listSchemaViews <- function(con, schema = "public") {
  sql <- glue::glue_sql(
    "SELECT table_name FROM information_schema.tables
     WHERE table_schema = {schema} AND table_type = 'VIEW'
     ORDER BY table_name",
    .con = con
  )
  res <- DBI::dbGetQuery(con, sql)
  res$table_name
}

#' Describe the columns of a table
#'
#' Returns column names, data types and nullability from
#' `information_schema.columns`, in ordinal order.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table.
#' @param schema Name of the schema.
#' @return A data frame with columns `column`, `type`, `nullable` and
#'   `position`.
#' @family introspection
#' @export
describeColumns <- function(con, table, schema = "public") {
  sql <- glue::glue_sql(
    "SELECT column_name, data_type, is_nullable, ordinal_position
     FROM information_schema.columns
     WHERE table_schema = {schema} AND table_name = {table}
     ORDER BY ordinal_position",
    .con = con
  )
  res <- DBI::dbGetQuery(con, sql)
  if (nrow(res) == 0) {
    stop("Table `", schema, ".", table, "` not found.", call. = FALSE)
  }
  data.frame(
    column = res$column_name,
    type = res$data_type,
    nullable = res$is_nullable == "YES",
    position = as.integer(res$ordinal_position),
    stringsAsFactors = FALSE
  )
}

#' List the primary key columns of a table
#'
#' Joins the table constraints with the key column usage view to find the
#' columns of the primary key, in key order.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table.
#' @param schema Name of the schema.
#' @return A character vector of column names, empty if there is no key.
#' @export
primaryKeyColumns <- function(con, table, schema = "public") {
  sql <- glue::glue_sql(
    "SELECT k.column_name
     FROM information_schema.table_constraints AS c
     JOIN information_schema.key_column_usage AS k
       ON c.constraint_name = k.constraint_name
      AND c.table_schema = k.table_schema
     WHERE c.constraint_type = 'PRIMARY KEY'
       AND c.table_schema = {schema} AND c.table_name = {table}
     ORDER BY k.ordinal_position",
    .con = con
  )
  DBI::dbGetQuery(con, sql)$column_name
}

#' Compare a data frame with a table definition
#'
#' Checks that every column of `data` exists in the table and reports the
#' table columns that `data` does not supply.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table.
#' @param data A data frame destined for the table.
#' @param schema Name of the schema.
#' @return A list with elements `extra` (in `data` only) and `missing`
#'   (non-nullable table columns absent from `data`).
#' @export
compareToTable <- function(con, table, data, schema = "public") {
  cols <- describeColumns(con, table, schema)
  extra <- setdiff(names(data), cols$column)
  required <- cols$column[!cols$nullable]
  missing <- setdiff(required, names(data))
  list(extra = extra, missing = missing)
}

#' Estimate row counts for every table in a schema
#'
#' Reads planner statistics from `pg_class` rather than counting, so it is
#' fast on large tables but only approximate.
#'
#' @param con A `DBIConnection` object.
#' @param schema Name of the schema.
#' @return A data frame with columns `table` and `rows`, largest first.
#' @export
estimateRowCounts <- function(con, schema = "public") {
  sql <- glue::glue_sql(
    "SELECT c.relname AS table_name, c.reltuples::bigint AS approx_rows
     FROM pg_class AS c
     JOIN pg_namespace AS n ON n.oid = c.relnamespace
     WHERE n.nspname = {schema} AND c.relkind = 'r'",
    .con = con
  )
  res <- DBI::dbGetQuery(con, sql)
  out <- data.frame(table = res$table_name, rows = as.numeric(res$approx_rows))
  out <- out[order(-out$rows, out$table), , drop = FALSE]
  rownames(out) <- NULL
  out
}
