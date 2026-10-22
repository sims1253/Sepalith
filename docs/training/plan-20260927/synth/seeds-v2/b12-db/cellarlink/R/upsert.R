#' @title Check that key columns are present and unique
#' @description Validates the incoming rows before they are written: every
#'   key column must exist, contain no missing values, and identify rows
#'   uniquely.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return `data`, invisibly, if all checks pass.
#' @keywords internal
check_keys <- function(data, keys) {
  if (length(keys) == 0) {
    stop("At least one key column is required.", call. = FALSE)
  }
  missing_cols <- setdiff(keys, names(data))
  if (length(missing_cols) > 0) {
    stop("Key columns not found in `data`: ", toString(missing_cols), call. = FALSE)
  }

  key_df <- data[keys]
  if (anyNA(key_df)) {
    stop("Key columns must not contain missing values.", call. = FALSE)
  }
  if (anyDuplicated(key_df) > 0) {
    stop("Key columns do not identify rows uniquely.", call. = FALSE)
  }

  return(invisible(data))
}

#' @title Write rows to a staging table
#' @description Writes `data` to a temporary table with a random suffix so
#'   that it can be merged into the target with a single statement.
#' @param con A `DBIConnection` object.
#' @param data A data frame of rows to write.
#' @param prefix Prefix for the staging table name.
#' @keywords internal
write_staging <- function(con, data, prefix = "stage") {
  # random suffix so concurrent sessions never share a staging table
  suffix <- paste(sample(c(letters, 0:9), 8, replace = TRUE), collapse = "")
  name <- paste(prefix, suffix, sep = "_")
  DBI::dbWriteTable(con, name, data, temporary = TRUE, overwrite = TRUE)
  return(name)
}

#' @title Upsert rows using ON CONFLICT
#' @description Inserts `data` into `table`, updating the non-key columns of
#'   rows whose keys already exist. Works on Postgres, SQLite and DuckDB.
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return The number of rows affected, invisibly.
#' @export
upsert_on_conflict <- function(con, table, data, keys) {
  check_keys(data, keys)
  value_cols <- setdiff(names(data), keys)
  stage <- write_staging(con, data)
  on.exit(DBI::dbRemoveTable(con, stage, temporary = TRUE), add = TRUE)

  # with only key columns there is nothing to update on a clash
  updates <- vapply(value_cols, function(col) {
    glue::glue_sql("{`col`} = excluded.{`col`}", .con = con)
  }, character(1))
  action <- if (length(updates) == 0) {
    DBI::SQL("DO NOTHING")
  } else {
    DBI::SQL(paste("DO UPDATE SET", paste(updates, collapse = ", ")))
  }

  sql <- glue::glue_sql(
    "INSERT INTO {`table`} ({`names(data)`*})
     SELECT {`names(data)`*} FROM {`stage`}
     ON CONFLICT ({`keys`*}) {action}",
    .con = con
  )
  n <- DBI::dbExecute(con, sql)
  return(invisible(n))
}

#' @title Upsert rows using delete-then-insert
#' @description Deletes rows of `table` whose keys appear in `data`, then
#'   appends `data`, inside one transaction. Useful on backends without
#'   `ON CONFLICT`.
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return The number of rows affected, invisibly.
#' @export
upsert_replace <- function(con, table, data, keys) {
  check_keys(data, keys)
  stage <- write_staging(con, data)
  on.exit(DBI::dbRemoveTable(con, stage, temporary = TRUE), add = TRUE)

  join_on <- vapply(keys, function(k) {
    glue::glue_sql("t.{`k`} = s.{`k`}", .con = con)
  }, character(1)) |>
    paste(collapse = " AND ") |>
    DBI::SQL()

  n <- DBI::dbWithTransaction(con, {
    DBI::dbExecute(con, glue::glue_sql(
      "DELETE FROM {`table`} AS t WHERE EXISTS (SELECT 1 FROM {`stage`} AS s WHERE {join_on})",
      .con = con
    ))
    DBI::dbAppendTable(con, table, data)
  })
  return(invisible(n))
}

#' @title Insert only rows whose keys are new
#' @description Appends rows of `data` whose keys do not already exist in
#'   `table`; existing rows are left untouched.
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @return The number of rows affected, invisibly.
#' @export
insert_new <- function(con, table, data, keys) {
  check_keys(data, keys)
  stage <- write_staging(con, data)
  on.exit(DBI::dbRemoveTable(con, stage, temporary = TRUE), add = TRUE)

  join_on <- vapply(keys, function(k) {
    glue::glue_sql("t.{`k`} = s.{`k`}", .con = con)
  }, character(1)) |>
    paste(collapse = " AND ") |>
    DBI::SQL()

  sql <- glue::glue_sql(
    "INSERT INTO {`table`} ({`names(data)`*})
     SELECT {`names(data)`*} FROM {`stage`} AS s
     WHERE NOT EXISTS (SELECT 1 FROM {`table`} AS t WHERE {join_on})",
    .con = con
  )
  n <- DBI::dbExecute(con, sql)
  return(invisible(n))
}

#' @title Upsert rows in batches
#' @description Splits `data` into chunks of `batch_size` rows and upserts
#'   each chunk with the chosen strategy, reporting progress as it goes.
#' @param con A `DBIConnection` object.
#' @param table Name of the target table.
#' @param data A data frame of rows to write.
#' @param keys A character vector of key column names.
#' @param batch_size Number of rows per batch.
#' @param method Upsert strategy, one of `"on_conflict"`, `"replace"` or
#'   `"insert_new"`.
#' @param verbose Whether to print a message after each batch.
#' @return A named integer vector of rows affected per batch, invisibly.
#' @export
upsert_batched <- function(con, table, data, keys, batch_size = 5000L,
                           method = c("on_conflict", "replace", "insert_new"),
                           verbose = TRUE) {
  method <- match.arg(method)
  writer <- switch(method,
    on_conflict = upsert_on_conflict,
    replace = upsert_replace,
    insert_new = insert_new
  )
  if (batch_size < 1) {
    stop("`batch_size` must be at least 1.", call. = FALSE)
  }

  n_rows <- nrow(data)
  if (n_rows == 0) {
    return(invisible(integer()))
  }

  # check keys across the whole input; per-batch checks miss cross-batch duplicates
  check_keys(data, keys)
  starts <- seq(1L, n_rows, by = batch_size)
  affected <- integer(length(starts))
  for (i in seq_along(starts)) {
    idx <- starts[i]:min(starts[i] + batch_size - 1L, n_rows)
    affected[i] <- writer(con, table, data[idx, , drop = FALSE], keys)
    if (verbose) {
      message(sprintf("[%s] batch %d/%d: %d rows", table, i, length(starts), affected[i]))
    }
  }

  names(affected) <- paste0("batch_", seq_along(starts))
  return(invisible(affected))
}
