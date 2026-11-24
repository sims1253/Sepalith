#' Quote identifiers or literals for statement builders
#'
#' A compact quoting routine shared by the statement builders in this file.
#'
#' @param x A character, numeric or logical vector.
#' @param kind Either `"identifier"` or `"literal"`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A character vector of quoted SQL fragments.
#' @keywords internal
sql_quote <- function(x, kind = c("identifier", "literal"),
                      dialect = c("postgres", "mysql", "sqlite")) {
  kind <- match.arg(kind)
  dialect <- match.arg(dialect)
  if (kind == "identifier") {
    q <- if (dialect == "mysql") "`" else "\""
    x <- gsub(q, paste0(q, q), x, fixed = TRUE)
    out <- paste0(q, x, q)
    out[x == "*"] <- "*"
    return(out)
  }
  out <- if (is.logical(x)) {
    if (dialect == "sqlite") ifelse(x, "1", "0") else ifelse(x, "TRUE", "FALSE")
  } else if (is.numeric(x)) {
    trimws(formatC(as.numeric(x), format = "fg", digits = 15))
  } else {
    paste0("'", gsub("'", "''", as.character(x), fixed = TRUE), "'")
  }
  out[is.na(x)] <- "NULL"
  out
}

#' Build a SELECT statement
#'
#' Emits one clause per non-empty argument, in the order of the signature.
#'
#' @param table The table to select from.
#' @param columns Column names, or `"*"` for all columns.
#' @param where Character vector of SQL conditions, combined with `AND`.
#' @param group_by Column names to group by.
#' @param having Character vector of SQL conditions on groups, combined with `AND`.
#' @param order_by Column names to sort by; prefix a name with `"-"` for descending.
#' @param limit Maximum number of rows, or `NULL`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL string, one clause per line.
#' @export
build_select <- function(table, columns = "*", where = NULL, group_by = NULL, having = NULL,
                         order_by = NULL, limit = NULL,
                         dialect = c("postgres", "mysql", "sqlite")) {
  dialect <- match.arg(dialect)
  clauses <- character()
  # SELECT
  cols <- sql_quote(columns, "identifier", dialect)
  clauses <- c(clauses, paste("SELECT", paste(cols, collapse = ", ")))
  # FROM
  clauses <- c(clauses, paste("FROM", sql_quote(table, "identifier", dialect)))
  # WHERE
  if (length(where) > 0) {
    clauses <- c(clauses, paste("WHERE", paste(where, collapse = " AND ")))
  }
  # GROUP BY
  if (length(group_by) > 0) {
    keys <- sql_quote(group_by, "identifier", dialect)
    clauses <- c(clauses, paste("GROUP BY", paste(keys, collapse = ", ")))
  }
  # HAVING
  if (length(having) > 0) {
    clauses <- c(clauses, paste("HAVING", paste(having, collapse = " AND ")))
  }
  # ORDER BY
  if (length(order_by) > 0) {
    desc <- startsWith(order_by, "-")
    keys <- sql_quote(sub("^-", "", order_by), "identifier", dialect)
    keys <- paste(keys, ifelse(desc, "DESC", "ASC"))
    clauses <- c(clauses, paste("ORDER BY", paste(keys, collapse = ", ")))
  }
  # LIMIT
  if (!is.null(limit)) {
    clauses <- c(clauses, paste("LIMIT", as.integer(limit)))
  }
  paste(clauses, collapse = "\n")
}

#' Build an INSERT statement
#'
#' Emits one clause per argument: target, column list, value rows, returning list.
#'
#' @param table The table to insert into.
#' @param values A data frame; one row per inserted row.
#' @param on_conflict Either `"error"`, `"ignore"` or `"replace"`.
#' @param returning Column names to return, or `NULL`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL string, one clause per line.
#' @export
build_insert <- function(table, values, on_conflict = c("error", "ignore", "replace"),
                         returning = NULL, dialect = c("postgres", "mysql", "sqlite")) {
  on_conflict <- match.arg(on_conflict)
  dialect <- match.arg(dialect)
  if (!is.data.frame(values) || nrow(values) == 0L) {
    stop("`values` must be a data frame with at least one row.", call. = FALSE)
  }
  clauses <- character()
  # INSERT INTO
  verb <- switch(on_conflict,
    error = "INSERT INTO",
    ignore = switch(dialect,
      postgres = "INSERT INTO", mysql = "INSERT IGNORE INTO", sqlite = "INSERT OR IGNORE INTO"
    ),
    replace = switch(dialect,
      postgres = "INSERT INTO", mysql = "REPLACE INTO", sqlite = "INSERT OR REPLACE INTO"
    )
  )
  clauses <- c(clauses, paste(verb, sql_quote(table, "identifier", dialect)))
  # Column list
  cols <- sql_quote(names(values), "identifier", dialect)
  clauses <- c(clauses, paste0("(", paste(cols, collapse = ", "), ")"))
  # VALUES
  quoted <- lapply(values, sql_quote, kind = "literal", dialect = dialect)
  rows <- do.call(paste, c(quoted, sep = ", "))
  clauses <- c(clauses, paste("VALUES", paste0("(", rows, ")", collapse = ",\n  ")))
  # ON CONFLICT
  if (dialect == "postgres" && on_conflict == "ignore") {
    clauses <- c(clauses, "ON CONFLICT DO NOTHING")
  }
  # RETURNING
  if (length(returning) > 0) {
    if (dialect == "mysql") {
      stop("`returning` is not supported for dialect \"mysql\".", call. = FALSE)
    }
    keys <- sql_quote(returning, "identifier", dialect)
    clauses <- c(clauses, paste("RETURNING", paste(keys, collapse = ", ")))
  }
  paste(clauses, collapse = "\n")
}

#' Build an UPDATE statement
#'
#' Emits one clause per argument: target, assignments, filter and returning list.
#'
#' @param table The table to update.
#' @param set A named list of new values, one element per column.
#' @param where Character vector of SQL conditions, combined with `AND`.
#' @param returning Column names to return, or `NULL`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL string, one clause per line.
#' @export
build_update <- function(table, set, where = NULL, returning = NULL,
                         dialect = c("postgres", "mysql", "sqlite")) {
  dialect <- match.arg(dialect)
  if (!is.list(set) || is.null(names(set)) || any(!nzchar(names(set)))) {
    stop("`set` must be a named list.", call. = FALSE)
  }
  clauses <- character()
  # UPDATE
  clauses <- c(clauses, paste("UPDATE", sql_quote(table, "identifier", dialect)))
  # SET
  lhs <- sql_quote(names(set), "identifier", dialect)
  rhs <- vapply(set, sql_quote, character(1), kind = "literal", dialect = dialect)
  clauses <- c(clauses, paste("SET", paste(lhs, "=", rhs, collapse = ", ")))
  # WHERE
  if (length(where) > 0) {
    clauses <- c(clauses, paste("WHERE", paste(where, collapse = " AND ")))
  }
  # RETURNING
  if (length(returning) > 0) {
    if (dialect == "mysql") {
      stop("`returning` is not supported for dialect \"mysql\".", call. = FALSE)
    }
    keys <- sql_quote(returning, "identifier", dialect)
    clauses <- c(clauses, paste("RETURNING", paste(keys, collapse = ", ")))
  }
  paste(clauses, collapse = "\n")
}

#' Build a DELETE statement
#'
#' Emits one clause per argument: target, filter and returning list.
#'
#' @param table The table to delete from.
#' @param where Character vector of SQL conditions, combined with `AND`.
#' @param returning Column names to return, or `NULL`.
#' @param allow_all Set to `TRUE` to permit a DELETE without `where`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL string, one clause per line.
#' @export
build_delete <- function(table, where = NULL, returning = NULL, allow_all = FALSE,
                         dialect = c("postgres", "mysql", "sqlite")) {
  dialect <- match.arg(dialect)
  if (length(where) == 0 && !isTRUE(allow_all)) {
    stop("`where` is empty; set `allow_all = TRUE` to delete every row.", call. = FALSE)
  }
  clauses <- character()
  # DELETE FROM
  clauses <- c(clauses, paste("DELETE FROM", sql_quote(table, "identifier", dialect)))
  # WHERE
  if (length(where) > 0) {
    clauses <- c(clauses, paste("WHERE", paste(where, collapse = " AND ")))
  }
  # RETURNING
  if (length(returning) > 0) {
    if (dialect == "mysql") {
      stop("`returning` is not supported for dialect \"mysql\".", call. = FALSE)
    }
    keys <- sql_quote(returning, "identifier", dialect)
    clauses <- c(clauses, paste("RETURNING", paste(keys, collapse = ", ")))
  }
  paste(clauses, collapse = "\n")
}

#' Build a JOIN clause
#'
#' Renders the join keyword for `type` and an `ON` condition per key pair.
#'
#' @param left,right The left and right table names.
#' @param by A character vector of join keys; names give the left-hand column
#'   when the two sides differ.
#' @param type One of `"inner"`, `"left"`, `"right"`, `"full"` or `"cross"`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL string starting with `FROM`.
#' @export
build_join <- function(left, right, by = character(),
                       type = c("inner", "left", "right", "full", "cross"),
                       dialect = c("postgres", "mysql", "sqlite")) {
  type <- match.arg(type)
  dialect <- match.arg(dialect)
  lt <- sql_quote(left, "identifier", dialect)
  rt <- sql_quote(right, "identifier", dialect)
  left_keys <- if (is.null(names(by))) by else ifelse(nzchar(names(by)), names(by), by)
  on <- paste(
    paste0(lt, ".", sql_quote(left_keys, "identifier", dialect)), "=",
    paste0(rt, ".", sql_quote(unname(by), "identifier", dialect)),
    collapse = " AND "
  )
  keyword <- switch(type,
    inner = {
      if (length(by) == 0) stop("`by` must name at least one key.", call. = FALSE)
      "INNER JOIN"
    },
    left = {
      if (length(by) == 0) stop("`by` must name at least one key.", call. = FALSE)
      "LEFT JOIN"
    },
    right = {
      if (length(by) == 0) stop("`by` must name at least one key.", call. = FALSE)
      if (dialect == "sqlite") {
        msg <- sprintf("%s is not supported for dialect \"%s\".", "RIGHT JOIN", dialect)
        stop(msg, call. = FALSE)
      }
      "RIGHT JOIN"
    },
    full = {
      if (length(by) == 0) stop("`by` must name at least one key.", call. = FALSE)
      if (dialect != "postgres") {
        msg <- sprintf("%s is not supported for dialect \"%s\".", "FULL JOIN", dialect)
        stop(msg, call. = FALSE)
      }
      "FULL JOIN"
    },
    cross = {
      if (length(by) > 0) stop("`by` must be empty for a cross join.", call. = FALSE)
      "CROSS JOIN"
    }
  )
  out <- paste("FROM", lt, keyword, rt)
  if (type != "cross") out <- paste(out, "ON", on)
  out
}
