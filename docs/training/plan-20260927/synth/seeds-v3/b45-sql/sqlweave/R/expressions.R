#' Compile a single filter into a SQL condition
#'
#' Quotes `column`, renders `value` as literals and applies the operator `op`.
#'
#' @param column A column name.
#' @param op One of `"eq"`, `"ne"`, `"lt"`, `"gt"`, `"in"`, `"not_in"`, `"between"`,
#'   `"like"`, `"is_null"` or `"not_null"`.
#' @param value The comparison value(s); ignored for the null tests.
#' @return A single SQL condition string.
#' @export
compile_filter <- function(column,
                           op = c("eq", "ne", "lt", "gt", "in", "not_in", "between",
                                  "like", "is_null", "not_null"),
                           value = NULL) {
  op <- match.arg(op)
  col <- paste0("\"", gsub("\"", "\"\"", column, fixed = TRUE), "\"")
  lit <- function(v) {
    if (is.numeric(v)) return(trimws(formatC(v, format = "fg", digits = 15)))
    paste0("'", gsub("'", "''", as.character(v), fixed = TRUE), "'")
  }
  switch(op,
    eq = {
      if (length(value) != 1L) stop("`value` must have length 1 for `eq`.", call. = FALSE)
      paste(col, "=", lit(value))
    },
    ne = {
      if (length(value) != 1L) stop("`value` must have length 1 for `ne`.", call. = FALSE)
      paste(col, "<>", lit(value))
    },
    lt = {
      if (length(value) != 1L) stop("`value` must have length 1 for `lt`.", call. = FALSE)
      paste(col, "<", lit(value))
    },
    gt = {
      if (length(value) != 1L) stop("`value` must have length 1 for `gt`.", call. = FALSE)
      paste(col, ">", lit(value))
    },
    `in` = {
      if (length(value) < 1L) stop("`value` must have length >= 1 for `in`.", call. = FALSE)
      paste0(col, " IN (", paste(lit(value), collapse = ", "), ")")
    },
    not_in = {
      if (length(value) < 1L) stop("`value` must have length >= 1 for `not_in`.", call. = FALSE)
      paste0(col, " NOT IN (", paste(lit(value), collapse = ", "), ")")
    },
    between = {
      if (length(value) != 2L) stop("`value` must have length 2 for `between`.", call. = FALSE)
      paste(col, "BETWEEN", lit(value[1]), "AND", lit(value[2]))
    },
    like = {
      if (length(value) != 1L) stop("`value` must have length 1 for `like`.", call. = FALSE)
      paste(col, "LIKE", lit(value))
    },
    is_null = {
      paste(col, "IS NULL")
    },
    not_null = {
      paste(col, "IS NOT NULL")
    }
  )
}

#' Compile an aggregate expression
#'
#' Renders the aggregate `fun` over `column` with an `AS` alias.
#'
#' @param column A column name, or `"*"` for `count`.
#' @param fun One of `"count"`, `"count_distinct"`, `"sum"`, `"avg"`, `"min"`, `"max"`
#'   or `"median"`.
#' @param alias The output column name; defaults to `<fun>_<column>`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL select-list item.
#' @export
compile_aggregate <- function(column,
                              fun = c("count", "count_distinct", "sum", "avg", "min", "max",
                                      "median"),
                              alias = NULL, dialect = c("postgres", "mysql", "sqlite")) {
  fun <- match.arg(fun)
  dialect <- match.arg(dialect)
  q <- if (dialect == "mysql") "`" else "\""
  col <- if (identical(column, "*")) "*" else paste0(q, column, q)
  if (is.null(alias)) {
    alias <- paste0(fun, "_", if (identical(column, "*")) "all" else column)
  }
  expr <- switch(fun,
    count = {
      sprintf("COUNT(%s)", col)
    },
    count_distinct = {
      if (col == "*") stop("`column` must name a column for `count_distinct`.", call. = FALSE)
      sprintf("COUNT(DISTINCT %s)", col)
    },
    sum = {
      if (col == "*") stop("`column` must name a column for `sum`.", call. = FALSE)
      sprintf("SUM(%s)", col)
    },
    avg = {
      if (col == "*") stop("`column` must name a column for `avg`.", call. = FALSE)
      sprintf("AVG(%s)", col)
    },
    min = {
      if (col == "*") stop("`column` must name a column for `min`.", call. = FALSE)
      sprintf("MIN(%s)", col)
    },
    max = {
      if (col == "*") stop("`column` must name a column for `max`.", call. = FALSE)
      sprintf("MAX(%s)", col)
    },
    median = {
      if (col == "*") stop("`column` must name a column for `median`.", call. = FALSE)
      if (dialect != "postgres") {
        stop(sprintf("`median` is not supported for dialect \"%s\".", dialect), call. = FALSE)
      }
      sprintf("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY %s)", col)
    }
  )
  paste(expr, "AS", paste0(q, alias, q))
}

#' Compile a window function call
#'
#' Renders `fun` with an `OVER (PARTITION BY ... ORDER BY ...)` clause.
#'
#' @param fun One of `"row_number"`, `"rank"`, `"dense_rank"`, `"lag"`, `"lead"` or
#'   `"running_sum"`.
#' @param column The column used by `lag`, `lead` and `running_sum`.
#' @param partition_by Column names to partition by.
#' @param order_by Column names to order by within each partition.
#' @param offset Row offset for `lag` and `lead`.
#' @return A single SQL expression.
#' @export
compile_window <- function(fun = c("row_number", "rank", "dense_rank", "lag", "lead",
                                   "running_sum"),
                           column = NULL, partition_by = NULL, order_by = NULL, offset = 1L) {
  fun <- match.arg(fun)
  qi <- function(x) paste0("\"", gsub("\"", "\"\"", x, fixed = TRUE), "\"")
  over <- character()
  if (length(partition_by) > 0) {
    over <- c(over, paste("PARTITION BY", paste(qi(partition_by), collapse = ", ")))
  }
  if (length(order_by) > 0) {
    over <- c(over, paste("ORDER BY", paste(qi(order_by), collapse = ", ")))
  }
  over <- paste0("OVER (", paste(over, collapse = " "), ")")
  call <- switch(fun,
    row_number = {
      "ROW_NUMBER()"
    },
    rank = {
      if (length(order_by) == 0) stop("`order_by` is required for `rank`.", call. = FALSE)
      "RANK()"
    },
    dense_rank = {
      if (length(order_by) == 0) stop("`order_by` is required for `dense_rank`.", call. = FALSE)
      "DENSE_RANK()"
    },
    lag = {
      if (is.null(column)) stop("`column` is required for `lag`.", call. = FALSE)
      sprintf("LAG(%s, %d)", qi(column), as.integer(offset))
    },
    lead = {
      if (is.null(column)) stop("`column` is required for `lead`.", call. = FALSE)
      sprintf("LEAD(%s, %d)", qi(column), as.integer(offset))
    },
    running_sum = {
      if (is.null(column)) stop("`column` is required for `running_sum`.", call. = FALSE)
      sprintf("SUM(%s)", qi(column))
    }
  )
  paste(call, over)
}

#' Compile a CASE expression that bins a numeric column
#'
#' Produces one `WHEN` arm per interval: below the first break, between each
#' pair of breaks, and at or above the last break.
#'
#' @param column A column name.
#' @param breaks A sorted numeric vector of interior cut points.
#' @param labels Labels for the `length(breaks) + 1` bins.
#' @param alias The output column name.
#' @return A single SQL expression.
#' @export
compile_case <- function(column, breaks, labels, alias = paste0(column, "_bin")) {
  if (!is.numeric(breaks) || is.unsorted(breaks, strictly = TRUE)) {
    stop("`breaks` must be a strictly increasing numeric vector.", call. = FALSE)
  }
  if (length(labels) != length(breaks) + 1L) {
    stop("`labels` must have one more element than `breaks`.", call. = FALSE)
  }
  col <- paste0("\"", column, "\"")
  num <- trimws(formatC(breaks, format = "fg", digits = 15))
  lab <- paste0("'", gsub("'", "''", labels, fixed = TRUE), "'")
  n <- length(breaks)
  arms <- character(n + 1L)
  # Lower open bin
  arms[1] <- sprintf("WHEN %s < %s THEN %s", col, num[1], lab[1])
  # Interior bins
  for (i in seq_len(n - 1L)) {
    arms[i + 1L] <- sprintf("WHEN %s >= %s AND %s < %s THEN %s",
                            col, num[i], col, num[i + 1L], lab[i + 1L])
  }
  # Upper open bin
  arms[n + 1L] <- sprintf("WHEN %s >= %s THEN %s", col, num[n], lab[n + 1L])
  body <- paste0("  ", arms, collapse = "\n")
  sprintf("CASE\n%s\n  ELSE NULL\nEND AS \"%s\"", body, alias)
}

#' Compile a date part extraction
#'
#' Extracts `part` from a date or timestamp column using the dialect's syntax.
#'
#' @param column A column name.
#' @param part One of `"year"`, `"month"`, `"day"`, `"hour"` or `"dow"`.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @return A single SQL expression returning an integer.
#' @export
compile_date_part <- function(column, part = c("year", "month", "day", "hour", "dow"),
                              dialect = c("postgres", "mysql", "sqlite")) {
  part <- match.arg(part)
  dialect <- match.arg(dialect)
  col <- if (dialect == "mysql") paste0("`", column, "`") else paste0("\"", column, "\"")
  # strftime codes used by sqlite, keyed by `part`
  codes <- c(year = "%Y", month = "%m", day = "%d", hour = "%H", dow = "%w")
  switch(part,
    year = switch(dialect,
      postgres = sprintf("EXTRACT(YEAR FROM %s)", col),
      mysql = sprintf("YEAR(%s)", col),
      sqlite = sprintf("CAST(strftime('%s', %s) AS INTEGER)", codes[["year"]], col)
    ),
    month = switch(dialect,
      postgres = sprintf("EXTRACT(MONTH FROM %s)", col),
      mysql = sprintf("MONTH(%s)", col),
      sqlite = sprintf("CAST(strftime('%s', %s) AS INTEGER)", codes[["month"]], col)
    ),
    day = switch(dialect,
      postgres = sprintf("EXTRACT(DAY FROM %s)", col),
      mysql = sprintf("DAY(%s)", col),
      sqlite = sprintf("CAST(strftime('%s', %s) AS INTEGER)", codes[["day"]], col)
    ),
    hour = switch(dialect,
      postgres = sprintf("EXTRACT(HOUR FROM %s)", col),
      mysql = sprintf("HOUR(%s)", col),
      sqlite = sprintf("CAST(strftime('%s', %s) AS INTEGER)", codes[["hour"]], col)
    ),
    dow = switch(dialect,
      postgres = sprintf("EXTRACT(DOW FROM %s)", col),
      mysql = sprintf("(DAYOFWEEK(%s) - 1)", col),
      sqlite = sprintf("CAST(strftime('%s', %s) AS INTEGER)", codes[["dow"]], col)
    )
  )
}
