#' Quote SQL identifiers for a dialect
#'
#' Wraps each part of a (possibly schema-qualified) name in the dialect's quotes.
#'
#' @param x A character vector of identifiers, parts separated by `"."`.
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"`, `"mssql"` or `"oracle"`.
#' @return A character vector of quoted identifiers, the same length as `x`.
#' @export
quote_identifier <- function(x, dialect = c("postgres", "mysql", "sqlite", "mssql", "oracle")) {
  dialect <- match.arg(dialect)
  if (!is.character(x)) {
    stop("`x` must be a character vector.", call. = FALSE)
  }
  if (anyNA(x) || any(!nzchar(x))) {
    stop("`x` must not contain missing or empty identifiers.", call. = FALSE)
  }
  parts <- strsplit(x, ".", fixed = TRUE)

  # Escape the closing quote by doubling it, then wrap each part
  quoted <- lapply(parts, function(p) {
    switch(dialect,
      postgres = {
        p <- gsub("\"", "\"\"", p, fixed = TRUE)
        paste0("\"", p, "\"")
      },
      mysql = {
        p <- gsub("`", "``", p, fixed = TRUE)
        paste0("`", p, "`")
      },
      sqlite = {
        p <- gsub("\"", "\"\"", p, fixed = TRUE)
        paste0("\"", p, "\"")
      },
      mssql = {
        p <- gsub("]", "]]", p, fixed = TRUE)
        paste0("[", p, "]")
      },
      oracle = {
        p <- gsub("\"", "\"\"", toupper(p), fixed = TRUE)
        paste0("\"", p, "\"")
      }
    )
  })
  return(vapply(quoted, paste, character(1), collapse = "."))
}

#' Quote R values as SQL literals
#'
#' Chooses a literal syntax from the class of `x`; missing values become `NULL`.
#'
#' @param x An atomic vector: character, numeric, logical, `Date` or `POSIXct`.
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"`, `"mssql"` or `"oracle"`.
#' @return A character vector of SQL literals, the same length as `x`.
#' @export
quote_literal <- function(x, dialect = c("postgres", "mysql", "sqlite", "mssql", "oracle")) {
  dialect <- match.arg(dialect)
  type <- if (inherits(x, "POSIXct")) {
    "timestamp"
  } else if (inherits(x, "Date")) {
    "date"
  } else if (is.logical(x)) {
    "logical"
  } else if (is.numeric(x)) {
    "numeric"
  } else {
    "character"
  }
  out <- switch(type,
    character = {
      s <- gsub("'", "''", as.character(x), fixed = TRUE)
      if (dialect == "mysql") s <- gsub("\\", "\\\\", s, fixed = TRUE)
      paste0("'", s, "'")
    },
    numeric = {
      s <- trimws(formatC(as.numeric(x), format = "fg", digits = 15))
      s
    },
    logical = {
      if (dialect %in% c("sqlite", "mssql", "oracle")) {
        ifelse(x, "1", "0")
      } else {
        ifelse(x, "TRUE", "FALSE")
      }
    },
    date = {
      s <- format(x, "%Y-%m-%d")
      if (dialect == "oracle") paste0("DATE '", s, "'") else paste0("'", s, "'")
    },
    timestamp = {
      s <- format(x, "%Y-%m-%d %H:%M:%S", tz = "UTC")
      if (dialect == "oracle") paste0("TIMESTAMP '", s, "'") else paste0("'", s, "'")
    }
  )
  out[is.na(x)] <- "NULL"
  return(out)
}

#' Syntax features of a SQL dialect
#'
#' Collects the small syntax differences the builders in this package rely on.
#'
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"`, `"mssql"` or `"oracle"`.
#' @return A list with elements `ident_open`, `ident_close`, `true`, `false`,
#'   `concat`, `limit_style`, `placeholder` and `returning`.
#' @keywords internal
dialect_features <- function(dialect = c("postgres", "mysql", "sqlite", "mssql", "oracle")) {
  dialect <- match.arg(dialect)
  switch(dialect,
    postgres = list(
      ident_open = "\"", ident_close = "\"",
      true = "TRUE", false = "FALSE",
      concat = "||", limit_style = "limit",
      placeholder = "dollar", returning = TRUE
    ),
    mysql = list(
      ident_open = "`", ident_close = "`",
      true = "TRUE", false = "FALSE",
      concat = "CONCAT", limit_style = "limit",
      placeholder = "qmark", returning = FALSE
    ),
    sqlite = list(
      ident_open = "\"", ident_close = "\"",
      true = "1", false = "0",
      concat = "||", limit_style = "limit",
      placeholder = "qmark", returning = TRUE
    ),
    mssql = list(
      ident_open = "[", ident_close = "]",
      true = "1", false = "0",
      concat = "+", limit_style = "fetch",
      placeholder = "at", returning = FALSE
    ),
    oracle = list(
      ident_open = "\"", ident_close = "\"",
      true = "1", false = "0",
      concat = "||", limit_style = "fetch",
      placeholder = "colon", returning = FALSE
    )
  )
}

#' Render a row-limiting clause
#'
#' Produces `LIMIT`/`OFFSET` or the standard `OFFSET ... FETCH` form.
#'
#' @param n The maximum number of rows to return.
#' @param offset The number of rows to skip first.
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"`, `"mssql"` or `"oracle"`.
#' @return A single string holding the clause.
#' @export
render_limit <- function(n, offset = 0,
                         dialect = c("postgres", "mysql", "sqlite", "mssql", "oracle")) {
  dialect <- match.arg(dialect)
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 0) {
    stop("`n` must be a single non-negative number.", call. = FALSE)
  }
  if (!is.numeric(offset) || length(offset) != 1L || is.na(offset) || offset < 0) {
    stop("`offset` must be a single non-negative number.", call. = FALSE)
  }
  n <- as.integer(n)
  offset <- as.integer(offset)
  switch(dialect,
    postgres = {
      out <- sprintf("LIMIT %d", n)
      if (offset > 0L) out <- sprintf("%s OFFSET %d", out, offset)
      out
    },
    mysql = {
      out <- sprintf("LIMIT %d", n)
      if (offset > 0L) out <- sprintf("LIMIT %d, %d", offset, n)
      out
    },
    sqlite = {
      out <- sprintf("LIMIT %d", n)
      if (offset > 0L) out <- sprintf("%s OFFSET %d", out, offset)
      out
    },
    mssql = {
      out <- sprintf("OFFSET %d ROWS FETCH NEXT %d ROWS ONLY", offset, n)
      out
    },
    oracle = {
      out <- sprintf("FETCH FIRST %d ROWS ONLY", n)
      if (offset > 0L) out <- sprintf("OFFSET %d ROWS FETCH NEXT %d ROWS ONLY", offset, n)
      out
    }
  )
}

#' Render bind parameter placeholders
#'
#' Numbers or repeats the placeholder token used by the dialect's driver.
#'
#' @param n The number of parameters.
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"`, `"mssql"` or `"oracle"`.
#' @param names Optional parameter names; used where the dialect supports them.
#' @return A character vector of `n` placeholders.
#' @export
render_placeholders <- function(n, dialect = c("postgres", "mysql", "sqlite", "mssql", "oracle"),
                                names = NULL) {
  dialect <- match.arg(dialect)
  if (!is.null(names) && length(names) != n) {
    stop("`names` must have length `n`.", call. = FALSE)
  }
  idx <- seq_len(n)
  switch(dialect,
    postgres = {
      paste0("$", idx)
    },
    mysql = {
      rep("?", n)
    },
    sqlite = {
      if (is.null(names)) paste0("?", idx) else paste0(":", names)
    },
    mssql = {
      if (is.null(names)) paste0("@p", idx) else paste0("@", names)
    },
    oracle = {
      if (is.null(names)) paste0(":", idx) else paste0(":", names)
    }
  )
}

#' Cast a SQL expression to a portable type
#'
#' Wraps `expr` in a `CAST()` using the dialect's name for the target type.
#'
#' @param expr A character vector of SQL expressions.
#' @param type One of `"integer"`, `"numeric"`, `"text"`, `"boolean"`, `"date"` or
#'   `"timestamp"`.
#' @param dialect One of `"postgres"`, `"mysql"`, `"sqlite"`, `"mssql"` or `"oracle"`.
#' @return A character vector of SQL expressions.
#' @export
cast_expression <- function(expr,
                            type = c("integer", "numeric", "text", "boolean", "date", "timestamp"),
                            dialect = c("postgres", "mysql", "sqlite", "mssql", "oracle")) {
  type <- match.arg(type)
  dialect <- match.arg(dialect)
  # One row per dialect, in the order of the `dialect` choices
  target <- switch(type,
    integer = c(
      postgres = "INTEGER", mysql = "SIGNED", sqlite = "INTEGER",
      mssql = "INT", oracle = "NUMBER(10)"
    ),
    numeric = c(
      postgres = "NUMERIC", mysql = "DECIMAL(38, 10)", sqlite = "REAL",
      mssql = "DECIMAL(38, 10)", oracle = "NUMBER"
    ),
    text = c(
      postgres = "TEXT", mysql = "CHAR", sqlite = "TEXT",
      mssql = "NVARCHAR(MAX)", oracle = "VARCHAR2(4000)"
    ),
    boolean = c(
      postgres = "BOOLEAN", mysql = "UNSIGNED", sqlite = "INTEGER",
      mssql = "BIT", oracle = "NUMBER(1)"
    ),
    date = c(
      postgres = "DATE", mysql = "DATE", sqlite = "TEXT",
      mssql = "DATE", oracle = "DATE"
    ),
    timestamp = c(
      postgres = "TIMESTAMP", mysql = "DATETIME", sqlite = "TEXT",
      mssql = "DATETIME2", oracle = "TIMESTAMP"
    )
  )
  sprintf("CAST(%s AS %s)", expr, target[[dialect]])
}
