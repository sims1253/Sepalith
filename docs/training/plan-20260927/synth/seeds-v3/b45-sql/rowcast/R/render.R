#' Turn result cells into display text
#'
#' Formats a column according to its kind and replaces missing values with `na`.
#'
#' @param x A result column.
#' @param na The text shown for missing values.
#' @param digits Significant digits for doubles.
#' @return A character vector the same length as `x`.
#' @keywords internal
cells.as.text <- function(x, na = "NULL", digits = 6L) {
  kind <- if (is.integer(x)) {
    "integer"
  } else if (is.numeric(x)) {
    "numeric"
  } else if (is.logical(x)) {
    "logical"
  } else if (inherits(x, "Date")) {
    "date"
  } else if (inherits(x, "POSIXct")) {
    "timestamp"
  } else {
    "character"
  }
  out <- switch(kind,
    integer = {
      as.character(x)
    },
    numeric = {
      trimws(formatC(x, digits = digits, format = "g"))
    },
    logical = {
      ifelse(x, "true", "false")
    },
    date = {
      format(x, "%Y-%m-%d")
    },
    timestamp = {
      format(x, "%Y-%m-%d %H:%M:%S", tz = "UTC")
    },
    character = {
      as.character(x)
    }
  )
  out[is.na(x)] <- na
  out
}

#' Render a result set as text lines
#'
#' Every format builds a row template `fmt`, the `head` lines and the `body` lines.
#'
#' @param df A data frame of query results.
#' @param format One of `"text"`, `"markdown"`, `"csv"`, `"html"` or `"insert"`.
#' @param table The target table name for `"insert"`.
#' @return A character vector of output lines.
#' @export
render.result <- function(df, format = c("text", "markdown", "csv", "html", "insert"),
                          table = "result") {
  format <- match.arg(format)
  header <- names(df)
  cells <- lapply(df, cells.as.text)
  fill <- function(fmt, cols) do.call(sprintf, c(list(fmt), unname(cols)))
  slots <- rep("%s", length(header))
  switch(format,
    text = {
      widths <- pmax(nchar(header), vapply(cells, function(v) max(0L, nchar(v)), integer(1)))
      fmt <- paste(sprintf("%%-%ds", widths), collapse = "  ")
      head <- c(fill(fmt, as.list(header)), fill(fmt, as.list(strrep("-", widths))))
      body <- fill(fmt, cells)
      c(head, body)
    },
    markdown = {
      fmt <- paste0("| ", paste(slots, collapse = " | "), " |")
      head <- c(fill(fmt, as.list(header)), fill(fmt, as.list(rep("---", length(header)))))
      body <- fill(fmt, cells)
      c(head, body)
    },
    csv = {
      fmt <- paste(slots, collapse = ",")
      quoted <- lapply(cells, function(v) {
        ifelse(grepl("[\",]", v), paste0("\"", gsub("\"", "\"\"", v), "\""), v)
      })
      head <- fill(fmt, as.list(header))
      body <- fill(fmt, quoted)
      c(head, body)
    },
    html = {
      fmt <- paste0("<tr>", paste0("<td>", slots, "</td>", collapse = ""), "</tr>")
      head <- c("<table>", paste0("<tr>", paste0("<th>", header, "</th>", collapse = ""), "</tr>"))
      body <- fill(fmt, cells)
      c(head, body, "</table>")
    },
    insert = {
      fmt <- sprintf("INSERT INTO %s (%s) VALUES (%s);", table,
                     paste(header, collapse = ", "), paste(slots, collapse = ", "))
      literal <- lapply(df, function(v) {
        txt <- cells.as.text(v)
        if (is.numeric(v)) txt else ifelse(is.na(v), "NULL", sprintf("'%s'", gsub("'", "''", txt)))
      })
      head <- character()
      body <- fill(fmt, literal)
      c(head, body)
    }
  )
}

#' Shorten a result set for display
#'
#' Keeps the first and last `edge` rows and marks the gap with a row of `"..."`.
#'
#' @param df A data frame of query results.
#' @param max.rows Results with at most this many rows are returned unchanged.
#' @param edge Rows to keep at the head and at the tail.
#' @return A data frame of character columns.
#' @export
shorten.result <- function(df, max.rows = 20L, edge = 5L) {
  if (edge * 2L >= max.rows) {
    stop("`edge` must be less than half of `max.rows`.", call. = FALSE)
  }
  cells <- as.data.frame(lapply(df, cells.as.text), stringsAsFactors = FALSE)
  names(cells) <- names(df)
  n <- nrow(cells)
  if (n <= max.rows) {
    return(cells)
  }
  # Head rows
  head.rows <- cells[seq_len(edge), , drop = FALSE]
  # Tail rows
  tail.rows <- cells[seq.int(n - edge + 1L, n), , drop = FALSE]
  # Gap marker
  gap <- as.data.frame(as.list(rep("...", ncol(cells))), stringsAsFactors = FALSE)
  names(gap) <- names(cells)
  out <- rbind(head.rows, gap, tail.rows)
  rownames(out) <- NULL
  out
}

#' Batch rows into multi-row INSERT statements
#'
#' Groups `batch.size` rows per statement, quoting values by column kind.
#'
#' @param df A data frame of rows to insert.
#' @param table The target table name.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`.
#' @param batch.size The maximum number of rows per statement.
#' @return A character vector with one statement per batch.
#' @export
result.to.inserts <- function(df, table, dialect = c("postgres", "mysql", "sqlite"),
                              batch.size = 500L) {
  dialect <- match.arg(dialect)
  quote <- if (dialect == "mysql") "`" else "\""
  qi <- function(x) paste0(quote, gsub(quote, strrep(quote, 2L), x, fixed = TRUE), quote)
  literal <- lapply(df, function(v) {
    txt <- cells.as.text(v)
    if (is.numeric(v)) {
      txt
    } else if (is.logical(v)) {
      if (dialect == "postgres") toupper(txt) else ifelse(v, "1", "0")
    } else {
      sprintf("'%s'", gsub("'", "''", txt, fixed = TRUE))
    }
  })
  # Missing values are NULL whatever the kind
  for (col in names(literal)) {
    literal[[col]][is.na(df[[col]])] <- "NULL"
  }
  tuples <- sprintf("(%s)", do.call(paste, c(unname(literal), sep = ", ")))
  head <- sprintf("INSERT INTO %s (%s) VALUES", qi(table), paste(qi(names(df)), collapse = ", "))
  batch <- ceiling(seq_along(tuples) / batch.size)
  out <- character(max(0, batch))
  for (b in unique(batch)) {
    out[b] <- paste0(head, "\n  ", paste(tuples[batch == b], collapse = ",\n  "), ";")
  }
  out
}

#' Header lines describing a query result
#'
#' Produces the lines `query`, `rows`, `columns` and `elapsed`, in that order.
#'
#' @param df A data frame of query results.
#' @param sql The SQL that produced `df`.
#' @param elapsed Elapsed time in seconds.
#' @param width Maximum width of the `query` line.
#' @return A named character vector with elements `query`, `rows`, `columns`, `elapsed`.
#' @export
result.header <- function(df, sql, elapsed, width = 60L) {
  one.line <- gsub("\\s+", " ", trimws(sql))
  # Line `query`
  if (nchar(one.line) > width) {
    one.line <- paste0(substr(one.line, 1L, width - 3L), "...")
  }
  query <- sprintf("-- query: %s", one.line)
  # Line `rows`
  rows <- sprintf("-- rows: %d", nrow(df))
  # Line `columns`
  columns <- sprintf("-- columns: %s", paste(names(df), collapse = ", "))
  # Line `elapsed`
  elapsed <- sprintf("-- elapsed: %.3f s", elapsed)
  c(query = query, rows = rows, columns = columns, elapsed = elapsed)
}

#' Write a rendered result set to a file
#'
#' Picks the file extension from `format`, renders with [render.result()] and writes it.
#'
#' @param df A data frame of query results.
#' @param stem The output path without extension.
#' @param format One of `"text"`, `"markdown"`, `"csv"`, `"html"` or `"insert"`.
#' @return The path written, invisibly.
#' @export
write.result <- function(df, stem, format = c("text", "markdown", "csv", "html", "insert")) {
  format <- match.arg(format)
  extensions <- c(text = "txt", markdown = "md", csv = "csv", html = "html", insert = "sql")
  path <- paste0(stem, ".", extensions[[format]])
  lines <- render.result(df, format, table = basename(stem))
  writeLines(lines, path)
  invisible(path)
}
