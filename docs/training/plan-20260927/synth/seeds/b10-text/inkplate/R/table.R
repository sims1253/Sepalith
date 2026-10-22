#' Format data frame columns as character
#'
#' Numbers are rounded to `digits` and formatted without scientific
#' notation; missing values become `na`.
#'
#' @param data A data frame to render.
#' @param digits Number of decimal places for numeric columns.
#' @param na String used for missing values.
#' @return A character matrix with the column names of `data`.
#' @noRd
format.cells <- function(data, digits = 2L, na = "") {
  cells <- vapply(data, function(col) {
    out <- if (is.numeric(col) && !is.integer(col)) {
      formatC(col, format = "f", digits = digits, big.mark = ",")
    } else {
      as.character(col)
    }
    out[is.na(col)] <- na
    out
  }, character(nrow(data)))
  cells <- matrix(cells, nrow = nrow(data), dimnames = list(NULL, names(data)))
  cells
}

#' Choose a default alignment for each column
#'
#' @param data A data frame to render.
#' @return A character vector of `"l"` or `"r"`, one per column.
#' @noRd
default.align <- function(data) {
  vapply(data, function(col) if (is.numeric(col)) "r" else "l", character(1), USE.NAMES = FALSE)
}

#' Render a data frame as a Markdown pipe table
#'
#' @param data A data frame to render.
#' @param align Optional character vector of `"l"`, `"c"` or `"r"`, one per
#'   column.
#' @param digits Number of decimal places for numeric columns.
#' @return A single string containing the table.
#' @family table renderers
#' @export
table.markdown <- function(data, align = NULL, digits = 2L) {
  align <- if (is.null(align)) default.align(data) else align
  cells <- format.cells(data, digits = digits)
  cells[] <- escape.markdown(cells)
  header <- escape.markdown(names(data))
  widths <- pmax(nchar(header), apply(rbind(cells, ""), 2, function(c) max(nchar(c))), 3L)
  rule <- vapply(seq_along(widths), function(j) {
    dashes <- strrep("-", widths[j] - 1L)
    switch(align[j],
      l = paste0(":", dashes),
      r = paste0(dashes, ":"),
      c = paste0(":", strrep("-", widths[j] - 2L), ":")
    )
  }, character(1))
  pad.row <- function(row) {
    paste0("| ", paste(formatC(row, width = widths, flag = "-"), collapse = " | "), " |")
  }
  body <- apply(cells, 1, pad.row)
  paste(c(pad.row(header), paste0("| ", paste(rule, collapse = " | "), " |"), body),
    collapse = "\n"
  )
}

#' Render a data frame as an HTML table
#'
#' @param data A data frame to render.
#' @param align Optional character vector of `"l"`, `"c"` or `"r"`, one per
#'   column.
#' @param digits Number of decimal places for numeric columns.
#' @return A single string containing the table.
#' @family table renderers
#' @export
table.html <- function(data, align = NULL, digits = 2L) {
  align <- if (is.null(align)) default.align(data) else align
  cells <- format.cells(data, digits = digits)
  cells[] <- escape.html(cells)
  css <- c(l = "left", c = "center", r = "right")[align]
  head <- paste0("<th style=\"text-align:", css, "\">", escape.html(names(data)), "</th>")
  rows <- apply(cells, 1, function(row) {
    tds <- paste0("<td style=\"text-align:", css, "\">", row, "</td>")
    paste0("<tr>", paste(tds, collapse = ""), "</tr>")
  })
  paste(c(
    "<table>",
    paste0("<thead><tr>", paste(head, collapse = ""), "</tr></thead>"),
    "<tbody>", rows, "</tbody>",
    "</table>"
  ), collapse = "\n")
}

#' Render a data frame as a LaTeX tabular
#'
#' @param data A data frame to render.
#' @param align Optional character vector of `"l"`, `"c"` or `"r"`, one per
#'   column.
#' @param digits Number of decimal places for numeric columns.
#' @return A single string containing the table.
#' @family table renderers
#' @export
table.latex <- function(data, align = NULL, digits = 2L) {
  align <- if (is.null(align)) default.align(data) else align
  cells <- format.cells(data, digits = digits)
  cells[] <- escape.latex(cells)
  header <- paste(escape.latex(names(data)), collapse = " & ")
  rows <- apply(cells, 1, function(row) paste(row, collapse = " & "))
  paste(c(
    sprintf("\\begin{tabular}{%s}", paste(align, collapse = "")),
    "\\hline",
    paste0(header, " \\\\"),
    "\\hline",
    paste0(rows, " \\\\"),
    "\\hline",
    "\\end{tabular}"
  ), collapse = "\n")
}

#' Render a data frame as a table in the chosen format
#'
#' @param data A data frame to render.
#' @param format Target output format.
#' @param ... Passed to the format-specific renderer.
#' @return A single string containing the table.
#' @family table renderers
#' @export
render.table <- function(data, format = c("markdown", "html", "latex"), ...) {
  format <- match.arg(format)
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  switch(format,
    markdown = table.markdown(data, ...),
    html = table.html(data, ...),
    latex = table.latex(data, ...)
  )
}

#' Truncate long cell text with an ellipsis
#'
#' @param x A character vector of cell values.
#' @param width Maximum number of characters to keep, including the
#'   ellipsis.
#' @return A character vector the same length as `x`.
#' @export
truncate.cells <- function(x, width = 40L) {
  too.long <- !is.na(x) & stringi::stri_length(x) > width
  x[too.long] <- x[too.long] %>%
    stringi::stri_sub(1L, width - 3L) %>%
    stringi::stri_trim_right() %>%
    paste0("...")
  x
}
