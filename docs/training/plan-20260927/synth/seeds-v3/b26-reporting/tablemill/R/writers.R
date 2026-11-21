#' Create a table object
#'
#' @description
#' Formats every cell to text once, so the writers below only have to lay
#' the cells out. Cells are written verbatim; escape them beforehand.
#'
#' @param data A data frame to tabulate.
#' @param caption An optional single string used as the table caption.
#' @param align A character vector of `"left"`, `"right"` or `"centre"`, one
#'   per column; numeric columns default to `"right"`.
#' @param digits Number of decimal places used for numeric columns.
#' @return An object of class `tm_table` with elements `cells`, `header`,
#'   `align` and `caption`.
#' @export
newTable = function(data, caption = NULL, align = NULL, digits = 2) {
  # Validate `data`
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  # Validate `caption`
  if (!is.null(caption) && (!is.character(caption) || length(caption) != 1L)) {
    stop("`caption` must be NULL or a single string.", call. = FALSE)
  }
  # Validate `align`
  if (is.null(align)) {
    align = ifelse(vapply(data, is.numeric, logical(1)), "right", "left")
  }

  if (length(align) != ncol(data) || !all(align %in% c("left", "right", "centre"))) {
    stop("`align` must give left, right or centre for every column.", call. = FALSE)
  }
  # Validate `digits`
  if (!is.numeric(digits) || length(digits) != 1L || digits < 0) {
    stop("`digits` must be a single non-negative number.", call. = FALSE)
  }

  cells = matrix("", nrow = nrow(data), ncol = ncol(data))
  for (j in seq_along(data)) {
    column = data[[j]]
    if (is.double(column)) {
      text = formatC(column, format = "f", digits = digits)
    } else {
      text = as.character(column)
    }
    cells[, j] = ifelse(is.na(column), "", text)
  }

  structure(
    list(cells = cells, header = names(data), align = unname(align), caption = caption),
    class = "tm_table"
  )
}

#' Write one table row
#'
#' @description
#' Low-level helpers shared by the table writers.
#'
#' @param cells A character vector with one entry per column.
#' @param tag The HTML cell tag, `"td"` or `"th"`.
#' @param align A character vector of alignments, one per column.
#' @return A single string.
#' @rdname tableRow
#' @noRd
markdownRow = function(cells) {
  paste0("| ", paste(cells, collapse = " | "), " |")
}

#' @rdname tableRow
#' @noRd
htmlRow = function(cells, tag = "td", align = rep("left", length(cells))) {
  style = sprintf(" style=\"text-align: %s\"", ifelse(align == "centre", "center", align))
  inner = paste0("<", tag, style, ">", cells, "</", tag, ">", collapse = "")
  paste0("    <tr>", inner, "</tr>")
}

#' @rdname tableRow
#' @noRd
latexRow = function(cells) {
  paste0(paste(cells, collapse = " & "), " \\\\")
}

#' Render a table in one output format
#'
#' @description
#' Each writer returns the lines of the table; [writeTable()] joins them and
#' writes them to disk.
#'
#' @param tbl A `tm_table` object created by [newTable()].
#' @return A character vector with one element per output line.
#' @rdname tableWriters
#' @export
asMarkdown = function(tbl) {
  stopifnot(inherits(tbl, "tm_table"))
  tokens = c(left = ":---", right = "---:", centre = ":---:")
  lines = c(markdownRow(tbl$header), markdownRow(tokens[tbl$align]))
  for (i in seq_len(nrow(tbl$cells))) {
    lines = c(lines, markdownRow(tbl$cells[i, ]))
  }
  if (!is.null(tbl$caption)) {
    lines = c(lines, "", paste0("Table: ", tbl$caption))
  }
  lines
}

#' @rdname tableWriters
#' @export
asHtml = function(tbl) {
  stopifnot(inherits(tbl, "tm_table"))
  lines = "<table>"
  if (!is.null(tbl$caption)) {
    lines = c(lines, paste0("  <caption>", tbl$caption, "</caption>"))
  }
  lines = c(lines, "  <thead>", htmlRow(tbl$header, "th", tbl$align), "  </thead>")
  lines = c(lines, "  <tbody>")
  for (i in seq_len(nrow(tbl$cells))) {
    lines = c(lines, htmlRow(tbl$cells[i, ], "td", tbl$align))
  }
  c(lines, "  </tbody>", "</table>")
}

#' @rdname tableWriters
#' @export
asLatex = function(tbl) {
  stopifnot(inherits(tbl, "tm_table"))
  spec = paste(substr(tbl$align, 1L, 1L), collapse = "")
  lines = c("\\begin{table}", "\\centering")
  if (!is.null(tbl$caption)) {
    lines = c(lines, paste0("\\caption{", tbl$caption, "}"))
  }
  lines = c(lines, paste0("\\begin{tabular}{", spec, "}"), "\\hline", latexRow(tbl$header))
  lines = c(lines, "\\hline")
  for (i in seq_len(nrow(tbl$cells))) {
    lines = c(lines, latexRow(tbl$cells[i, ]))
  }
  c(lines, "\\hline", "\\end{tabular}", "\\end{table}")
}

#' Write a table to a file
#'
#' @description
#' With `format = "auto"` the format is taken from the file extension:
#' `.md`, `.html` or `.tex`.
#'
#' @param tbl A `tm_table` object created by [newTable()].
#' @param path Path of the file to write.
#' @param format One of `"auto"`, `"markdown"`, `"html"` or `"latex"`.
#' @return `path`, invisibly.
#' @export
writeTable = function(tbl, path, format = c("auto", "markdown", "html", "latex")) {
  format = match.arg(format)
  if (format == "auto") {
    ext = tolower(tools::file_ext(path))
    format = switch(ext,
      md = "markdown",
      html = "html",
      tex = "latex",
      stop(sprintf("Cannot infer the format of `%s`.", path), call. = FALSE)
    )
  }

  lines = switch(format,
    markdown = asMarkdown(tbl),
    html = asHtml(tbl),
    latex = asLatex(tbl)
  )
  writeLines(lines, path)
  invisible(path)
}
