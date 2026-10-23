#' Escape text for an output format
#'
#' @param x A character vector.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @return `x` with the special characters of `format` escaped.
#' @keywords internal
escape.text = function(x, format = c("markdown", "html", "latex", "plain")) {
  format = match.arg(format)
  x = as.character(x)

  switch(format,
    markdown = {
      x = gsub("([\\\\`*_|#])", "\\\\\\1", x)
      x
    },
    html = {
      x = gsub("&", "&amp;", x, fixed = TRUE)
      x = gsub("<", "&lt;", x, fixed = TRUE)
      x = gsub(">", "&gt;", x, fixed = TRUE)
      x
    },
    latex = {
      x = gsub("\\\\", "\\\\textbackslash{}", x)
      x = gsub("([&%$#_{}])", "\\\\\\1", x)
      x
    },
    plain = {
      x = gsub("[[:cntrl:]]", " ", x)
      x
    }
  )
}

#' Render a heading
#'
#' @param text Heading text.
#' @param level Heading level, from 1 to 4.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @return A character vector of output lines.
#' @keywords internal
render.heading = function(text, level = 2L,
                          format = c("markdown", "html", "latex", "plain")) {
  format = match.arg(format)
  if (!level %in% 1:4) {
    stop("`level` must be between 1 and 4.", call. = FALSE)
  }
  text = escape.text(text, format)

  switch(format,
    markdown = {
      paste(strrep("#", level), text)
    },
    html = {
      sprintf("<h%d>%s</h%d>", level, text, level)
    },
    latex = {
      commands = c("section", "subsection", "subsubsection", "paragraph")
      sprintf("\\%s{%s}", commands[level], text)
    },
    plain = {
      underline = c("=", "-", "~", ".")[level]
      c(text, strrep(underline, nchar(text)))
    }
  )
}

#' Render named items as a list
#'
#' @param items A named character vector.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @return A character vector of output lines, empty when `items` is empty.
#' @keywords internal
render.items = function(items, format = c("markdown", "html", "latex", "plain")) {
  format = match.arg(format)
  if (length(items) == 0L) {
    return(character())
  }
  keys = escape.text(names(items), format)
  values = escape.text(unname(items), format)

  switch(format,
    markdown = {
      sprintf("- **%s**: %s", keys, values)
    },
    html = {
      c("<ul>", sprintf("  <li><b>%s</b>: %s</li>", keys, values), "</ul>")
    },
    latex = {
      c("\\begin{itemize}", sprintf("  \\item \\textbf{%s}: %s", keys, values),
        "\\end{itemize}")
    },
    plain = {
      sprintf("  %s: %s", formatC(keys, width = -max(nchar(keys))), values)
    }
  )
}

#' Render a data frame as a table
#'
#' @param df A data frame.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @return A character vector of output lines.
#' @keywords internal
render.table = function(df, format = c("markdown", "html", "latex", "plain")) {
  format = match.arg(format)
  # Format every cell as text before laying out the rows
  header = escape.text(names(df), format)
  cells = vapply(df, function(col) escape.text(format(col), format),
                 character(nrow(df)))
  cells = matrix(cells, nrow = nrow(df))

  switch(format,
    markdown = {
      rows = apply(cells, 1L, paste, collapse = " | ")
      c(paste("|", paste(header, collapse = " | "), "|"),
        paste0("|", strrep("---|", length(header))),
        paste("|", rows, "|"))
    },
    html = {
      rows = apply(cells, 1L, function(r) paste0("<td>", r, "</td>", collapse = ""))
      c("<table>",
        paste0("<tr>", paste0("<th>", header, "</th>", collapse = ""), "</tr>"),
        paste0("<tr>", rows, "</tr>"),
        "</table>")
    },
    latex = {
      rows = apply(cells, 1L, paste, collapse = " & ")
      c(sprintf("\\begin{tabular}{%s}", strrep("l", length(header))),
        paste(paste(header, collapse = " & "), "\\\\"),
        paste(rows, "\\\\"),
        "\\end{tabular}")
    },
    plain = {
      rows = apply(cells, 1L, paste, collapse = "  ")
      c(paste(header, collapse = "  "),
        rows)
    }
  )
}

#' Render one section
#'
#' @param section A `mill_section` object.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @param level Heading level for the section title.
#' @return A character vector of output lines; empty sections render as a
#'   heading followed by `"No data."`.
#' @export
render.section = function(section, format = "markdown", level = 3L) {
  heading = render.heading(section$name, level, format)
  if (isTRUE(section$empty)) {
    return(c(heading, escape.text("No data.", format)))
  }
  c(heading, render.items(section$items, format))
}

#' Render a report from grouped sections
#'
#' @param sections A named list of groups, each a list of `mill_section`
#'   objects.
#' @param title Report title.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @param summary Optional data frame rendered as a table after the title.
#' @param toc Add a contents list with the number of sections per group?
#' @return A single string with the rendered report, lines joined by
#'   newlines.
#' @export
render.report = function(sections, title, format = "markdown", summary = NULL,
                         toc = FALSE) {
  out = render.heading(title, 1L, format)
  if (toc) {
    entries = stats::setNames(as.character(lengths(sections)), names(sections))
    out = c(out, "", render.heading("Contents", 2L, format), render.items(entries, format))
  }
  if (!is.null(summary)) {
    out = c(out, "", render.table(summary, format))
  }

  # One heading per group, then its sections
  for (group in names(sections)) {
    out = c(out, "", render.heading(group, 2L, format))
    for (section in sections[[group]]) {
      out = c(out, "", render.section(section, format, 3L))
    }
  }
  paste(out, collapse = "\n")
}
