#' Escape text for Markdown tables
#'
#' @description
#' Backslash-escapes the characters that would otherwise end a cell or start
#' inline formatting.
#'
#' @param x A character vector of cell contents.
#' @return A character vector the same length as `x`.
#' @family escapers
#' @export
escapeMarkdown = function(x) {
  x = as.character(x)
  x = gsub("\\", "\\\\", x, fixed = TRUE)
  x = gsub("|", "\\|", x, fixed = TRUE)
  x = gsub("*", "\\*", x, fixed = TRUE)
  x = gsub("_", "\\_", x, fixed = TRUE)
  x = gsub("`", "\\`", x, fixed = TRUE)
  x
}

#' Undo Markdown escaping
#'
#' @description
#' The inverse of [escapeMarkdown()]: drops the backslash in front of every
#' escaped character.
#'
#' @param x A character vector of cell contents.
#' @return A character vector the same length as `x`.
#' @family escapers
#' @export
unescapeMarkdown = function(x) {
  x = as.character(x)
  gsub("\\\\([\\\\|*_`])", "\\1", x)
}

#' Escape text for HTML tables
#'
#' @description
#' Replaces the five characters with special meaning in HTML by their
#' entities. The ampersand goes first so entities are not escaped twice.
#'
#' @param x A character vector of cell contents.
#' @return A character vector the same length as `x`.
#' @family escapers
#' @export
escapeHtml = function(x) {
  x = as.character(x)
  x = gsub("&", "&amp;", x, fixed = TRUE)
  x = gsub("<", "&lt;", x, fixed = TRUE)
  x = gsub(">", "&gt;", x, fixed = TRUE)
  x = gsub("\"", "&quot;", x, fixed = TRUE)
  x = gsub("'", "&#39;", x, fixed = TRUE)
  x
}

#' Undo HTML escaping
#'
#' @description
#' The inverse of [escapeHtml()]: entities are replaced in the reverse order,
#' so the ampersand comes last.
#'
#' @param x A character vector of cell contents.
#' @return A character vector the same length as `x`.
#' @family escapers
#' @export
unescapeHtml = function(x) {
  x = as.character(x)
  x = gsub("&#39;", "'", x, fixed = TRUE)
  x = gsub("&quot;", "\"", x, fixed = TRUE)
  x = gsub("&gt;", ">", x, fixed = TRUE)
  x = gsub("&lt;", "<", x, fixed = TRUE)
  x = gsub("&amp;", "&", x, fixed = TRUE)
  x
}

#' Escape text for LaTeX tables
#'
#' @description
#' Backslashes are swapped for a marker first, so the braces of
#' `\textbackslash{}` are not escaped by the later steps.
#'
#' @param x A character vector of cell contents.
#' @return A character vector the same length as `x`.
#' @family escapers
#' @export
escapeLatex = function(x) {
  x = as.character(x)
  x = gsub("\\", "\001", x, fixed = TRUE)
  x = gsub("([&%$#_{}])", "\\\\\\1", x)
  x = gsub("~", "\\textasciitilde{}", x, fixed = TRUE)
  x = gsub("^", "\\textasciicircum{}", x, fixed = TRUE)
  x = gsub("\001", "\\textbackslash{}", x, fixed = TRUE)
  x
}

#' Escape cells for a target format
#'
#' @description
#' Applies the escaper for `format` and turns embedded newlines into the
#' line break of that format.
#'
#' @param x A character vector of cell contents.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @return A character vector the same length as `x`; missing values stay
#'   missing.
#' @export
escapeCells = function(x, format = c("markdown", "html", "latex", "plain")) {
  format = match.arg(format)
  missing = is.na(x)
  out = switch(format,
    markdown = {
      x = escapeMarkdown(x)
      gsub("\n", "<br>", x, fixed = TRUE)
    },
    html = {
      x = escapeHtml(x)
      gsub("\n", "<br>", x, fixed = TRUE)
    },
    latex = {
      x = escapeLatex(x)
      gsub("\n", "\\newline ", x, fixed = TRUE)
    },
    plain = {
      x = as.character(x)
      gsub("\n", " ", x, fixed = TRUE)
    }
  )
  out[missing] = NA_character_
  out
}

#' Escape the text columns and names of a data frame
#'
#' @description
#' Character and factor columns are escaped with [escapeCells()]; other
#' columns are left untouched so they can still be formatted as numbers.
#'
#' @param data A data frame whose text should be escaped.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"plain"`.
#' @param names Whether to escape the column names as well.
#' @return A data frame with the same shape as `data`.
#' @export
escapeTable = function(data, format = c("markdown", "html", "latex", "plain"),
                       names = TRUE) {
  format = match.arg(format)
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }

  for (j in seq_along(data)) {
    column = data[[j]]
    if (is.character(column) || is.factor(column)) {
      data[[j]] = escapeCells(as.character(column), format)
    }
  }

  if (names) {
    base::names(data) = escapeCells(base::names(data), format)
  }

  data
}

#' Remove markup from formatted text
#'
#' @description
#' Drops links, emphasis and code markers, then undoes the escaping of the
#' format, so the result can be measured or used as plain text.
#'
#' @param x A character vector of cell contents.
#' @param format One of `"markdown"`, `"html"` or `"latex"`.
#' @return A character vector the same length as `x`.
#' @export
stripMarkup = function(x, format = c("markdown", "html", "latex")) {
  format = match.arg(format)
  x = as.character(x)
  switch(format,
    markdown = {
      x = gsub("\\[([^]]*)\\]\\([^)]*\\)", "\\1", x)
      x = gsub("\\*\\*([^*]+)\\*\\*", "\\1", x)
      x = gsub("(?<![\\\\*])\\*([^*]+)\\*", "\\1", x, perl = TRUE)
      x = gsub("(?<!\\\\)`([^`]*)`", "\\1", x, perl = TRUE)
      x = gsub("<br>", "\n", x, fixed = TRUE)
      unescapeMarkdown(x)
    },
    html = {
      x = gsub("<br\\s*/?>", "\n", x, perl = TRUE)
      x = gsub("<[^>]+>", "", x)
      unescapeHtml(x)
    },
    latex = {
      x = gsub("\\\\newline\\s*", "\n", x, perl = TRUE)
      x = gsub("\\\\(textbf|textit|emph|texttt)\\{([^}]*)\\}", "\\2", x)
      x = gsub("\\textasciitilde{}", "~", x, fixed = TRUE)
      x = gsub("\\textasciicircum{}", "^", x, fixed = TRUE)
      x = gsub("\\\\([&%$#_{}])", "\\1", x)
      gsub("\\textbackslash{}", "\\", x, fixed = TRUE)
    }
  )
}
