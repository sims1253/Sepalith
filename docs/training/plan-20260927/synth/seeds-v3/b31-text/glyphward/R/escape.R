#' @title Build a regex alternation of literal strings
#' @description Longer strings come first so that they win over their
#'   prefixes.
#' @param keys A character vector of literal strings.
#' @return A single PCRE pattern.
#' @keywords internal
literal.alternation <- function(keys) {
  if (length(keys) == 0L) {
    stop("`keys` must not be empty.", call. = FALSE)
  }
  keys <- unique(keys[order(-nchar(keys))])
  quoted <- gsub("([][{}()+*^$|\\\\?./])", "\\\\\\1", keys)
  paste(quoted, collapse = "|")
}

#' @title Replace matches through a lookup table
#' @description Every match of `pattern` is replaced in one pass, so
#'   replacements are never matched again.
#' @param x A character vector.
#' @param pattern A PCRE pattern whose matches are all names of `table`.
#' @param table A named character vector mapping matches to replacements.
#' @return A character vector the same length as `x`; `NA` stays `NA`.
#' @keywords internal
map.matches <- function(x, pattern, table) {
  ok <- !is.na(x)
  hits <- gregexpr(pattern, x[ok], perl = TRUE)
  found <- regmatches(x[ok], hits)
  replaced <- x[ok]
  regmatches(replaced, hits) <- lapply(found, function(m) unname(table[m]))
  x[ok] <- replaced
  x
}

#' @title Escape tables
#' @description The lookup table behind [escape.text()] and
#'   [unescape.text()] for one format.
#' @param format The target format.
#' @return A named character vector mapping raw characters to their escaped
#'   form.
#' @keywords internal
escape.table <- function(format = c("html", "latex", "json", "markdown")) {
  format <- match.arg(format)
  switch(format,
    html = c(
      "&" = "&amp;", "<" = "&lt;", ">" = "&gt;", "\"" = "&quot;", "'" = "&#39;"
    ),
    latex = c(
      "\\" = "\\textbackslash{}", "{" = "\\{", "}" = "\\}", "$" = "\\$",
      "&" = "\\&", "#" = "\\#", "%" = "\\%", "_" = "\\_",
      "^" = "\\textasciicircum{}", "~" = "\\textasciitilde{}"
    ),
    json = c(
      "\\" = "\\\\", "\"" = "\\\"", "\n" = "\\n", "\r" = "\\r", "\t" = "\\t",
      "/" = "\\/"
    ),
    markdown = c(
      "\\" = "\\\\", "*" = "\\*", "_" = "\\_", "`" = "\\`", "#" = "\\#",
      "[" = "\\[", "]" = "\\]", "|" = "\\|"
    )
  )
}

#' @title Escape special characters
#' @description Replaces the characters that are special in `format` by
#'   their escaped form. [unescape.text()] is the inverse.
#' @param x A character vector.
#' @param format The target format.
#' @param minimal Leave out the optional entries: quotes for `"html"`, `~`
#'   and `^` for `"latex"`, the slash for `"json"` and the pipe for
#'   `"markdown"`.
#' @return A character vector the same length as `x`.
#' @export
escape.text <- function(x, format = c("html", "latex", "json", "markdown"),
                        minimal = FALSE) {
  format <- match.arg(format)
  table <- escape.table(format)
  if (minimal) {
    optional <- switch(format,
      html = c("\"", "'"),
      latex = c("~", "^"),
      json = c("/"),
      markdown = c("|")
    )
    table <- table[!names(table) %in% optional]
  }

  pattern <- literal.alternation(names(table))
  out <- map.matches(x, pattern, table)
  names(out) <- names(x)
  out
}

#' @title Unescape special characters
#' @description Replaces escaped sequences of `format` by the characters they
#'   stand for. This is the inverse of [escape.text()].
#' @param x A character vector.
#' @param format The source format.
#' @param minimal Leave out the optional entries: quotes for `"html"`, `~`
#'   and `^` for `"latex"`, the slash for `"json"` and the pipe for
#'   `"markdown"`.
#' @return A character vector the same length as `x`.
#' @export
unescape.text <- function(x, format = c("html", "latex", "json", "markdown"),
                          minimal = FALSE) {
  format <- match.arg(format)
  table <- escape.table(format)
  if (minimal) {
    optional <- switch(format,
      html = c("\"", "'"),
      latex = c("~", "^"),
      json = c("/"),
      markdown = c("|")
    )
    table <- table[!names(table) %in% optional]
  }

  pattern <- literal.alternation(unname(table))
  out <- map.matches(x, pattern, stats::setNames(names(table), table))
  names(out) <- names(x)
  out
}

#' @title Check that escaping round-trips
#' @param x A character vector of sample texts.
#' @param formats The formats to check.
#' @return A data frame with one row per format and columns `format`,
#'   `n_changed` (elements altered by escaping) and `roundtrip` (`TRUE` when
#'   unescaping gives back `x`).
#' @keywords internal
escape.roundtrip <- function(x, formats = c("html", "latex", "json", "markdown")) {
  formats <- match.arg(formats, several.ok = TRUE)
  n.changed <- integer(length(formats))
  roundtrip <- logical(length(formats))
  for (i in seq_along(formats)) {
    escaped <- escape.text(x, formats[i])
    restored <- unescape.text(escaped, formats[i])
    n.changed[i] <- sum(escaped != x, na.rm = TRUE)
    roundtrip[i] <- identical(restored, x)
  }
  data.frame(format = formats, n_changed = n.changed, roundtrip = roundtrip)
}
