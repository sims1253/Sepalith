#' Replace characters using a lookup table
#'
#' Applies fixed-string replacements in the order given, which matters when
#' one replacement introduces characters another one would touch.
#'
#' @param x A character vector to escape.
#' @param table A named character vector mapping characters to replacements.
#' @noRd
replace.by.table <- function(x, table) {
  out <- x
  for (key in names(table)) {
    out <- gsub(key, table[[key]], out, fixed = TRUE)
  }
  out
}

#' Escape text for HTML
#'
#' `escape.html()` escapes text for element content. `escape.attr()` also
#' escapes both kinds of quote so the value can sit inside either quoting
#' style.
#'
#' @param x A character vector to escape.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @rdname escape.html
#' @export
escape.html <- function(x, na = "") {
  x <- as.character(x)
  # "&" must go first or the other entities would be escaped twice
  out <- replace.by.table(x, c(
    "&" = "&amp;",
    "<" = "&lt;",
    ">" = "&gt;"
  ))
  out[is.na(x)] <- na
  out
}

#' @rdname escape.html
#' @export
escape.attr <- function(x, na = "") {
  x <- as.character(x)
  out <- replace.by.table(x, c(
    "&" = "&amp;",
    "<" = "&lt;",
    ">" = "&gt;",
    "\"" = "&quot;",
    "'" = "&#39;"
  ))
  out[is.na(x)] <- na
  out
}

#' Escape text for inline Markdown
#'
#' Backslash-escapes characters that would otherwise start emphasis, code,
#' links or table cells.
#'
#' @param x A character vector to escape.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @export
escape.markdown <- function(x, na = "") {
  x <- as.character(x)
  out <- gsub("([\\\\`*_{}\\[\\]()#+!|<>~])", "\\\\\\1", x, perl = TRUE)

  # line-start markers that would turn the text into a list item; numbers
  # such as "1.50" or "-3" are not list markers and stay as they are
  out <- gsub("^([0-9]+)\\.(\\s|$)", "\\1\\\\.\\2", out, perl = TRUE)
  out <- gsub("^-(\\s|$)", "\\\\-\\1", out, perl = TRUE)

  out[is.na(x)] <- na
  out
}

#' Escape text for LaTeX
#'
#' @param x A character vector to escape.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @export
escape.latex <- function(x, na = "") {
  x <- as.character(x)
  # backslash first, via a placeholder, so later braces are not re-escaped
  out <- gsub("\\", "\001", x, fixed = TRUE)
  out <- replace.by.table(out, c(
    "{" = "\\{",
    "}" = "\\}",
    "$" = "\\$",
    "&" = "\\&",
    "%" = "\\%",
    "#" = "\\#",
    "_" = "\\_",
    "~" = "\\textasciitilde{}",
    "^" = "\\textasciicircum{}"
  ))
  out <- gsub("\001", "\\textbackslash{}", out, fixed = TRUE)

  out[is.na(x)] <- na
  out
}

#' Escape text for a given output format
#'
#' @param x A character vector to escape.
#' @param format Target output format.
#' @param na How to render missing values.
#' @return A character vector the same length as `x`.
#' @export
escape.for <- function(x, format = c("html", "attr", "markdown", "latex"), na = "") {
  format <- match.arg(format)
  switch(format,
    html = escape.html(x, na = na),
    attr = escape.attr(x, na = na),
    markdown = escape.markdown(x, na = na),
    latex = escape.latex(x, na = na)
  )
}

#' Decode common HTML entities
#'
#' Handles the named entities produced by [escape.attr()] plus decimal and
#' hexadecimal numeric references.
#'
#' @param x A character vector of HTML-escaped text.
#' @return A character vector the same length as `x`.
#' @export
unescape.html <- function(x) {
  # "&amp;" is protected first and decoded last, so "&amp;lt;" stays "&lt;"
  # and a decoded "&#38;" cannot start a second round of decoding
  out <- stringi::stri_replace_all_fixed(x, "&amp;", "\001")
  out <- out %>%
    stringi::stri_replace_all_fixed(
      c("&lt;", "&gt;", "&quot;", "&#39;", "&nbsp;"),
      c("<", ">", "\"", "'", " "),
      vectorize_all = FALSE
    )

  refs <- stringi::stri_extract_all_regex(out, "&#(x[0-9A-Fa-f]+|[0-9]+);")
  for (i in seq_along(out)) {
    codes <- unique(refs[[i]])
    codes <- codes[!is.na(codes)]
    for (ref in codes) {
      body <- stringi::stri_sub(ref, 3, -2)
      num <- if (startsWith(body, "x")) strtoi(substring(body, 2), 16L) else as.integer(body)
      out[i] <- stringi::stri_replace_all_fixed(out[i], ref, intToUtf8(num))
    }
  }

  stringi::stri_replace_all_fixed(out, "\001", "&")
}
