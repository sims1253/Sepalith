#' Format amounts as currency
#'
#' @param x A numeric vector.
#' @param symbol Currency symbol placed before the amount.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `x`.
#' @family formatters
#' @export
formatMoney <- function(x, symbol = "$", digits = 2) {
  out <- paste0(symbol, formatC(abs(x), format = "f", digits = digits, big.mark = ","))
  out[x < 0 & !is.na(x)] <- paste0("-", out[x < 0 & !is.na(x)])
  out[is.na(x)] <- "NA"
  out
}

#' Format proportions as percentages
#'
#' @param x A numeric vector of proportions.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `x`.
#' @family formatters
#' @export
formatPercent <- function(x, digits = 1) {
  out <- sprintf(paste0("%.", digits, "f%%"), 100 * x)
  out[is.na(x)] <- "NA"
  out
}

#' Format whole-number counts
#'
#' @param x A numeric vector.
#' @param digits Ignored for counts; kept for a common formatter signature.
#' @return A character vector the same length as `x`.
#' @family formatters
#' @export
formatCount <- function(x, digits = 0) {
  out <- formatC(round(x), format = "d", big.mark = ",")
  out[is.na(x)] <- "NA"
  out
}

#' Pad strings to a fixed width
#'
#' @param x A character vector.
#' @param width Target width in characters.
#' @param align One of "left", "right" or "centre".
#' @return A character vector the same length as `x`.
#' @export
padText <- function(x, width, align = c("left", "right", "centre")) {
  align <- match.arg(align)
  x <- as.character(x)
  gap <- pmax(0, width - nchar(x))
  switch(align,
    left = paste0(x, strrep(" ", gap)),
    right = paste0(strrep(" ", gap), x),
    centre = paste0(strrep(" ", gap %/% 2), x, strrep(" ", gap - gap %/% 2))
  )
}

#' Build a report heading
#'
#' @param title Heading text.
#' @param width Total width of the rule lines.
#' @param char Character used for the rule lines.
#' @return A character vector of three lines.
#' @export
reportHeading <- function(title, width = 60, char = "=") {
  rule <- strrep(char, width)
  c(rule, padText(toupper(title), width, "centre"), rule)
}

#' Build a label and value line
#'
#' Dots fill the space between the label and the right-aligned value.
#'
#' @param label Text on the left.
#' @param value Already formatted text on the right.
#' @param width Total width of the line.
#' @return A single string.
#' @export
reportLine <- function(label, value, width = 60) {
  dots <- max(1, width - nchar(label) - nchar(value) - 2)
  sprintf("%s %s %s", label, strrep(".", dots), value)
}

#' Render a data frame as fixed-width text
#'
#' Numeric columns are right-aligned, others left-aligned. Column widths
#' fit the widest entry including the header.
#'
#' @param data A data frame to render.
#' @param formatters Optional named list of functions used to format
#'   individual columns.
#' @param sep String placed between columns.
#' @return A character vector with a header line, a rule and one line per
#'   row.
#' @export
reportTable <- function(data, formatters = list(), sep = "  ") {
  cells <- lapply(names(data), function(nm) {
    f <- formatters[[nm]]
    if (is.null(f)) format(data[[nm]], trim = TRUE) else f(data[[nm]])
  })
  names(cells) <- names(data)
  widths <- mapply(function(v, nm) max(nchar(c(v, nm))), cells, names(cells))
  numeric <- vapply(data, is.numeric, logical(1))
  cols <- Map(function(v, nm, w, num) {
    a <- if (num) "right" else "left"
    c(padText(nm, w, a), padText(v, w, a))
  }, cells, names(cells), widths, numeric)
  lines <- do.call(paste, c(cols, sep = sep))
  c(lines[1], strrep("-", nchar(lines[1])), lines[-1])
}

#' Write report lines to a file or the console
#'
#' @param lines A character vector of report lines.
#' @param file Path to write to; `""` prints to the console.
#' @param append If `TRUE`, add to an existing file.
#' @return `lines`, invisibly.
#' @export
writeReport <- function(lines, file = "", append = FALSE) {
  if (nzchar(file) && !append && file.exists(file)) {
    message("overwriting ", file)
  }
  cat(lines, file = file, sep = "\n", append = append)
  invisible(lines)
}
