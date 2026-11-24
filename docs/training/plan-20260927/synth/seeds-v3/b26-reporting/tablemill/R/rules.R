#' Normalise column alignments
#'
#' @description
#' Accepts full words (`"left"`, `"right"`, `"centre"` or `"center"`) or a
#' compact string such as `"lrc"`, and recycles the result to `n` columns.
#'
#' @param align A character vector of alignments or a compact string.
#' @param n Number of columns.
#' @return A character vector of length `n`.
#' @noRd
normaliseAlign = function(align, n) {
  words = c(l = "left", r = "right", c = "centre")
  if (length(align) == 1L && grepl("^[lrc]+$", align)) {
    align = words[strsplit(align, "")[[1]]]
  }

  align = unname(align)
  align[align == "center"] = "centre"
  bad = !align %in% words
  if (any(bad)) {
    stop(sprintf("Unknown alignment: %s.", paste(unique(align[bad]), collapse = ", ")),
         call. = FALSE)
  }

  if (n %% length(align) != 0L) {
    stop("`align` cannot be recycled to `n` columns.", call. = FALSE)
  }

  rep_len(align, n)
}

#' Characters used to draw a text box
#'
#' @description
#' All styles use ASCII characters only.
#'
#' @param style One of `"ascii"`, `"rounded"`, `"double"`, `"minimal"` or
#'   `"heavy"`.
#' @return A list with elements `horizontal`, `vertical`, `top`, `middle`
#'   and `bottom`; the last three are the joints of the matching rules.
#' @export
borderChars = function(style = c("ascii", "rounded", "double", "minimal", "heavy")) {
  style = match.arg(style)
  switch(style,
    ascii = list(
      horizontal = "-",
      vertical = "|",
      top = "+",
      middle = "+",
      bottom = "+"
    ),
    rounded = list(
      horizontal = "-",
      vertical = "|",
      top = ".",
      middle = "+",
      bottom = "'"
    ),
    double = list(
      horizontal = "=",
      vertical = "|",
      top = "+",
      middle = "+",
      bottom = "+"
    ),
    minimal = list(
      horizontal = "-",
      vertical = " ",
      top = " ",
      middle = " ",
      bottom = " "
    ),
    heavy = list(
      horizontal = "#",
      vertical = "#",
      top = "#",
      middle = "#",
      bottom = "#"
    )
  )
}

#' Horizontal rule of a text box
#'
#' @param widths An integer vector of column widths.
#' @param position One of `"top"`, `"middle"` or `"bottom"`.
#' @param style A style accepted by [borderChars()].
#' @param padding Number of spaces on each side of a cell.
#' @return A single string.
#' @export
boxRule = function(widths, position = c("top", "middle", "bottom"), style = "ascii",
                   padding = 1L) {
  position = match.arg(position)
  chars = borderChars(style)
  joint = chars[[position]]
  segments = strrep(chars$horizontal, widths + 2L * padding)
  paste0(joint, paste(segments, collapse = joint), joint)
}

#' One row of a text box
#'
#' @inheritParams boxRule
#' @param cells A character vector with one entry per column.
#' @param align A character vector of alignments, one per column.
#' @return A single string.
#' @export
boxRow = function(cells, widths, align = "left", style = "ascii", padding = 1L) {
  chars = borderChars(style)
  align = normaliseAlign(align, length(cells))
  pad = strrep(" ", padding)
  body = ifelse(
    align == "right",
    sprintf("%*s", widths, cells),
    sprintf("%-*s", widths, cells)
  )
  paste0(chars$vertical, paste0(pad, body, pad, collapse = chars$vertical), chars$vertical)
}

#' Draw a complete text box table
#'
#' @description
#' Optional title row, top rule, header row, middle rule, one row per line
#' of `cells`, bottom rule.
#'
#' @param cells A character matrix of formatted cells, or a data frame that
#'   is converted with [as.matrix()].
#' @param header A character vector of column headers.
#' @param align A character vector of alignments, one per column.
#' @param style A style accepted by [borderChars()].
#' @param title An optional single string centred above the table.
#' @return A character vector with one element per output line.
#' @export
boxTable = function(cells, header, align = "left", style = "ascii", title = NULL) {
  if (is.data.frame(cells)) {
    cells = as.matrix(cells)
  }
  if (ncol(cells) != length(header)) {
    stop("`cells` must have one column per `header` entry.", call. = FALSE)
  }
  if (!is.null(title) && (!is.character(title) || length(title) != 1L)) {
    stop("`title` must be NULL or a single string.", call. = FALSE)
  }

  widths = nchar(header)
  for (j in seq_along(header)) {
    widths[j] = max(widths[j], nchar(cells[, j]))
  }

  align = normaliseAlign(align, length(header))
  lines = c(
    boxRule(widths, "top", style),
    boxRow(header, widths, align, style),
    boxRule(widths, "middle", style)
  )
  for (i in seq_len(nrow(cells))) {
    lines = c(lines, boxRow(cells[i, ], widths, align, style))
  }
  lines = c(lines, boxRule(widths, "bottom", style))

  if (!is.null(title)) {
    total = nchar(lines[1L])
    gap = max(total - nchar(title), 0L)
    before = gap %/% 2L
    lines = c(paste0(strrep(" ", before), title, strrep(" ", gap - before)), lines)
  }
  lines
}

#' Delimiter row of a Markdown table
#'
#' @description
#' Colons mark the alignment: on the left for `"left"`, on the right for
#' `"right"` and on both sides for `"centre"`.
#'
#' @param widths An integer vector of column widths.
#' @param align A character vector of alignments or a compact string.
#' @return A single string.
#' @export
markdownRule = function(widths, align = "left") {
  align = normaliseAlign(align, length(widths))
  rule = strrep("-", pmax(widths, 3L))
  left = align %in% c("left", "centre")
  right = align %in% c("right", "centre")
  rule[left] = paste0(":", substring(rule[left], 2L))
  rule[right] = paste0(substring(rule[right], 1L, nchar(rule[right]) - 1L), ":")
  paste0("| ", paste(rule, collapse = " | "), " |")
}

#' Column specification of a LaTeX tabular
#'
#' @param align A character vector of alignments or a compact string.
#' @param n Number of columns; by default one per alignment.
#' @param vertical Whether to draw vertical lines between columns.
#' @return A single string such as `"lrr"` or `"|l|r|r|"`.
#' @export
latexColSpec = function(align, n = NULL, vertical = FALSE) {
  if (is.null(n)) {
    compact = length(align) == 1L && grepl("^[lrc]+$", align)
    n = if (compact) nchar(align) else length(align)
  }
  letters = substr(normaliseAlign(align, n), 1L, 1L)
  if (vertical) {
    paste0("|", paste(letters, collapse = "|"), "|")
  } else {
    paste(letters, collapse = "")
  }
}

#' Horizontal rule of a LaTeX tabular
#'
#' @param position One of `"top"`, `"mid"` or `"bottom"`.
#' @param booktabs Whether to use the booktabs rules instead of `\hline`.
#' @return A single string.
#' @export
latexRule = function(position = c("top", "mid", "bottom"), booktabs = TRUE) {
  position = match.arg(position)
  if (booktabs) {
    paste0("\\", position, "rule")
  } else {
    "\\hline"
  }
}
