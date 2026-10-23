#' Shorten text to a maximum width
#'
#' @description
#' Text longer than `width` loses characters on the chosen side, which is
#' replaced by `ellipsis`.
#'
#' @param x A character vector.
#' @param width Maximum width in characters, including the ellipsis.
#' @param side One of `"right"`, `"left"` or `"middle"`.
#' @param ellipsis String marking the removed characters.
#' @return A character vector the same length as `x`.
#' @export
truncateText = function(x, width, side = c("right", "left", "middle"), ellipsis = "...") {
  side = match.arg(side)
  x = as.character(x)
  long = !is.na(x) & nchar(x) > width
  keep = width - nchar(ellipsis)
  if (any(long) && keep < 1L) {
    stop("`width` must leave room for at least one character.", call. = FALSE)
  }

  y = x[long]
  x[long] = switch(side,
    right = {
      paste0(substr(y, 1L, keep), ellipsis)
    },
    left = {
      paste0(ellipsis, substr(y, nchar(y) - keep + 1L, nchar(y)))
    },
    middle = {
      head = ceiling(keep / 2)
      tail = keep - head
      paste0(substr(y, 1L, head), ellipsis, substr(y, nchar(y) - tail + 1L, nchar(y)))
    }
  )
  x
}

#' Pad text to a fixed width
#'
#' @param x A character vector.
#' @param width Target width in characters; longer text is left as is.
#' @param align One of `"left"`, `"right"` or `"centre"`.
#' @param fill Single character used for padding.
#' @return A character vector the same length as `x`.
#' @export
padText = function(x, width, align = c("left", "right", "centre"), fill = " ") {
  align = match.arg(align)
  x = as.character(x)
  gap = pmax(width - nchar(x), 0L)
  switch(align,
    left = {
      paste0(x, strrep(fill, gap))
    },
    right = {
      paste0(strrep(fill, gap), x)
    },
    centre = {
      before = gap %/% 2L
      paste0(strrep(fill, before), x, strrep(fill, gap - before))
    }
  )
}

#' Align numbers on their decimal mark
#'
#' @description
#' Pads the whole part on the left and the fraction on the right so the
#' decimal marks of all values line up.
#'
#' @param x A character vector of formatted numbers.
#' @param mark The decimal mark.
#' @return A character vector of equal-width strings.
#' @export
alignDecimal = function(x, mark = ".") {
  x = as.character(x)
  x[is.na(x)] = ""
  parts = strsplit(x, mark, fixed = TRUE)
  whole = vapply(parts, function(p) if (length(p) > 0L) p[1L] else "", character(1))
  fraction = vapply(parts, function(p) if (length(p) > 1L) p[2L] else "", character(1))
  fraction = ifelse(grepl(mark, x, fixed = TRUE), paste0(mark, fraction), "")
  whole = padText(whole, max(nchar(whole)), "right")
  fraction = padText(fraction, max(nchar(fraction)), "left")
  paste0(whole, fraction)
}

#' Wrap cell text onto several lines
#'
#' @param x A character vector of cells.
#' @param width Maximum line width in characters.
#' @return A list with one character vector of lines per cell; empty and
#'   missing cells give a single empty line.
#' @noRd
wrapCell = function(x, width) {
  lapply(as.character(x), function(cell) {
    if (is.na(cell) || !nzchar(cell)) {
      return("")
    }
    truncateText(strwrap(cell, width = width + 1L), width)
  })
}

#' Lay out one table row over several lines
#'
#' @param cells A character vector with one entry per column.
#' @param widths An integer vector of column widths.
#' @param align A character vector of alignments, one per column.
#' @param sep String placed between columns.
#' @return A character vector with one element per output line.
#' @noRd
stackCells = function(cells, widths, align, sep = " | ") {
  columns = vector("list", length(cells))
  for (j in seq_along(cells)) {
    columns[[j]] = wrapCell(cells[j], widths[j])[[1]]
  }
  height = max(lengths(columns))
  for (j in seq_along(columns)) {
    lines = c(columns[[j]], rep("", height - length(columns[[j]])))
    columns[[j]] = padText(lines, widths[j], align[j])
  }
  do.call(paste, c(columns, list(sep = sep)))
}

#' Caption line for a plain text table
#'
#' @param caption A single string.
#' @param width Width of the table in characters.
#' @param position One of `"above"` or `"below"`; the underline sits between
#'   the caption and the table.
#' @param underline Character repeated to draw the underline.
#' @return A character vector of two lines.
#' @export
plainCaption = function(caption, width, position = c("above", "below"), underline = "-") {
  position = match.arg(position)
  text = padText(truncateText(caption, width), width, "centre")
  rule = strrep(underline, width)
  if (position == "above") {
    c(text, rule)
  } else {
    c(rule, text)
  }
}

#' Render a data frame as plain text
#'
#' @description
#' Steps: convert cells to text, choose column widths capped at `maxWidth`,
#' write the header, write a rule, write every row with wrapping, and add
#' the caption.
#'
#' @param data A data frame to tabulate.
#' @param maxWidth Maximum width of a column in characters.
#' @param sep String placed between columns.
#' @param caption An optional single string used as the table caption.
#' @return A character vector with one element per output line.
#' @export
asPlainText = function(data, maxWidth = 20L, sep = " | ", caption = NULL) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }

  # 1. Convert cells to text
  cells = lapply(data, function(column) {
    ifelse(is.na(column), "", as.character(column))
  })
  align = ifelse(vapply(data, is.numeric, logical(1)), "right", "left")

  # 2. Choose column widths capped at `maxWidth`
  widths = integer(length(cells))
  for (j in seq_along(cells)) {
    widths[j] = min(max(nchar(c(names(data)[j], cells[[j]]))), maxWidth)
  }

  # 3. Write the header
  lines = stackCells(names(data), widths, align, sep)
  # 4. Write a rule
  joint = gsub("|", "+", gsub(" ", "-", sep), fixed = TRUE)
  lines = c(lines, paste(strrep("-", widths), collapse = joint))
  # 5. Write every row with wrapping
  for (i in seq_len(nrow(data))) {
    row = vapply(cells, `[`, character(1), i)
    lines = c(lines, stackCells(row, widths, align, sep))
  }

  # 6. Add the caption
  if (!is.null(caption)) {
    lines = c(plainCaption(caption, nchar(lines[1L])), lines)
  }

  lines
}

#' Render one row of a data frame as a record
#'
#' @description
#' Writes one `label value` line per column. The `"table"` style draws a
#' two-column box instead.
#'
#' @param data A data frame to tabulate.
#' @param row Index of the row to render.
#' @param style One of `"colon"`, `"dots"` or `"table"`.
#' @param labelWidth Width of the label column; the longest name when
#'   `NULL`.
#' @return A character vector with one element per output line.
#' @export
plainRecord = function(data, row = 1L, style = c("colon", "dots", "table"),
                       labelWidth = NULL) {
  style = match.arg(style)
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }

  if (row < 1L || row > nrow(data)) {
    stop("`row` must index a row of `data`.", call. = FALSE)
  }

  labels = names(data)
  values = vapply(data[row, , drop = FALSE], function(value) {
    if (is.na(value)) "" else as.character(value)
  }, character(1))
  if (is.null(labelWidth)) {
    labelWidth = max(nchar(labels))
  }

  switch(style,
    colon = {
      keys = padText(paste0(labels, ":"), labelWidth + 1L)
      paste(keys, values)
    },
    dots = {
      keys = padText(paste0(labels, " "), labelWidth + 3L, fill = ".")
      paste(keys, values)
    },
    table = {
      valueWidth = max(nchar(values))
      keys = padText(labels, labelWidth)
      rule = paste0("+", strrep("-", labelWidth + 2L), "+", strrep("-", valueWidth + 2L), "+")
      body = paste0("| ", keys, " | ", padText(values, valueWidth), " |")
      c(rule, body, rule)
    }
  )
}
