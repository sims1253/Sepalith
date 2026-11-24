#' Trim a set of characters from the ends of strings
#'
#' @param x A character vector.
#' @param chars A single string listing the characters to remove.
#' @param side Which end to trim.
#' @return A character vector the same length as `x`.
#' @export
trimChars = function(x, chars = " ", side = c("both", "left", "right")) {
  side = match.arg(side)
  # Escape the characters that are special inside a bracket expression
  escaped = gsub("([]\\\\^-])", "\\\\\\1", chars)
  set = paste0("[", escaped, "]")
  if (side %in% c("both", "left")) {
    x = sub(paste0("^", set, "+"), "", x, perl = TRUE)
  }
  if (side %in% c("both", "right")) {
    x = sub(paste0(set, "+$"), "", x, perl = TRUE)
  }
  x
}

#' Remove a prefix and a suffix
#'
#' @param x A character vector.
#' @param prefix A literal prefix, or `NULL` to keep the start unchanged.
#' @param suffix A literal suffix, or `NULL` to keep the end unchanged.
#' @return A character vector the same length as `x`.
#' @export
stripAffix = function(x, prefix = NULL, suffix = NULL) {
  # Remove `prefix`
  if (!is.null(prefix) && nzchar(prefix)) {
    hasPrefix = startsWith(x, prefix)
    x[hasPrefix] = substr(x[hasPrefix], nchar(prefix) + 1L, nchar(x[hasPrefix]))
  }
  # Remove `suffix`
  if (!is.null(suffix) && nzchar(suffix)) {
    hasSuffix = endsWith(x, suffix)
    x[hasSuffix] = substr(x[hasSuffix], 1L, nchar(x[hasSuffix]) - nchar(suffix))
  }
  x
}

#' Pad strings to a common width
#'
#' @param x A character vector.
#' @param width Target width in characters. Longer strings are left as they
#'   are.
#' @param side Where to add the padding.
#' @param pad A single padding character.
#' @return A character vector the same length as `x`.
#' @export
padText = function(x, width, side = c("left", "right", "both"), pad = " ") {
  side = match.arg(side)
  missing = pmax(width - nchar(x), 0L)
  switch(side,
    left = {
      leftPad = missing
      rightPad = 0L
    },
    right = {
      leftPad = 0L
      rightPad = missing
    },
    both = {
      leftPad = missing %/% 2L
      rightPad = missing - leftPad
    }
  )

  paste0(strrep(pad, leftPad), x, strrep(pad, rightPad))
}

#' Shorten strings to a maximum width
#'
#' @param x A character vector.
#' @param width Maximum width in characters, including `ellipsis`.
#' @param from Where to cut: the `"tail"` keeps the start, the `"head"`
#'   keeps the end, `"middle"` keeps both ends and `"word"` keeps the start
#'   up to the last complete word.
#' @param ellipsis Marker inserted where text was removed.
#' @return A character vector the same length as `x`.
#' @export
truncateText = function(x, width, from = c("tail", "head", "middle", "word"),
                        ellipsis = "...") {
  from = match.arg(from)
  if (!is.numeric(width) || length(width) != 1L || is.na(width)) {
    stop("`width` must be a single number.", call. = FALSE)
  }
  keep = width - nchar(ellipsis)
  if (keep < 1L) {
    stop("`width` must be larger than the width of `ellipsis`.", call. = FALSE)
  }

  n = nchar(x)
  long = !is.na(x) & n > width
  switch(from,
    tail = {
      kept = substr(x[long], 1L, keep)
      x[long] = paste0(kept, ellipsis)
    },
    head = {
      kept = substr(x[long], n[long] - keep + 1L, n[long])
      x[long] = paste0(ellipsis, kept)
    },
    middle = {
      headKept = substr(x[long], 1L, ceiling(keep / 2))
      tailKept = substr(x[long], n[long] - floor(keep / 2) + 1L, n[long])
      x[long] = paste0(headKept, ellipsis, tailKept)
    },
    word = {
      kept = substr(x[long], 1L, keep + 1L)
      kept = sub("[[:space:]]+[^[:space:]]*$", "", kept)
      x[long] = paste0(substr(kept, 1L, keep), ellipsis)
    }
  )
  x
}

#' Wrap a paragraph greedily
#'
#' @param text A single string.
#' @param width Maximum line width in characters. Words longer than `width`
#'   get a line of their own.
#' @param indent Number of spaces before every line after the first.
#' @return A character vector of lines.
#' @export
wrapLine = function(text, width = 72L, indent = 0L) {
  words = strsplit(trimws(text), "[[:space:]]+")[[1]]
  if (length(words) == 0L) {
    return(character())
  }

  lines = character()
  current = words[1]
  prefix = strrep(" ", indent)
  for (word in words[-1]) {
    limit = if (length(lines) == 0L) width else width - indent
    if (nchar(current) + 1L + nchar(word) <= limit) {
      current = paste(current, word)
    } else {
      lines = c(lines, current)
      current = word
    }
  }

  lines = c(lines, current)
  lines[-1] = paste0(prefix, lines[-1])
  lines
}

#' Align lines within a fixed width
#'
#' @param lines A character vector of lines, for example from [wrapLine()].
#' @param width The line width to align to. Longer lines are left unchanged.
#' @param align How to align: `"left"`, `"right"`, `"centre"` or `"justify"`.
#'   Justified text spreads the missing spaces over the word gaps, left to
#'   right, and leaves the last line left-aligned.
#' @return A character vector the same length as `lines`.
#' @export
alignBlock = function(lines, width = 72L, align = c("left", "right", "centre", "justify")) {
  align = match.arg(align)
  lines = trimws(lines)
  missing = pmax(width - nchar(lines), 0L)

  switch(align,
    left = {
      out = paste0(lines, strrep(" ", missing))
    },
    right = {
      out = paste0(strrep(" ", missing), lines)
    },
    centre = {
      leftPad = missing %/% 2L
      rightPad = missing - leftPad
      out = paste0(strrep(" ", leftPad), lines, strrep(" ", rightPad))
    },
    justify = {
      out = lines
      for (i in seq_along(lines)) {
        words = strsplit(lines[i], " +")[[1]]
        gaps = length(words) - 1L
        # The last line and single words stay left-aligned
        if (i == length(lines) || gaps < 1L) {
          out[i] = paste0(lines[i], strrep(" ", missing[i]))
          next
        }
        spaces = width - sum(nchar(words))
        each = rep(spaces %/% gaps, gaps)
        extra = spaces %% gaps
        each[seq_len(extra)] = each[seq_len(extra)] + 1L
        out[i] = paste0(paste0(words[-length(words)], strrep(" ", each), collapse = ""),
                        words[length(words)])
      }
    }
  )
  out
}

#' Align delimited lines into columns
#'
#' @param lines A character vector of delimited lines; the first line is the
#'   header.
#' @param sep The field delimiter, matched literally.
#' @param gap Spaces between columns.
#' @param rule Insert a rule of dashes below the header?
#' @return A character vector of padded lines without trailing spaces.
#'   Columns whose body cells are all numbers are right-aligned.
#' @export
alignColumns = function(lines, sep = ",", gap = 2L, rule = TRUE) {
  fields = strsplit(lines, sep, fixed = TRUE)
  nCol = max(lengths(fields))
  fields = lapply(fields, function(f) c(f, rep("", nCol - length(f))))
  cells = trimws(do.call(rbind, fields))
  widths = apply(nchar(cells), 2L, max)
  body = cells[-1, , drop = FALSE]
  numeric = rep(FALSE, nCol)
  if (nrow(body) > 0L) {
    numeric = apply(body, 2L, function(v) all(grepl("^-?[0-9.,]+$", v[v != ""])))
  }

  for (j in seq_len(nCol)) {
    side = if (numeric[j]) "left" else "right"
    cells[, j] = padText(cells[, j], widths[j], side = side)
  }

  out = apply(cells, 1L, paste, collapse = strrep(" ", gap))
  if (rule) {
    dashes = paste(strrep("-", widths), collapse = strrep(" ", gap))
    out = c(out[1], dashes, out[-1])
  }
  sub(" +$", "", out)
}
