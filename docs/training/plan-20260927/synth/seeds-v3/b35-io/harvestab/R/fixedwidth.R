#' Fixed-width layout from field widths
#'
#' @param widths A named integer vector of field widths, in file order.
#' @return A data frame with columns `name`, `width`, `start` and `end`.
#' @noRd
fwf.layout <- function(widths) {
  if (is.null(names(widths)) || any(names(widths) == "")) {
    stop("`widths` must be a fully named vector.", call. = FALSE)
  }
  if (any(widths < 1)) {
    stop("`widths` must all be positive.", call. = FALSE)
  }
  end <- cumsum(widths)
  data.frame(name = names(widths), width = as.integer(widths),
             start = as.integer(end - widths + 1), end = as.integer(end),
             stringsAsFactors = FALSE)
}

#' Pad values to a fixed width
#'
#' @param x A character vector.
#' @param width Target width.
#' @param align One of `"left"`, `"right"` or `"centre"`.
#' @return A character vector whose elements all have `width` characters.
#' @noRd
pad.field <- function(x, width, align = c("left", "right", "centre")) {
  align <- match.arg(align)
  if (!is.numeric(width) || length(width) != 1 || width < 1) {
    stop("`width` must be a single positive number.", call. = FALSE)
  }
  x[is.na(x)] <- ""
  if (any(nchar(x) > width)) {
    stop(sprintf("Values longer than %d characters cannot be padded.", width), call. = FALSE)
  }

  gap <- width - nchar(x)
  switch(align,
    left = {
      paste0(x, strrep(" ", gap))
    },
    right = {
      paste0(strrep(" ", gap), x)
    },
    centre = {
      before <- gap %/% 2
      paste0(strrep(" ", before), x, strrep(" ", gap - before))
    }
  )
}

#' Remove fixed-width padding
#'
#' @param x A character vector of padded values.
#' @param align One of `"left"`, `"right"` or `"centre"`.
#' @return A character vector; empty values become `NA`.
#' @noRd
unpad.field <- function(x, align = c("left", "right", "centre")) {
  align <- match.arg(align)
  out <- switch(align,
    left = {
      sub(" +$", "", x)
    },
    right = {
      sub("^ +", "", x)
    },
    centre = {
      trimws(x)
    }
  )
  out[out == ""] <- NA_character_
  out
}

#' Write a fixed-width file
#'
#' Numeric columns are right-aligned and all other columns left-aligned;
#' dates are written as `YYYY-MM-DD`.
#'
#' @param data A data frame whose columns are named in `widths`.
#' @param path Destination file path.
#' @param widths A named integer vector of field widths, in file order.
#' @param header Whether to write a first line of column names, each cut to
#'   its field width and aligned like the values below it.
#' @return `path`, invisibly.
#' @export
write.fwf <- function(data, path, widths, header = FALSE) {
  layout <- fwf.layout(widths)
  missing.cols <- setdiff(layout$name, names(data))
  if (length(missing.cols) > 0) {
    stop(sprintf("`data` is missing column(s): %s", paste(missing.cols, collapse = ", ")),
         call. = FALSE)
  }
  cells <- vector("list", nrow(layout))
  aligns <- character(nrow(layout))
  for (i in seq_len(nrow(layout))) {
    column <- data[[layout$name[i]]]
    aligns[i] <- if (is.numeric(column)) "right" else "left"
    if (inherits(column, "Date")) {
      column <- format(column, "%Y-%m-%d")
    }
    cells[[i]] <- pad.field(as.character(column), layout$width[i], aligns[i])
  }
  lines <- do.call(paste0, cells)

  # Optional header: column names padded like their values
  if (header) {
    titles <- character(nrow(layout))
    for (i in seq_len(nrow(layout))) {
      titles[i] <- pad.field(substr(layout$name[i], 1, layout$width[i]), layout$width[i],
                             aligns[i])
    }
    lines <- c(paste(titles, collapse = ""), lines)
  }
  writeLines(lines, path)
  invisible(path)
}

#' Read a fixed-width file
#'
#' Mirrors `write.fwf()`: each field is cut out by position, its padding is
#' removed from both sides and its type is guessed with
#' [utils::type.convert()].
#'
#' @param path Path to the fixed-width file.
#' @param widths A named integer vector of field widths, in file order.
#' @param header Whether the first line holds column names, which must match
#'   the names in `widths` cut to their field widths.
#' @return A data frame with one column per element of `widths`.
#' @export
read.fwf.lines <- function(path, widths, header = FALSE) {
  layout <- fwf.layout(widths)
  lines <- readLines(path)

  # Optional header: check it against the layout and drop it
  if (header) {
    titles <- trimws(substring(lines[1], layout$start, layout$end))
    if (!identical(titles, substr(layout$name, 1, layout$width))) {
      stop(sprintf("Header of `%s` does not match the layout.", path), call. = FALSE)
    }
    lines <- lines[-1]
  }
  short <- which(nchar(lines) < max(layout$end))
  if (length(short) > 0) {
    stop(sprintf("Line %d is shorter than the layout (%d characters).", short[1],
                 max(layout$end)), call. = FALSE)
  }

  columns <- vector("list", nrow(layout))
  names(columns) <- layout$name
  for (i in seq_len(nrow(layout))) {
    raw <- substring(lines, layout$start[i], layout$end[i])
    columns[[i]] <- utils::type.convert(unpad.field(raw, "centre"), as.is = TRUE)
  }
  as.data.frame(columns, stringsAsFactors = FALSE)
}

#' Summarise the layout of a fixed-width file
#'
#' @param path Path to the fixed-width file.
#' @param widths A named integer vector of field widths, in file order.
#' @return The layout data frame with an added `filled` column: the share of
#'   lines with a non-blank value in each field.
#' @export
fwf.profile <- function(path, widths) {
  layout <- fwf.layout(widths)
  lines <- readLines(path)
  filled <- numeric(nrow(layout))
  for (i in seq_len(nrow(layout))) {
    raw <- substring(lines, layout$start[i], layout$end[i])
    filled[i] <- if (length(lines) == 0) NA_real_ else mean(nzchar(trimws(raw)))
  }
  layout$filled <- filled
  layout
}
