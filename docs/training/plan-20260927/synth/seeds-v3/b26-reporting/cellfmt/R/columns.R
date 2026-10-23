#' Classify a column for formatting
#'
#' @param x A column of a data frame.
#' @return One of `"integer"`, `"double"`, `"logical"`, `"date"` or `"text"`.
#' @keywords internal
column_type <- function(x) {
  if (inherits(x, c("Date", "POSIXt"))) {
    return("date")
  }

  if (is.logical(x)) {
    return("logical")
  }

  if (is.integer(x)) {
    return("integer")
  }

  if (is.double(x)) {
    return("double")
  }

  return("text")
}

#' Format one column according to its type
#'
#' @param x A column of a data frame.
#' @param type One of `"integer"`, `"double"`, `"percent"`, `"scientific"`,
#'   `"logical"`, `"date"` or `"text"`.
#' @param digits Number of decimal places used for `"double"`, `"percent"`
#'   and `"scientific"` columns.
#' @param na_string String used in place of missing values.
#' @param big_mark Character used to separate thousands.
#' @return A character vector the same length as `x`.
#' @family column formatters
#' @export
format_column <- function(x,
                          type = c("integer", "double", "percent", "scientific",
                                   "logical", "date", "text"),
                          digits = 2, na_string = "", big_mark = ",") {
  type <- match.arg(type)
  numeric_types <- c("integer", "double", "percent", "scientific")
  if (type %in% numeric_types && !is.numeric(x)) {
    stop(sprintf("A `%s` column must be numeric.", type), call. = FALSE)
  }

  out <- switch(type,
    integer = {
      value <- round(x)
      formatC(value, format = "d", big.mark = big_mark)
    },
    double = {
      value <- round(x, digits)
      formatC(value, format = "f", digits = digits, big.mark = big_mark)
    },
    percent = {
      value <- round(100 * x, digits)
      text <- formatC(value, format = "f", digits = digits, big.mark = big_mark)
      paste0(text, "%")
    },
    scientific = {
      value <- signif(x, digits + 1)
      formatC(value, format = "e", digits = digits)
    },
    logical = {
      ifelse(x, "yes", "no")
    },
    date = {
      format(x, "%Y-%m-%d")
    },
    text = {
      as.character(x)
    }
  )
  out[is.na(x)] <- na_string
  return(out)
}

#' Default display settings for a column type
#'
#' @param type One of `"integer"`, `"double"`, `"percent"`, `"scientific"`,
#'   `"logical"`, `"date"` or `"text"`.
#' @return A list with elements `align`, `digits`, `na_string` and
#'   `min_width`.
#' @keywords internal
column_defaults <- function(type = c("integer", "double", "percent", "scientific",
                                     "logical", "date", "text")) {
  type <- match.arg(type)
  defaults <- switch(type,
    integer = list(
      align = "right",
      digits = 0,
      na_string = "",
      min_width = 1L
    ),
    double = list(
      align = "right",
      digits = 2,
      na_string = "",
      min_width = 4L
    ),
    percent = list(
      align = "right",
      digits = 1,
      na_string = "",
      min_width = 4L
    ),
    scientific = list(
      align = "right",
      digits = 3,
      na_string = "",
      min_width = 9L
    ),
    logical = list(
      align = "centre",
      digits = 0,
      na_string = "-",
      min_width = 3L
    ),
    date = list(
      align = "left",
      digits = 0,
      na_string = "-",
      min_width = 10L
    ),
    text = list(
      align = "left",
      digits = 0,
      na_string = "",
      min_width = 1L
    )
  )
  return(defaults)
}

#' Default alignment for column types
#'
#' Numbers are right-aligned, logical values centred and everything else
#' left-aligned.
#'
#' @param types A character vector of column types.
#' @return A character vector of `"left"`, `"right"` or `"centre"`, named
#'   like `types`.
#' @keywords internal
column_align <- function(types) {
  align <- rep("left", length(types))
  align[types %in% c("integer", "double", "percent")] <- "right"
  align[types == "logical"] <- "centre"
  names(align) <- names(types)
  return(align)
}

#' Pad cells to a common width
#'
#' @param x A character vector of cells.
#' @param width Target width in characters; longer cells are left as is.
#' @param align One of `"left"`, `"right"` or `"centre"`.
#' @return A character vector the same length as `x`.
#' @family column formatters
#' @export
pad_cells <- function(x, width, align = c("left", "right", "centre")) {
  align <- match.arg(align)
  gap <- pmax(width - nchar(x), 0L)
  out <- switch(align,
    left = {
      paste0(x, strrep(" ", gap))
    },
    right = {
      paste0(strrep(" ", gap), x)
    },
    centre = {
      before <- gap %/% 2L
      after <- gap - before
      paste0(strrep(" ", before), x, strrep(" ", after))
    }
  )
  return(out)
}

#' Display width of each column
#'
#' @param cells A list of character vectors, one per column.
#' @param header A character vector of column headers.
#' @return An integer vector with the widest cell or header per column.
#' @keywords internal
column_widths <- function(cells, header) {
  if (length(cells) != length(header)) {
    stop("`cells` and `header` must have the same length.", call. = FALSE)
  }
  widths <- integer(length(cells))
  for (i in seq_along(cells)) {
    widths[i] <- max(nchar(c(header[i], cells[[i]])))
  }
  return(widths)
}

#' Format every column of a data frame
#'
#' Columns named in `percent` are formatted as percentages; the others use
#' the type reported by [column_type()].
#'
#' @param data A data frame to format.
#' @param digits Number of decimal places used for `"double"` and
#'   `"percent"` columns.
#' @param percent Names of columns holding proportions.
#' @param na_string String used in place of missing values.
#' @return A list with elements `cells` (one character vector per column),
#'   `types` and `align`.
#' @export
format_columns <- function(data, digits = 2, percent = character(), na_string = "") {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }

  unknown <- setdiff(percent, names(data))
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown column(s) in `percent`: %s.", paste(unknown, collapse = ", ")),
         call. = FALSE)
  }

  types <- vapply(data, column_type, character(1))
  types[names(types) %in% percent] <- "percent"

  cells <- vector("list", ncol(data))
  names(cells) <- names(data)
  for (name in names(data)) {
    cells[[name]] <- format_column(data[[name]], types[[name]], digits, na_string)
  }

  return(list(cells = cells, types = types, align = column_align(types)))
}

#' Render a data frame as a fixed-width text table
#'
#' Steps: format the columns, measure them, pad the header and every row,
#' draw a rule under the header, then add a footer with the row count.
#'
#' @param data A data frame to format.
#' @param digits Number of decimal places used for `"double"` and
#'   `"percent"` columns.
#' @param percent Names of columns holding proportions.
#' @param sep String placed between columns.
#' @param rule Character repeated to draw the rule under the header.
#' @param footer Whether to add a closing rule and the row count.
#' @return A character vector with one element per output line.
#' @export
format_text_table <- function(data, digits = 2, percent = character(), sep = "  ",
                              rule = "-", footer = FALSE) {
  # 1. Format the columns
  formatted <- format_columns(data, digits = digits, percent = percent)
  cells <- formatted$cells
  align <- formatted$align
  header <- names(data)

  # 2. Measure them
  widths <- column_widths(cells, header)

  # 3. Pad the header and every row
  head_cells <- character(length(header))
  for (i in seq_along(header)) {
    head_cells[i] <- pad_cells(header[i], widths[i], align[[i]])
  }

  body <- vector("list", length(cells))
  for (i in seq_along(cells)) {
    body[[i]] <- pad_cells(cells[[i]], widths[i], align[[i]])
  }

  head_line <- paste(head_cells, collapse = sep)
  body_lines <- do.call(paste, c(body, list(sep = sep)))

  # 4. Draw a rule under the header
  rule_line <- paste(strrep(rule, widths), collapse = sep)
  lines <- c(head_line, rule_line, body_lines)

  # 5. Add a footer with the row count
  if (footer) {
    noun <- if (nrow(data) == 1L) "row" else "rows"
    count <- paste(formatC(nrow(data), big.mark = ","), noun)
    lines <- c(lines, rule_line, count)
  }
  return(lines)
}
