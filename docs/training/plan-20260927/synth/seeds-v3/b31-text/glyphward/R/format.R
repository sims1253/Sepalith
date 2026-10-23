#' @title Format byte counts
#' @param x A numeric vector of sizes in bytes.
#' @param digits Number of decimal places for sizes of one kilobyte and above.
#' @param si Use powers of 1000 (`"kB"`) instead of powers of 1024 (`"KiB"`)?
#' @return A character vector such as `"512 B"` or `"1.5 MiB"`.
#' @family formatters
#' @export
fmt.bytes <- function(x, digits = 1L, si = FALSE) {
  if (si) {
    base <- 1000
    units <- c("B", "kB", "MB", "GB", "TB")
  } else {
    base <- 1024
    units <- c("B", "KiB", "MiB", "GiB", "TiB")
  }
  power <- pmin(floor(log(pmax(abs(x), 1), base)), length(units) - 1)
  scaled <- x / base^power
  whole <- sprintf("%d %s", as.integer(round(scaled)), units[power + 1])
  fractional <- sprintf("%.*f %s", as.integer(digits), scaled, units[power + 1])
  out <- ifelse(power == 0, whole, fractional)
  out[is.na(x)] <- NA_character_
  out
}

#' @title Format durations
#' @param x A numeric vector of durations in seconds.
#' @return A character vector such as `"1h 02m 03s"`, `"4m 05s"` or `"7s"`.
#' @family formatters
#' @export
fmt.duration <- function(x) {
  secs <- round(x)
  out <- rep(NA_character_, length(x))
  # Hours
  hours <- secs %/% 3600
  secs <- secs - hours * 3600
  # Minutes
  minutes <- secs %/% 60
  secs <- secs - minutes * 60

  # Pick the layout by the largest non-zero unit
  has.h <- !is.na(secs) & hours > 0
  has.m <- !is.na(secs) & (minutes > 0 | has.h)
  out[has.h] <- sprintf("%dh %02dm %02ds", as.integer(hours[has.h]),
                        as.integer(minutes[has.h]), as.integer(secs[has.h]))
  only.m <- has.m & !has.h
  out[only.m] <- sprintf("%dm %02ds", as.integer(minutes[only.m]), as.integer(secs[only.m]))
  only.s <- !is.na(secs) & !has.m
  out[only.s] <- sprintf("%ds", as.integer(secs[only.s]))
  out
}

#' @title Format values of a named kind
#' @param x A numeric vector.
#' @param type The kind of value in `x`: sizes in bytes, durations in
#'   seconds, proportions, counts or ranks.
#' @param digits Number of decimal places, where the formatter uses any.
#' @param width Minimum width; results are right-aligned with spaces.
#' @return A character vector the same length as `x`; missing values are
#'   shown as `"-"`.
#' @family formatters
#' @export
fmt.value <- function(x, type = c("bytes", "duration", "percent", "count", "ordinal"),
                      digits = 1L, width = 0L) {
  type <- match.arg(type)
  if (!is.numeric(x)) {
    stop("`x` must be a numeric vector.", call. = FALSE)
  }

  out <- switch(type,
    bytes = {
      fmt.bytes(x, digits = digits)
    },
    duration = {
      fmt.duration(x)
    },
    percent = {
      value <- sprintf("%.*f%%", as.integer(digits), 100 * x)
      value[is.na(x)] <- NA_character_
      value
    },
    count = {
      value <- formatC(round(x), format = "d", big.mark = ",")
      value[is.na(x)] <- NA_character_
      value
    },
    ordinal = {
      n <- as.integer(round(x))
      last.two <- n %% 100L
      suffix <- c("th", "st", "nd", "rd", rep("th", 6L))[n %% 10L + 1L]
      suffix[last.two %in% 11:13] <- "th"
      value <- paste0(n, suffix)
      value[is.na(x)] <- NA_character_
      value
    }
  )

  out[is.na(out)] <- "-"
  sprintf("%*s", as.integer(width), out)
}

#' @title Describe a numeric vector in one line
#' @param x A numeric vector.
#' @param digits Number of decimal places for every element but `n`.
#' @return A named character vector with elements `n`, `mean`, `sd`, `min`,
#'   `q1`, `median`, `q3` and `max`, plus an attribute `line` that joins them
#'   as `name=value`.
#' @export
describe.numbers <- function(x, digits = 2L) {
  x <- x[!is.na(x)]
  q <- stats::quantile(x, c(0.25, 0.5, 0.75), names = FALSE)
  fmt <- function(v) sprintf("%.*f", as.integer(digits), v)
  out <- c(
    n = sprintf("%d", length(x)),
    mean = fmt(mean(x)),
    sd = fmt(stats::sd(x)),
    min = fmt(min(x)),
    q1 = fmt(q[1]),
    median = fmt(q[2]),
    q3 = fmt(q[3]),
    max = fmt(max(x))
  )

  attr(out, "line") <- paste(sprintf("%s=%s", names(out), out), collapse = " ")
  out
}

#' @title Format a data frame as fixed-width text
#' @param data A data frame; every column is converted with `as.character()`.
#' @param align Alignment per column, `"l"` or `"r"`, recycled.
#' @param sep String placed between columns.
#' @param title Optional title printed above the table.
#' @param footer Optional note printed below the table.
#' @return A character vector with the title, a header line, a rule, one line
#'   per row, a closing rule and the footer.
#' @export
fmt.table <- function(data, align = "l", sep = "  ", title = NULL, footer = NULL) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }

  cells <- lapply(data, as.character)
  cells <- lapply(cells, function(v) ifelse(is.na(v), "", v))
  align <- rep_len(align, length(cells))
  if (!all(align %in% c("l", "r"))) {
    stop("`align` must contain only \"l\" and \"r\".", call. = FALSE)
  }
  widths <- pmax(nchar(names(data)), vapply(cells, function(v) max(0L, nchar(v)), 0L))

  # Left-aligned columns pad on the right, right-aligned ones on the left
  specs <- ifelse(align == "l", "%-*s", "%*s")
  header <- character(length(cells))
  body <- vector("list", length(cells))
  for (j in seq_along(cells)) {
    header[j] <- sprintf(specs[j], widths[j], names(data)[j])
    body[[j]] <- sprintf(specs[j], widths[j], cells[[j]])
  }

  rule <- paste(vapply(widths, function(w) strrep("-", w), ""), collapse = sep)
  rows <- if (nrow(data) > 0L) do.call(paste, c(body, sep = sep)) else character()

  # Assemble title, header, rule, rows, rule and footer
  out <- c(paste(header, collapse = sep), rule, rows, rule)
  if (!is.null(title)) {
    out <- c(title, out)
  }
  if (!is.null(footer)) {
    out <- c(out, footer)
  }
  out
}

#' @title Format a named list as aligned key-value lines
#' @param x A named list or named vector; each value is collapsed with `", "`.
#' @param sep String placed between key and value.
#' @param indent Number of spaces before every line.
#' @return A character vector with one line per element of `x`.
#' @export
fmt.kv <- function(x, sep = ": ", indent = 0L) {
  keys <- names(x)
  if (is.null(keys) || any(keys == "")) {
    stop("Every element of `x` must be named.", call. = FALSE)
  }
  values <- vapply(x, function(v) {
    if (length(v) == 0L) "" else paste(as.character(v), collapse = ", ")
  }, character(1))
  width <- max(nchar(keys))
  prefix <- strrep(" ", indent)
  paste0(prefix, sprintf("%-*s", width, keys), sep, values)
}
