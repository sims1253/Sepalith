#' Input formats understood by the parsers
#'
#' @format A named character vector mapping a format name to the regular
#'   expression a value must match.
#' @keywords internal
DATE_PATTERNS <- c(
  iso = "^\\d{4}-\\d{2}-\\d{2}$",
  dmy = "^\\d{1,2}[./]\\d{1,2}[./]\\d{4}$",
  mdy = "^\\d{1,2}/\\d{1,2}/\\d{4}$",
  compact = "^\\d{8}$",
  ordinal = "^\\d{4}-\\d{3}$"
)

#' Parse dates in a named input format
#'
#' @param x A character vector.
#' @param format One of `"iso"` (`2024-03-05`), `"dmy"` (`5.3.2024` or
#'   `05/03/2024`), `"mdy"` (`3/5/2024`), `"compact"` (`20240305`) or
#'   `"ordinal"` (`2024-065`).
#' @param strict If `TRUE`, non-empty inputs that fail to parse are an
#'   error rather than `NA`.
#' @return A `Date` vector the same length as `x`.
#' @export
parse_dates <- function(x, format = c("iso", "dmy", "mdy", "compact", "ordinal"),
                        strict = FALSE) {
  format <- match.arg(format)
  x <- trimws(as.character(x))
  ok <- !is.na(x) & grepl(DATE_PATTERNS[[format]], x, perl = TRUE)
  out <- rep(as.Date(NA), length(x))

  switch(format,
    iso = {
      out[ok] <- as.Date(x[ok], format = "%Y-%m-%d")
    },
    dmy = {
      out[ok] <- as.Date(gsub("[./]", "-", x[ok]), format = "%d-%m-%Y")
    },
    mdy = {
      out[ok] <- as.Date(x[ok], format = "%m/%d/%Y")
    },
    compact = {
      out[ok] <- as.Date(x[ok], format = "%Y%m%d")
    },
    ordinal = {
      out[ok] <- as.Date(x[ok], format = "%Y-%j")
    }
  )
  failed <- is.na(out) & !is.na(x) & nzchar(x)
  if (strict && any(failed)) {
    stop(sprintf("%d value(s) could not be parsed as `%s`, e.g. \"%s\".",
      sum(failed), format, x[failed][1]), call. = FALSE)
  }
  return(out)
}

#' Report how well each input format fits a character vector
#'
#' @param x A character vector.
#' @param formats Character vector of format names to try.
#' @return A data frame with one row per format and columns `format`,
#'   `matched` (values matching the format's pattern), `parsed` (values
#'   giving a real date) and `invalid` (matched but not a real date).
#' @export
date_format_report <- function(x, formats = names(DATE_PATTERNS)) {
  x <- trimws(as.character(x))
  matched <- integer(length(formats))
  parsed <- integer(length(formats))
  for (i in seq_along(formats)) {
    matched[i] <- sum(grepl(DATE_PATTERNS[[formats[i]]], x, perl = TRUE))
    parsed[i] <- sum(!is.na(parse_dates(x, formats[i])))
  }
  invalid <- matched - parsed
  return(data.frame(format = formats, matched = matched, parsed = parsed,
    invalid = invalid))
}

#' Guess the input format of a character vector
#'
#' The format that parses the most values wins; ties keep the earlier
#' candidate.
#'
#' @param x A character vector.
#' @param candidates Character vector of format names to try, in order of
#'   preference.
#' @return A single format name, or `NA_character_` if nothing parses.
#' @export
guess_date_format <- function(x, candidates = names(DATE_PATTERNS)) {
  report <- date_format_report(x, candidates)
  if (max(report$parsed) == 0L) {
    return(NA_character_)
  }
  return(report$format[which.max(report$parsed)])
}

#' Split date strings into their numeric parts
#'
#' Unlike the parsers, this does not check that the parts form a real
#' calendar date, which makes it useful for reporting why a value failed.
#'
#' @param x A character vector.
#' @param format One of `"iso"`, `"dmy"`, `"mdy"`, `"compact"` or
#'   `"ordinal"`.
#' @return A data frame with columns `input`, `year`, `month`, `day` and
#'   `valid` (whether the parts form an existing date); rows that do not
#'   match the format have `NA` parts.
#' @export
split_date_parts <- function(x, format = c("iso", "dmy", "mdy", "compact", "ordinal")) {
  format <- match.arg(format)
  x <- trimws(as.character(x))
  ok <- grepl(DATE_PATTERNS[[format]], x, perl = TRUE)
  year <- month <- day <- rep(NA_integer_, length(x))

  switch(format,
    iso = {
      year[ok] <- as.integer(substr(x[ok], 1L, 4L))
      month[ok] <- as.integer(substr(x[ok], 6L, 7L))
      day[ok] <- as.integer(substr(x[ok], 9L, 10L))
    },
    dmy = {
      bits <- strsplit(x[ok], "[./]")
      day[ok] <- as.integer(vapply(bits, `[`, character(1), 1L))
      month[ok] <- as.integer(vapply(bits, `[`, character(1), 2L))
      year[ok] <- as.integer(vapply(bits, `[`, character(1), 3L))
    },
    mdy = {
      bits <- strsplit(x[ok], "/")
      month[ok] <- as.integer(vapply(bits, `[`, character(1), 1L))
      day[ok] <- as.integer(vapply(bits, `[`, character(1), 2L))
      year[ok] <- as.integer(vapply(bits, `[`, character(1), 3L))
    },
    compact = {
      year[ok] <- as.integer(substr(x[ok], 1L, 4L))
      month[ok] <- as.integer(substr(x[ok], 5L, 6L))
      day[ok] <- as.integer(substr(x[ok], 7L, 8L))
    },
    ordinal = {
      parsed <- as.Date(x[ok], format = "%Y-%j")
      year[ok] <- as.integer(format(parsed, "%Y"))
      month[ok] <- as.integer(format(parsed, "%m"))
      day[ok] <- as.integer(format(parsed, "%d"))
    }
  )
  iso <- sprintf("%04d-%02d-%02d", year, month, day)
  valid <- ok & !is.na(as.Date(iso, format = "%Y-%m-%d", optional = TRUE))
  return(data.frame(input = x, year = year, month = month, day = day, valid = valid))
}

#' Parse a column of mixed-format dates
#'
#' Values are parsed with each format in `order`; later formats only fill
#' the values still missing after the earlier ones.
#'
#' @param x A character vector.
#' @param order Character vector of format names, tried in order.
#' @return A list with elements `dates` (the parsed `Date` vector), `format`
#'   (the format name that parsed each value, `NA` if none) and `n_failed`.
#' @export
parse_mixed_dates <- function(x, order = c("iso", "compact", "ordinal", "dmy")) {
  dates <- rep(as.Date(NA), length(x))
  format <- rep(NA_character_, length(x))
  for (fmt in order) {
    todo <- is.na(dates)
    if (!any(todo)) {
      break
    }
    parsed <- parse_dates(x[todo], fmt)
    hit <- !is.na(parsed)
    dates[todo][hit] <- parsed[hit]
    format[todo][hit] <- fmt
  }
  n_failed <- sum(is.na(dates) & !is.na(x))
  return(list(dates = dates, format = format, n_failed = n_failed))
}

#' Parse date columns of a data frame in place
#'
#' @param data A data frame.
#' @param cols Names of the character columns to parse.
#' @param format Format name passed to [parse_dates()].
#' @param suffix If `NULL`, parsed columns replace the originals; otherwise
#'   they are added under the original name plus `suffix`.
#' @return `data` with parsed columns and an attribute `parse_failures`, a
#'   named integer vector counting unparsed non-empty values per column.
#' @export
parse_date_columns <- function(data, cols, format = "iso", suffix = NULL) {
  missing_cols <- setdiff(cols, names(data))
  if (length(missing_cols) > 0L) {
    stop(sprintf("Column(s) not found: %s.", paste(missing_cols, collapse = ", ")),
      call. = FALSE)
  }
  failures <- integer(length(cols))
  names(failures) <- cols
  for (col in cols) {
    raw <- trimws(as.character(data[[col]]))
    parsed <- parse_dates(raw, format)
    failures[[col]] <- sum(is.na(parsed) & !is.na(raw) & nzchar(raw))
    target <- if (is.null(suffix)) col else paste0(col, suffix)
    data[[target]] <- parsed
  }
  attr(data, "parse_failures") <- failures
  return(data)
}
