#' @title Guess the field delimiter of a delimited text sample
#'
#' @description Picks the candidate that splits every sample line into the
#' same number of fields, preferring the one that yields the most fields.
#'
#' @param sample Character vector of lines from the start of the file.
#' @param candidates Candidate delimiters to try.
#' @return A single character delimiter.
#' @keywords internal
guessDelimiter <- function(sample, candidates = c(",", ";", "\t", "|")) {
  sample <- sample[!is.na(sample) & nzchar(sample)]
  if (length(sample) == 0L) {
    stop("Cannot guess a delimiter from an empty sample.", call. = FALSE)
  }

  # A delimiter only scores if it gives a constant field count on every line.
  # Count separators directly: strsplit() drops a trailing empty field.
  scores <- vapply(candidates, function(d) {
    counts <- nchar(sample) - nchar(gsub(d, "", sample, fixed = TRUE)) + 1L
    if (length(unique(counts)) == 1L && counts[1] > 1L) counts[1] else 0L
  }, integer(1))

  if (all(scores == 0L)) {
    return(",")
  }
  return(candidates[which.max(scores)])
}

#' @title Test whether character values look like a given type
#'
#' @description These helpers drop missing entries (both `NA` and the strings
#' in `na`) and then check that every remaining value matches the type. An
#' all-missing vector never matches.
#'
#' @param x Character vector of raw cell values.
#' @param na Strings treated as missing.
#' @return `TRUE` if every non-missing value parses as this type.
#' @keywords internal
looksInteger <- function(x, na = c("", "NA")) {
  x <- x[!is.na(x) & !x %in% na]
  if (length(x) == 0L) return(FALSE)
  return(all(grepl("^[-+]?[0-9]+$", x)))
}

#' @rdname looksInteger
looksNumeric <- function(x, na = c("", "NA")) {
  x <- x[!is.na(x) & !x %in% na]
  if (length(x) == 0L) return(FALSE)
  return(all(grepl("^[-+]?([0-9]+\\.?[0-9]*|\\.[0-9]+)([eE][-+]?[0-9]+)?$", x)))
}

#' @rdname looksInteger
looksDate <- function(x, na = c("", "NA")) {
  x <- x[!is.na(x) & !x %in% na]
  if (length(x) == 0L) return(FALSE)
  return(all(!is.na(as.Date(x, format = "%Y-%m-%d"))))
}

#' @rdname looksInteger
looksLogical <- function(x, na = c("", "NA")) {
  x <- x[!is.na(x) & !x %in% na]
  if (length(x) == 0L) return(FALSE)
  return(all(toupper(x) %in% c("TRUE", "FALSE", "T", "F")))
}

#' @title Guess a column class for each column of raw values
#'
#' @description Checks types from most to least specific, so `"1"` becomes
#' integer rather than numeric, and falls back to character.
#'
#' @param columns A named list of character vectors, one per column.
#' @param na Strings treated as missing.
#' @return A named character vector of column classes.
#' @keywords internal
guessColumnTypes <- function(columns, na = c("", "NA")) {
  types <- character(length(columns))
  names(types) <- names(columns)

  for (i in seq_along(columns)) {
    x <- columns[[i]]
    # Order matters: logical before integer, integer before numeric
    types[i] <- if (looksLogical(x, na)) {
      "logical"
    } else if (looksInteger(x, na)) {
      "integer"
    } else if (looksNumeric(x, na)) {
      "numeric"
    } else if (looksDate(x, na)) {
      "Date"
    } else {
      "character"
    }
  }

  return(types)
}

#' @title Read a delimited file after sniffing its layout
#'
#' @description Reads the first `nSniff` lines to guess the delimiter and
#' column types, then reads the whole file with those settings.
#'
#' @param path Path to the delimited file.
#' @param nSniff Number of lines used for guessing.
#' @param na Strings treated as missing.
#' @return A data frame with a `"sniffed"` attribute describing the guesses.
#' @export
readCsvSniffed <- function(path, nSniff = 200L, na = c("", "NA")) {
  head <- readLines(path, n = nSniff + 1L, warn = FALSE)
  delim <- guessDelimiter(head)

  # Read the sample as plain text so the type guesses see the raw strings
  sample <- utils::read.table(
    text = head,
    sep = delim,
    header = TRUE,
    colClasses = "character",
    quote = "\"",
    comment.char = "",
    check.names = FALSE
  )
  types <- guessColumnTypes(as.list(sample), na)

  out <- utils::read.table(
    path,
    sep = delim,
    header = TRUE,
    colClasses = unname(types),
    na.strings = na,
    quote = "\"",
    comment.char = "",
    check.names = FALSE
  )
  # keep the guesses so callers can reproduce the read
  attr(out, "sniffed") <- list(delimiter = delim, types = types, lines = length(head))
  return(out)
}

#' @title Profile the columns of a delimited file
#'
#' @description Reads the file as character, then reports per column the
#' guessed type, the share of missing values, the number of distinct values
#' and an example. Useful as a first look at an unfamiliar export.
#'
#' @param path Path to the delimited file.
#' @param na Strings treated as missing.
#' @param maxRows Maximum number of data rows to read.
#' @return A data frame with one row per column of the file.
#' @export
profileCsv <- function(path, na = c("", "NA"), maxRows = 10000L) {
  head <- readLines(path, n = 50L, warn = FALSE)
  delim <- guessDelimiter(head)

  # Keep every cell as a string; `na` is applied by hand below
  raw <- utils::read.table(
    path,
    sep = delim,
    header = TRUE,
    colClasses = "character",
    nrows = maxRows,
    quote = "\"",
    comment.char = "",
    check.names = FALSE,
    na.strings = character(0)
  )
  # guesses use the same rules as readCsvSniffed()
  types <- guessColumnTypes(as.list(raw), na)

  nCols <- ncol(raw)
  profile <- data.frame(
    column = names(raw),
    type = unname(types),
    missing = numeric(nCols),
    distinct = integer(nCols),
    example = character(nCols),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nCols)) {
    x <- raw[[i]]
    isNa <- is.na(x) | x %in% na
    profile$missing[i] <- if (length(x) > 0L) round(mean(isNa), 3) else NA_real_
    profile$distinct[i] <- length(unique(x[!isNa]))
    profile$example[i] <- if (any(!isNa)) x[!isNa][1] else NA_character_
  }

  attr(profile, "rows") <- nrow(raw)
  attr(profile, "delimiter") <- delim
  return(profile)
}
