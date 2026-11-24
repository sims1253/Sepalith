#' Encode one ledger field for the line format
#'
#' The line format stores one record per line with fields separated by `|`.
#' Each field has its own text encoding:
#' `entry_id` is zero-padded to six digits, `posted` is written as
#' `YYYYMMDD`, `account` is upper-cased, `amount` is stored as signed whole
#' cents, `currency` is a three-letter upper-case code and `memo` is
#' percent-encoded so that it never contains the separator or a newline.
#' Missing values become empty strings.
#'
#' @param x A vector of field values.
#' @param field The field to encode.
#' @return A character vector the same length as `x`.
#' @seealso [decode_field()] for the inverse.
#' @keywords internal
encode_field <- function(x, field = c("entry_id", "posted", "account", "amount",
                                      "currency", "memo")) {
  field <- match.arg(field)
  out <- switch(field,
    entry_id = {
      sprintf("%06d", as.integer(x))
    },
    posted = {
      format(as.Date(x), "%Y%m%d")
    },
    account = {
      toupper(trimws(as.character(x)))
    },
    amount = {
      cents <- as.integer(round(as.numeric(x) * 100))
      sprintf("%+d", cents)
    },
    currency = {
      code <- toupper(trimws(as.character(x)))
      ifelse(grepl("^[A-Z]{3}$", code), code, NA_character_)
    },
    memo = {
      text <- gsub("%", "%25", as.character(x), fixed = TRUE)
      text <- gsub("|", "%7C", text, fixed = TRUE)
      gsub("\n", "%0A", text, fixed = TRUE)
    }
  )
  out[is.na(x) | is.na(out)] <- ""
  return(out)
}

#' Decode one ledger field from the line format
#'
#' Reverses [encode_field()]; empty strings become `NA`. Accounts come back
#' in lower case.
#'
#' @param x A character vector of encoded values.
#' @param field The field to decode.
#' @return A vector of the field's storage type, the same length as `x`.
#' @seealso [encode_field()] for the inverse.
#' @keywords internal
decode_field <- function(x, field = c("entry_id", "posted", "account", "amount",
                                      "currency", "memo")) {
  field <- match.arg(field)
  x[x == ""] <- NA_character_
  out <- switch(field,
    entry_id = {
      as.integer(x)
    },
    posted = {
      as.Date(x, format = "%Y%m%d")
    },
    account = {
      tolower(x)
    },
    amount = {
      cents <- as.integer(x)
      cents / 100
    },
    currency = {
      code <- toupper(x)
      ifelse(grepl("^[A-Z]{3}$", code), code, NA_character_)
    },
    memo = {
      text <- gsub("%0A", "\n", x, fixed = TRUE)
      text <- gsub("%7C", "|", text, fixed = TRUE)
      gsub("%25", "%", text, fixed = TRUE)
    }
  )
  return(out)
}

#' Encode ledger records as lines
#'
#' @param data A ledger data frame with columns `entry_id`, `posted`,
#'   `account`, `amount`, `currency` and `memo`.
#' @param sep Field separator.
#' @return A character vector with one encoded line per row.
#' @keywords internal
encode_records <- function(data, sep = "|") {
  fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")
  missing_cols <- setdiff(fields, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Column `%s` is required.", missing_cols[1]), call. = FALSE)
  }
  if (nrow(data) == 0) {
    return(character())
  }

  # Encode `entry_id`
  entry_id <- encode_field(data$entry_id, "entry_id")
  # Encode `posted`
  posted <- encode_field(data$posted, "posted")
  # Encode `account`
  account <- encode_field(data$account, "account")
  # Encode `amount`
  amount <- encode_field(data$amount, "amount")
  # Encode `currency`
  currency <- encode_field(data$currency, "currency")
  # Encode `memo`
  memo <- encode_field(data$memo, "memo")

  lines <- paste(entry_id, posted, account, amount, currency, memo, sep = sep)
  return(lines)
}

#' Decode ledger records from lines
#'
#' @param lines A character vector with one encoded line per record.
#' @param sep Field separator.
#' @return A ledger data frame with columns `entry_id`, `posted`, `account`,
#'   `amount`, `currency` and `memo`.
#' @keywords internal
decode_records <- function(lines, sep = "|") {
  fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")
  parts <- strsplit(lines, sep, fixed = TRUE)
  widths <- lengths(parts)
  # A trailing empty memo is dropped by strsplit(), so pad it back
  short <- widths == length(fields) - 1
  parts[short] <- lapply(parts[short], c, "")
  bad <- which(lengths(parts) != length(fields))
  if (length(bad) > 0) {
    stop(sprintf("Line %d has %d fields, expected %d.", bad[1], widths[bad[1]], length(fields)),
         call. = FALSE)
  }

  cells <- matrix(as.character(unlist(parts)), ncol = length(fields), byrow = TRUE)

  data <- data.frame(
    entry_id = decode_field(cells[, 1], "entry_id"),
    posted = decode_field(cells[, 2], "posted"),
    account = decode_field(cells[, 3], "account"),
    amount = decode_field(cells[, 4], "amount"),
    currency = decode_field(cells[, 5], "currency"),
    memo = decode_field(cells[, 6], "memo"),
    stringsAsFactors = FALSE
  )
  return(data)
}

#' Write a ledger in the line format
#'
#' The file starts with three header lines: `#ledgerio lines v1`,
#' `#fields: ` followed by the field names, and `#records: ` followed by the
#' number of records. One encoded record per line follows.
#'
#' @param data A ledger data frame.
#' @param path Destination file path.
#' @param sep Field separator.
#' @return `path`, invisibly.
#' @export
write_ledger_lines <- function(data, path, sep = "|") {
  fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")
  lines <- encode_records(data, sep = sep)

  # Header line 1: format and version
  version_line <- "#ledgerio lines v1"
  # Header line 2: field names
  fields_line <- paste0("#fields: ", paste(fields, collapse = sep))
  # Header line 3: record count
  records_line <- paste0("#records: ", length(lines))

  con <- file(path, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(c(version_line, fields_line, records_line, lines), con)
  return(invisible(path))
}

#' Read a ledger in the line format
#'
#' @param path Path to a file written by [write_ledger_lines()].
#' @param sep Field separator.
#' @return A ledger data frame.
#' @export
read_ledger_lines <- function(path, sep = "|") {
  fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")
  con <- file(path, open = "r", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  lines <- readLines(con)
  if (length(lines) < 3) {
    stop(sprintf("`%s` is not a ledgerio lines file.", path), call. = FALSE)
  }

  # Header line 1: format and version
  if (lines[1] != "#ledgerio lines v1") {
    stop(sprintf("`%s` is not a ledgerio lines file.", path), call. = FALSE)
  }
  # Header line 2: field names
  if (lines[2] != paste0("#fields: ", paste(fields, collapse = sep))) {
    stop(sprintf("`%s` has unexpected fields.", path), call. = FALSE)
  }
  # Header line 3: record count
  expected <- as.integer(sub("^#records: ", "", lines[3]))
  if (is.na(expected) || expected != length(lines) - 3) {
    stop(sprintf("`%s` is truncated or has extra lines.", path), call. = FALSE)
  }

  data <- decode_records(lines[-(1:3)], sep = sep)
  return(data)
}
