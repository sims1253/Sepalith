#' Coerce ledger columns to their declared types
#'
#' Every reader returns text columns; this step converts them one column at
#' a time and fails on the first column that holds values which cannot be
#' converted.
#'
#' @param data A data frame of raw ledger columns.
#' @param path Path the data was read from, used in error messages.
#' @return `data` restricted to the columns `entry_id` (integer), `posted`
#'   (Date), `account` (character), `amount` (numeric), `currency`
#'   (character) and `memo` (character), in that order.
#' @keywords internal
coerce_ledger_columns <- function(data, path) {
  fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")
  missing_cols <- setdiff(fields, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("`%s` is missing column(s): %s", path,
                 paste(missing_cols, collapse = ", ")), call. = FALSE)
  }

  # Coerce `entry_id`
  data$entry_id <- suppressWarnings(as.integer(data$entry_id))
  if (anyNA(data$entry_id)) {
    stop(sprintf("`%s`: column `%s` has invalid values.", path, "entry_id"), call. = FALSE)
  }
  # Coerce `posted`
  data$posted <- as.Date(data$posted, format = "%Y-%m-%d")
  if (anyNA(data$posted)) {
    stop(sprintf("`%s`: column `%s` has invalid values.", path, "posted"), call. = FALSE)
  }
  # Coerce `account`
  data$account <- as.character(data$account)
  if (anyNA(data$account)) {
    stop(sprintf("`%s`: column `%s` has invalid values.", path, "account"), call. = FALSE)
  }
  # Coerce `amount`
  data$amount <- suppressWarnings(as.numeric(data$amount))
  if (anyNA(data$amount)) {
    stop(sprintf("`%s`: column `%s` has invalid values.", path, "amount"), call. = FALSE)
  }
  # Coerce `currency`
  data$currency <- toupper(as.character(data$currency))
  if (anyNA(data$currency)) {
    stop(sprintf("`%s`: column `%s` has invalid values.", path, "currency"), call. = FALSE)
  }
  # Coerce `memo`
  data$memo <- as.character(data$memo)
  if (anyNA(data$memo)) {
    stop(sprintf("`%s`: column `%s` has invalid values.", path, "memo"), call. = FALSE)
  }

  rownames(data) <- NULL
  return(data[fields])
}

#' Read a ledger from a comma-separated file
#'
#' @param path Path to the ledger file.
#' @param encoding Encoding of the file, passed on to the reader.
#' @param sort Whether to order the rows by `posted` and `entry_id`.
#' @return A ledger data frame with attributes `source` and `format`.
#' @family ledger readers
#' @export
read_ledger_csv <- function(path, encoding = "UTF-8", sort = TRUE) {
  # 1. Check the file
  if (!file.exists(path)) {
    stop(sprintf("Ledger file `%s` does not exist.", path), call. = FALSE)
  }
  if (file.size(path) == 0) {
    stop(sprintf("Ledger file `%s` is empty.", path), call. = FALSE)
  }

  # 2. Read every column as text
  data <- utils::read.csv(path, colClasses = "character", na.strings = character(),
                          encoding = encoding, stringsAsFactors = FALSE)
  # 3. Coerce to the ledger types
  data <- coerce_ledger_columns(data, path)
  # 4. Order by posting date
  if (sort) {
    data <- data[order(data$posted, data$entry_id), , drop = FALSE]
    rownames(data) <- NULL
  }

  # 5. Record where the rows came from
  attr(data, "source") <- path
  attr(data, "format") <- "csv"
  return(data)
}

#' Read a ledger from a tab-separated file
#'
#' @inheritParams read_ledger_csv
#' @return A ledger data frame with attributes `source` and `format`.
#' @family ledger readers
#' @export
read_ledger_tsv <- function(path, encoding = "UTF-8", sort = TRUE) {
  # 1. Check the file
  if (!file.exists(path)) {
    stop(sprintf("Ledger file `%s` does not exist.", path), call. = FALSE)
  }
  if (file.size(path) == 0) {
    stop(sprintf("Ledger file `%s` is empty.", path), call. = FALSE)
  }

  # 2. Read every column as text
  data <- utils::read.delim(path, colClasses = "character", na.strings = character(),
                            encoding = encoding, stringsAsFactors = FALSE)
  # 3. Coerce to the ledger types
  data <- coerce_ledger_columns(data, path)
  # 4. Order by posting date
  if (sort) {
    data <- data[order(data$posted, data$entry_id), , drop = FALSE]
    rownames(data) <- NULL
  }

  # 5. Record where the rows came from
  attr(data, "source") <- path
  attr(data, "format") <- "tsv"
  return(data)
}

#' Read a ledger from an RDS file
#'
#' @inheritParams read_ledger_csv
#' @return A ledger data frame with attributes `source` and `format`.
#' @family ledger readers
#' @export
read_ledger_rds <- function(path, encoding = "UTF-8", sort = TRUE) {
  # 1. Check the file
  if (!file.exists(path)) {
    stop(sprintf("Ledger file `%s` does not exist.", path), call. = FALSE)
  }
  if (file.size(path) == 0) {
    stop(sprintf("Ledger file `%s` is empty.", path), call. = FALSE)
  }

  # 2. Read the stored data frame
  data <- readRDS(path)
  if (!is.data.frame(data)) {
    stop(sprintf("`%s` does not contain a data frame.", path), call. = FALSE)
  }
  # 3. Coerce to the ledger types
  data <- coerce_ledger_columns(data, path)
  # 4. Order by posting date
  if (sort) {
    data <- data[order(data$posted, data$entry_id), , drop = FALSE]
    rownames(data) <- NULL
  }

  # 5. Record where the rows came from
  attr(data, "source") <- path
  attr(data, "format") <- "rds"
  return(data)
}

#' Read a ledger from a DCF file
#'
#' Each record is one paragraph of `field: value` lines.
#'
#' @inheritParams read_ledger_csv
#' @return A ledger data frame with attributes `source` and `format`.
#' @family ledger readers
#' @export
read_ledger_dcf <- function(path, encoding = "UTF-8", sort = TRUE) {
  # 1. Check the file
  if (!file.exists(path)) {
    stop(sprintf("Ledger file `%s` does not exist.", path), call. = FALSE)
  }
  if (file.size(path) == 0) {
    stop(sprintf("Ledger file `%s` is empty.", path), call. = FALSE)
  }

  # 2. Read every field as text
  records <- read.dcf(path, all = TRUE, keep.white = "memo")
  data <- as.data.frame(records, stringsAsFactors = FALSE)
  # 3. Coerce to the ledger types
  data <- coerce_ledger_columns(data, path)
  # 4. Order by posting date
  if (sort) {
    data <- data[order(data$posted, data$entry_id), , drop = FALSE]
    rownames(data) <- NULL
  }

  # 5. Record where the rows came from
  attr(data, "source") <- path
  attr(data, "format") <- "dcf"
  return(data)
}

#' Read a ledger file in any supported format
#'
#' With `format = "auto"` the format is taken from the file extension.
#'
#' @param path Path to the ledger file.
#' @param format One of `"auto"`, `"csv"`, `"tsv"`, `"rds"` or `"dcf"`.
#' @param ... Passed to the format-specific reader.
#' @return A ledger data frame.
#' @export
read_ledger <- function(path, format = c("auto", "csv", "tsv", "rds", "dcf"), ...) {
  format <- match.arg(format)
  if (format == "auto") {
    format <- tolower(tools::file_ext(path))
    if (!format %in% c("csv", "tsv", "rds", "dcf")) {
      stop(sprintf("Cannot guess the ledger format of `%s`.", path), call. = FALSE)
    }
  }

  switch(format,
    csv = {
      data <- read_ledger_csv(path, ...)
    },
    tsv = {
      data <- read_ledger_tsv(path, ...)
    },
    rds = {
      data <- read_ledger_rds(path, ...)
    },
    dcf = {
      data <- read_ledger_dcf(path, ...)
    }
  )
  return(data)
}

#' Read every ledger file in a directory
#'
#' @param dir Directory to scan.
#' @param pattern Regular expression the file names must match.
#' @return A single ledger data frame with an extra `file` column naming the
#'   source file of each row, ordered by `posted` and `entry_id`. The
#'   attribute `files` holds a data frame with columns `file`, `format` and
#'   `rows`, one row per file read.
#' @export
read_ledger_dir <- function(dir, pattern = "\\.(csv|tsv|rds|dcf)$") {
  paths <- sort(list.files(dir, pattern = pattern, full.names = TRUE))
  if (length(paths) == 0) {
    stop(sprintf("No ledger files found in `%s`.", dir), call. = FALSE)
  }
  parts <- vector("list", length(paths))
  formats <- character(length(paths))
  rows <- integer(length(paths))
  for (i in seq_along(paths)) {
    part <- read_ledger(paths[[i]], sort = FALSE)
    part$file <- basename(paths[[i]])
    parts[[i]] <- part
    formats[i] <- attr(part, "format")
    rows[i] <- nrow(part)
  }

  data <- do.call(rbind, parts)
  data <- data[order(data$posted, data$entry_id), , drop = FALSE]
  rownames(data) <- NULL
  attr(data, "files") <- data.frame(file = basename(paths), format = formats, rows = rows,
                                    stringsAsFactors = FALSE)
  return(data)
}
