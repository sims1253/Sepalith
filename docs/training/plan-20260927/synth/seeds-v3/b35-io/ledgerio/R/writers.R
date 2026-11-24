#' Prepare a ledger for writing
#'
#' Converts each ledger column to its on-disk text form so that every writer
#' produces the same values.
#'
#' @param data A ledger data frame.
#' @param digits Number of decimal places kept for `amount`.
#' @return A data frame of character columns `entry_id`, `posted`,
#'   `account`, `amount`, `currency` and `memo`.
#' @keywords internal
prepare_ledger_output <- function(data, digits = 2) {
  fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")
  missing_cols <- setdiff(fields, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Ledger is missing column(s): %s", paste(missing_cols, collapse = ", ")),
         call. = FALSE)
  }
  out <- data.frame(row.names = seq_len(nrow(data)))

  # Encode `entry_id`
  out$entry_id <- format(as.integer(data$entry_id), scientific = FALSE, trim = TRUE)
  # Encode `posted`
  out$posted <- format(as.Date(data$posted), "%Y-%m-%d")
  # Encode `account`
  out$account <- trimws(as.character(data$account))
  # Encode `amount`
  out$amount <- formatC(as.numeric(data$amount), format = "f", digits = digits)
  # Encode `currency`
  out$currency <- toupper(trimws(as.character(data$currency)))
  # Encode `memo`
  out$memo <- gsub("[\r\n]+", " ", as.character(data$memo))

  rownames(out) <- NULL
  return(out)
}

#' Write a ledger to a comma-separated file
#'
#' @param data A ledger data frame.
#' @param path Destination file path.
#' @param overwrite Whether an existing file may be replaced.
#' @return `path`, invisibly.
#' @family ledger writers
#' @export
write_ledger_csv <- function(data, path, overwrite = FALSE) {
  # 1. Check the destination
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Ledger file `%s` already exists.", path), call. = FALSE)
  }
  if (nrow(data) == 0) {
    warning(sprintf("Writing an empty ledger to `%s`.", path), call. = FALSE)
  }

  # 2. Encode every column as text
  out <- prepare_ledger_output(data)
  # 3. Write to a temporary file so a failed write leaves `path` intact
  tmp <- paste0(path, ".part")
  on.exit(unlink(tmp), add = TRUE)
  utils::write.csv(out, tmp, row.names = FALSE)

  # 4. Move the finished file into place
  if (!file.rename(tmp, path)) {
    stop(sprintf("Could not move `%s` to `%s`.", tmp, path), call. = FALSE)
  }
  return(invisible(path))
}

#' Write a ledger to a tab-separated file
#'
#' @inheritParams write_ledger_csv
#' @return `path`, invisibly.
#' @family ledger writers
#' @export
write_ledger_tsv <- function(data, path, overwrite = FALSE) {
  # 1. Check the destination
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Ledger file `%s` already exists.", path), call. = FALSE)
  }
  if (nrow(data) == 0) {
    warning(sprintf("Writing an empty ledger to `%s`.", path), call. = FALSE)
  }

  # 2. Encode every column as text
  out <- prepare_ledger_output(data)
  # 3. Write to a temporary file so a failed write leaves `path` intact
  tmp <- paste0(path, ".part")
  on.exit(unlink(tmp), add = TRUE)
  utils::write.table(out, tmp, sep = "\t", quote = FALSE, row.names = FALSE)

  # 4. Move the finished file into place
  if (!file.rename(tmp, path)) {
    stop(sprintf("Could not move `%s` to `%s`.", tmp, path), call. = FALSE)
  }
  return(invisible(path))
}

#' Write a ledger to an RDS file
#'
#' @inheritParams write_ledger_csv
#' @return `path`, invisibly.
#' @family ledger writers
#' @export
write_ledger_rds <- function(data, path, overwrite = FALSE) {
  # 1. Check the destination
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Ledger file `%s` already exists.", path), call. = FALSE)
  }
  if (nrow(data) == 0) {
    warning(sprintf("Writing an empty ledger to `%s`.", path), call. = FALSE)
  }

  # 2. Encode every column as text
  out <- prepare_ledger_output(data)
  # 3. Write to a temporary file so a failed write leaves `path` intact
  tmp <- paste0(path, ".part")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(out, tmp)

  # 4. Move the finished file into place
  if (!file.rename(tmp, path)) {
    stop(sprintf("Could not move `%s` to `%s`.", tmp, path), call. = FALSE)
  }
  return(invisible(path))
}

#' Write a ledger to a DCF file
#'
#' @inheritParams write_ledger_csv
#' @return `path`, invisibly.
#' @family ledger writers
#' @export
write_ledger_dcf <- function(data, path, overwrite = FALSE) {
  # 1. Check the destination
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Ledger file `%s` already exists.", path), call. = FALSE)
  }
  if (nrow(data) == 0) {
    warning(sprintf("Writing an empty ledger to `%s`.", path), call. = FALSE)
  }

  # 2. Encode every column as text
  out <- prepare_ledger_output(data)
  # 3. Write to a temporary file so a failed write leaves `path` intact
  tmp <- paste0(path, ".part")
  on.exit(unlink(tmp), add = TRUE)
  write.dcf(out, tmp, keep.white = "memo")

  # 4. Move the finished file into place
  if (!file.rename(tmp, path)) {
    stop(sprintf("Could not move `%s` to `%s`.", tmp, path), call. = FALSE)
  }
  return(invisible(path))
}

#' Write a ledger in any supported format
#'
#' @param data A ledger data frame.
#' @param path Destination file path.
#' @param format One of `"csv"`, `"tsv"`, `"rds"` or `"dcf"`.
#' @param overwrite Whether an existing file may be replaced.
#' @return `path`, invisibly.
#' @export
write_ledger <- function(data, path, format = c("csv", "tsv", "rds", "dcf"),
                         overwrite = FALSE) {
  format <- match.arg(format)
  dir <- dirname(path)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  switch(format,
    csv = {
      write_ledger_csv(data, path, overwrite = overwrite)
    },
    tsv = {
      write_ledger_tsv(data, path, overwrite = overwrite)
    },
    rds = {
      write_ledger_rds(data, path, overwrite = overwrite)
    },
    dcf = {
      write_ledger_dcf(data, path, overwrite = overwrite)
    }
  )
  return(invisible(path))
}

#' Write one ledger file per account
#'
#' Splits the ledger by `account` and writes each part to
#' `<dir>/<account>.<format>`, replacing characters that are unsafe in file
#' names with `_`.
#'
#' @param data A ledger data frame.
#' @param dir Output directory, created if needed.
#' @param format One of `"csv"`, `"tsv"`, `"rds"` or `"dcf"`.
#' @param overwrite Whether existing files may be replaced.
#' @return A data frame with columns `account`, `file`, `rows` and `total`
#'   (sum of `amount`), one row per account.
#' @export
write_ledger_split <- function(data, dir, format = c("csv", "tsv", "rds", "dcf"),
                               overwrite = FALSE) {
  format <- match.arg(format)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  accounts <- sort(unique(trimws(as.character(data$account))))
  files <- paste0(gsub("[^A-Za-z0-9_-]", "_", accounts), ".", format)
  if (anyDuplicated(files) > 0) {
    stop(sprintf("Accounts map to the same file name: %s",
                 paste(unique(files[duplicated(files)]), collapse = ", ")), call. = FALSE)
  }

  rows <- integer(length(accounts))
  total <- numeric(length(accounts))
  for (i in seq_along(accounts)) {
    part <- data[trimws(data$account) == accounts[i], , drop = FALSE]
    write_ledger(part, file.path(dir, files[i]), format = format, overwrite = overwrite)
    rows[i] <- nrow(part)
    total[i] <- sum(part$amount)
  }
  return(data.frame(account = accounts, file = files, rows = rows, total = total,
                    stringsAsFactors = FALSE))
}
