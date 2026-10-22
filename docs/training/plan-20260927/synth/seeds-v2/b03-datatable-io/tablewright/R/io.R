#' Read a delimited extract into a data.table
#'
#' Wraps [data.table::fread()] with the defaults used for nightly extracts:
#' empty strings become `NA`, and column names are normalised.
#'
#' @param path Path to the input file.
#' @param col_types Optional named character vector of column classes.
#' @param ... Passed to [data.table::fread()].
#' @return A data.table.
#' @export
read_extract_csv <- function(path, col_types = NULL, ...) {
  check_readable(path)
  dt <- data.table::fread(
    path,
    colClasses = col_types,
    na.strings = c("", "NA", "NULL"),
    encoding = "UTF-8",
    ...
  )

  normalise_names(dt)
}

#' Read a tab separated extract into a data.table
#'
#' Same as [read_extract_csv()] but forces a tab separator, which avoids
#' fread guessing wrong on files with very few columns.
#'
#' @inheritParams read_extract_csv
#' @return A data.table.
#' @export
read_extract_tsv <- function(path, col_types = NULL, ...) {
  check_readable(path)
  # fread's separator detection struggles with one or two columns
  dt <- data.table::fread(
    path,
    sep = "\t",
    colClasses = col_types,
    na.strings = c("", "NA", "NULL"),
    encoding = "UTF-8",
    ...
  )
  normalise_names(dt)
}

#' Read a JSON lines extract into a data.table
#'
#' Each line must hold one flat JSON object. Nested values are kept as list
#' columns.
#'
#' @inheritParams read_extract_csv
#' @return A data.table.
#' @export
read_extract_jsonl <- function(path, col_types = NULL, ...) {
  check_readable(path)
  # blank lines (often a trailing newline) are not records
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]

  records <- lapply(lines, jsonlite::fromJSON, simplifyVector = TRUE, ...)
  dt <- data.table::rbindlist(records, use.names = TRUE, fill = TRUE)

  # step_cast() converts by reference, so there is nothing to reassign
  if (!is.null(col_types)) {
    step_cast(dt, col_types)
  }
  normalise_names(dt)
}

#' Write a table to a delimited file atomically
#'
#' The table is written to a temporary file next to `path` and renamed into
#' place, so readers never see a half-written file.
#'
#' @param dt A data.table to write.
#' @param path Path to the output file.
#' @param sep Field separator.
#' @param compress Whether to gzip the output; inferred from `path` by default.
#' @return `path`, invisibly.
#' @export
write_extract <- function(dt, path, sep = ",", compress = grepl("\\.gz$", path)) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  # The temporary file must live on the same file system for rename to work
  tmp <- tempfile(tmpdir = dirname(path), fileext = if (compress) ".gz" else "")
  on.exit(unlink(tmp), add = TRUE)

  data.table::fwrite(
    dt,
    tmp,
    sep = sep,
    na = "",
    dateTimeAs = "ISO",
    compress = if (compress) "gzip" else "none"
  )

  if (!file.rename(tmp, path)) {
    stop("Could not move temporary file into place: ", path, call. = FALSE)
  }
  invisible(path)
}

#' Read many extracts and bind them
#'
#' Files are matched with a glob, read with the reader chosen by `format`, and
#' bound with a `file` column recording the source basename.
#'
#' @param dir Directory containing the extracts.
#' @param pattern Glob pattern passed to [Sys.glob()].
#' @param format One of `"csv"`, `"tsv"` or `"jsonl"`.
#' @param ... Passed to the reader.
#' @return A data.table, or an empty data.table if no files matched.
#' @export
read_extract_dir <- function(dir, pattern = "*", format = c("csv", "tsv", "jsonl"), ...) {
  format <- match.arg(format)
  reader <- switch(format,
    csv = read_extract_csv,
    tsv = read_extract_tsv,
    jsonl = read_extract_jsonl
  )

  files <- sort(Sys.glob(file.path(dir, pattern)))
  if (length(files) == 0L) {
    return(data.table::data.table())
  }

  tables <- lapply(files, reader, ...)
  names(tables) <- basename(files)
  # names(tables) become the values of the `file` column
  data.table::rbindlist(tables, use.names = TRUE, fill = TRUE, idcol = "file")
}

#' Check that a file exists and is readable
#'
#' @param path Path to the input file.
#' @noRd
check_readable <- function(path) {
  if (!file.exists(path)) {
    stop("File not found: ", path, call. = FALSE)
  }
  # file.access() returns 0 on success, -1 on failure
  if (file.access(path, mode = 4L) != 0L) {
    stop("File is not readable: ", path, call. = FALSE)
  }
  invisible(path)
}

#' Normalise column names to lower snake case
#'
#' @param dt A data.table whose names should be normalised.
#' @noRd
normalise_names <- function(dt) {
  # "Order ID" and "order-id" both become "order_id"
  nms <- tolower(names(dt))
  nms <- gsub("[^a-z0-9]+", "_", nms)
  nms <- gsub("^_+|_+$", "", nms)
  nms <- make.unique(nms, sep = "_")
  data.table::setnames(dt, nms)
  dt
}

#' Load new extracts into a consolidated file
#'
#' Keeps a small state file recording which extracts have already been loaded
#' together with their size and modification time (in whole seconds). Only new
#' or modified files are read, appended to `dest`, and recorded.
#'
#' @param src_dir Directory containing incoming extracts.
#' @param dest Path to the consolidated output file.
#' @param key Character vector of columns that together form the key.
#' @param pattern Glob pattern passed to [Sys.glob()].
#' @param state_file Path to the state file; defaults to a file next to `dest`.
#' @return A data.table describing the files loaded in this run, invisibly.
#' @export
sync_extracts <- function(src_dir, dest, key, pattern = "*.csv",
                          state_file = paste0(dest, ".state.csv")) {
  files <- sort(Sys.glob(file.path(src_dir, pattern)))
  # Whole seconds only: fractional mtimes do not survive the CSV round trip,
  # and an inexact match would reload every file on every run
  info <- data.table::data.table(
    file = basename(files),
    size = as.numeric(file.size(files)),
    mtime = floor(as.numeric(file.mtime(files)))
  )

  state <- if (file.exists(state_file)) {
    data.table::fread(state_file, colClasses = c(file = "character"))
  } else {
    data.table::data.table(file = character(), size = numeric(), mtime = numeric())
  }

  todo <- info[!state, on = c("file", "size", "mtime")]
  if (nrow(todo) == 0L) {
    message("No new extracts in ", src_dir)
    return(invisible(todo))
  }

  incoming <- lapply(file.path(src_dir, todo$file), read_extract_csv)
  incoming <- data.table::rbindlist(incoming, use.names = TRUE, fill = TRUE)
  # the first run has nothing to append to
  current <- if (file.exists(dest)) read_extract_csv(dest) else NULL
  combined <- data.table::rbindlist(list(current, incoming), use.names = TRUE, fill = TRUE)
  check_has_cols(combined, key, "sync_extracts")

  # later files win when the same key appears twice
  combined <- unique(combined, by = key, fromLast = TRUE)
  data.table::setkeyv(combined, key)
  write_extract(combined, dest)

  state <- data.table::rbindlist(list(state[!todo, on = "file"], todo), use.names = TRUE)
  write_extract(state, state_file)
  message(sprintf("Loaded %d file(s), %d row(s) total", nrow(todo), nrow(combined)))
  invisible(todo)
}
