#' @title Read a delimited table
#' @description Column names are trimmed and made unique after reading.
#' @param path Path to the file.
#' @param format One of `"csv"`, `"tsv"` or `"psv"` (pipe-separated).
#' @param colClasses Optional column classes passed to [data.table::fread()].
#' @param naStrings Strings read as missing values.
#' @return A `data.table` with attribute `source`.
#' @importFrom data.table fread fwrite
#' @export
readTable = function(path, format = c("csv", "tsv", "psv"), colClasses = NULL,
                     naStrings = c("", "NA")) {
  format = match.arg(format)
  if (!file.exists(path)) {
    stop(sprintf("Table file `%s` does not exist.", path), call. = FALSE)
  }
  data = switch(format,
    csv = {
      data.table::fread(path, sep = ",", colClasses = colClasses, na.strings = naStrings,
                        encoding = "UTF-8")
    },
    tsv = {
      data.table::fread(path, sep = "\t", colClasses = colClasses, na.strings = naStrings,
                        encoding = "UTF-8")
    },
    psv = {
      data.table::fread(path, sep = "|", colClasses = colClasses, na.strings = naStrings,
                        encoding = "UTF-8")
    }
  )

  data.table::setnames(data, make.unique(trimws(names(data))))
  data.table::setattr(data, "source", normalizePath(path))
  data
}

#' @title Write a delimited table
#' @param data A data frame or `data.table`.
#' @param path Destination file path.
#' @param format One of `"csv"`, `"tsv"` or `"psv"` (pipe-separated).
#' @param na String written for missing values.
#' @return `path`, invisibly.
#' @export
writeTable = function(data, path, format = c("csv", "tsv", "psv"), na = "") {
  format = match.arg(format)
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  if (!dir.exists(dirname(path))) {
    dir.create(dirname(path), recursive = TRUE)
  }

  switch(format,
    csv = {
      data.table::fwrite(data, path, sep = ",", na = na, dateTimeAs = "ISO")
    },
    tsv = {
      data.table::fwrite(data, path, sep = "\t", na = na, dateTimeAs = "ISO")
    },
    psv = {
      data.table::fwrite(data, path, sep = "|", na = na, dateTimeAs = "ISO")
    }
  )
  invisible(path)
}

#' @title Append rows to a delimited table
#' @description Writes a header only when the file does not exist yet; when
#'   it does, the column names of `data` must match the existing header.
#' @param data A data frame or `data.table`.
#' @param path Destination file path.
#' @param format One of `"csv"`, `"tsv"` or `"psv"` (pipe-separated).
#' @return The number of rows appended, invisibly.
#' @export
appendTable = function(data, path, format = c("csv", "tsv", "psv")) {
  format = match.arg(format)
  seps = c(csv = ",", tsv = "\t", psv = "|")
  if (nrow(data) == 0) {
    return(invisible(0L))
  }
  if (!file.exists(path)) {
    data.table::fwrite(data, path, sep = seps[[format]], dateTimeAs = "ISO")
    return(invisible(nrow(data)))
  }

  header = names(data.table::fread(path, sep = seps[[format]], nrows = 0))
  if (!identical(header, names(data))) {
    stop(sprintf("Columns of `data` do not match the header of `%s`.", path), call. = FALSE)
  }
  data.table::fwrite(data, path, sep = seps[[format]], append = TRUE, dateTimeAs = "ISO")
  invisible(nrow(data))
}

#' @title Cast table columns by type
#' @param data A `data.table`, modified by reference.
#' @param integerCols,numericCols,dateCols,logicalCols Names of the columns
#'   to convert to each type.
#' @return `data`, invisibly.
#' @export
castColumns = function(data, integerCols = character(), numericCols = character(),
                       dateCols = character(), logicalCols = character()) {
  wanted = c(integerCols, numericCols, dateCols, logicalCols)
  unknown = setdiff(wanted, names(data))
  if (length(unknown) > 0) {
    stop(sprintf("Unknown column(s): %s", paste(unknown, collapse = ", ")), call. = FALSE)
  }
  # Cast `integerCols`
  for (col in integerCols) {
    data.table::set(data, j = col, value = as.integer(data[[col]]))
  }
  # Cast `numericCols`
  for (col in numericCols) {
    data.table::set(data, j = col, value = as.numeric(data[[col]]))
  }
  # Cast `dateCols`
  for (col in dateCols) {
    data.table::set(data, j = col, value = data.table::as.IDate(data[[col]]))
  }
  # Cast `logicalCols`
  for (col in logicalCols) {
    data.table::set(data, j = col, value = as.logical(data[[col]]))
  }
  invisible(data)
}

#' @title Read several tables into one
#' @param paths Character vector of file paths, all in the same format.
#' @param format One of `"csv"`, `"tsv"` or `"psv"` (pipe-separated).
#' @param idcol Name of the column recording each row's source file.
#' @return A single `data.table`, with attribute `rows` giving the number of
#'   rows read from each file.
#' @export
readTableSet = function(paths, format = c("csv", "tsv", "psv"), idcol = "source") {
  format = match.arg(format)
  if (length(paths) == 0) {
    stop("`paths` must name at least one file.", call. = FALSE)
  }
  missingFiles = paths[!file.exists(paths)]
  if (length(missingFiles) > 0) {
    stop(sprintf("Table file `%s` does not exist.", missingFiles[1]), call. = FALSE)
  }

  parts = vector("list", length(paths))
  names(parts) = basename(paths)
  for (i in seq_along(paths)) {
    parts[[i]] = readTable(paths[i], format = format)
  }

  data = data.table::rbindlist(parts, use.names = TRUE, fill = TRUE, idcol = idcol)
  data.table::setattr(data, "rows", vapply(parts, nrow, integer(1)))
  data
}

#' @title Export a set of tables
#' @description Writes each element of `tables` to `<dir>/<name>.<format>`
#'   and lists every written file.
#' @param tables A named list of data frames.
#' @param dir Output directory, created if needed.
#' @param format One of `"csv"`, `"tsv"` or `"psv"` (pipe-separated).
#' @return A `data.table` with columns `table`, `file`, `rows`, `cols`,
#'   `bytes` and `md5`, one row per table.
#' @export
exportTables = function(tables, dir, format = c("csv", "tsv", "psv")) {
  format = match.arg(format)
  if (is.null(names(tables)) || any(names(tables) == "")) {
    stop("`tables` must be a fully named list.", call. = FALSE)
  }
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }

  files = file.path(dir, paste0(names(tables), ".", format))
  for (i in seq_along(tables)) {
    writeTable(tables[[i]], files[i], format = format)
  }
  data.table::data.table(
    table = names(tables),
    file = basename(files),
    rows = vapply(tables, nrow, integer(1)),
    cols = vapply(tables, ncol, integer(1)),
    bytes = file.size(files),
    md5 = unname(tools::md5sum(files))
  )
}
