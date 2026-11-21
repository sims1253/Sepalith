#' Describe one column for the codebook
#'
#' @description The summary depends on the column type: a range for
#'   `integer`, `numeric` and `Date` columns, the number of distinct values
#'   for `character` columns and the counts of `TRUE` and `FALSE` for
#'   `logical` columns.
#'
#' @param x A column vector.
#' @param name The column name.
#' @return A list with elements `name`, `type`, `missing` and `summary`.
#' @keywords internal
codebook.entry <- function(x, name) {
  type <- class(x)[1]
  missing <- sum(is.na(x))
  if (missing == length(x)) {
    return(list(name = name, type = type, missing = missing, summary = "all missing"))
  }

  summary <- switch(type,
    integer = {
      rng <- range(x, na.rm = TRUE)
      sprintf("%d to %d", rng[1], rng[2])
    },
    numeric = {
      rng <- range(x, na.rm = TRUE)
      sprintf("%g to %g", rng[1], rng[2])
    },
    Date = {
      rng <- range(x, na.rm = TRUE)
      sprintf("%s to %s", format(rng[1]), format(rng[2]))
    },
    character = {
      n <- length(unique(x[!is.na(x)]))
      sprintf("%d distinct values", n)
    },
    logical = {
      sprintf("%d TRUE, %d FALSE", sum(x, na.rm = TRUE), sum(!x, na.rm = TRUE))
    },
    stop(sprintf("Column `%s` has unsupported type `%s`.", name, type), call. = FALSE)
  )
  list(name = name, type = type, missing = missing, summary = summary)
}

#' Build the codebook of a table
#'
#' @param data A data frame to describe.
#' @param table The table name recorded in the codebook.
#' @return A data frame with columns `table`, `name`, `type`, `missing` and
#'   `summary`, one row per column of `data`.
#' @export
codebook.table <- function(data, table) {
  entries <- vector("list", ncol(data))
  for (i in seq_along(data)) {
    entries[[i]] <- codebook.entry(data[[i]], names(data)[i])
  }

  codebook <- data.frame(
    table = rep(table, length(entries)),
    name = vapply(entries, `[[`, character(1), "name"),
    type = vapply(entries, `[[`, character(1), "type"),
    missing = vapply(entries, `[[`, integer(1), "missing"),
    summary = vapply(entries, `[[`, character(1), "summary"),
    stringsAsFactors = FALSE
  )
  codebook
}

#' Write a codebook
#'
#' @param codebook A codebook data frame from `codebook.table()`.
#' @param path Destination file path.
#' @param format `"dcf"` (one paragraph per column), `"csv"`, or `"txt"`
#'   (aligned plain text for reading).
#' @return `path`, invisibly.
#' @export
write.codebook <- function(codebook, path, format = c("dcf", "csv", "txt")) {
  format <- match.arg(format)

  switch(format,
    dcf = {
      write.dcf(codebook, path)
    },
    csv = {
      utils::write.csv(codebook, path, row.names = FALSE)
    },
    txt = {
      name.width <- max(nchar(codebook$name), nchar("name"))
      type.width <- max(nchar(codebook$type), nchar("type"))
      lines <- sprintf("%-*s  %-*s  %7s  %s", name.width, codebook$name, type.width,
                       codebook$type, codebook$missing, codebook$summary)
      header <- sprintf("%-*s  %-*s  %7s  %s", name.width, "name", type.width, "type",
                        "missing", "summary")
      writeLines(c(header, lines), path)
    }
  )
  invisible(path)
}

#' Read a codebook
#'
#' @param path Path to a codebook written by `write.codebook()`.
#' @param format `"dcf"` or `"csv"`; plain-text codebooks cannot be read
#'   back.
#' @return A codebook data frame.
#' @export
read.codebook <- function(path, format = c("dcf", "csv")) {
  format <- match.arg(format)
  codebook <- switch(format,
    dcf = {
      as.data.frame(read.dcf(path), stringsAsFactors = FALSE)
    },
    csv = {
      utils::read.csv(path, colClasses = "character", stringsAsFactors = FALSE)
    }
  )

  required <- c("table", "name", "type", "missing", "summary")
  absent <- setdiff(required, names(codebook))
  if (length(absent) > 0) {
    stop(sprintf("Codebook `%s` is missing field(s): %s", path,
                 paste(absent, collapse = ", ")), call. = FALSE)
  }
  codebook$missing <- as.integer(codebook$missing)
  codebook[required]
}

#' Compare a table against its codebook
#'
#' @param codebook A codebook data frame from `codebook.table()`.
#' @param data The data frame to check.
#' @return A list with character vectors `missing` (codebook columns absent
#'   from `data`), `extra` (columns of `data` not in the codebook), `retyped`
#'   (columns whose type changed) and `more.missing` (columns with more
#'   missing values than the codebook records).
#' @export
codebook.check <- function(codebook, data) {
  # Columns in the codebook but not in `data`
  missing <- setdiff(codebook$name, names(data))
  # Columns in `data` but not in the codebook
  extra <- setdiff(names(data), codebook$name)
  # Columns in both whose type changed
  common <- intersect(codebook$name, names(data))
  old.type <- codebook$type[match(common, codebook$name)]
  new.type <- vapply(data[common], function(col) class(col)[1], character(1))
  retyped <- common[old.type != new.type]
  # Columns in both with more missing values than recorded
  old.missing <- codebook$missing[match(common, codebook$name)]
  new.missing <- vapply(data[common], function(col) sum(is.na(col)), integer(1))
  more.missing <- common[new.missing > old.missing]
  list(missing = missing, extra = extra, retyped = retyped, more.missing = more.missing)
}
