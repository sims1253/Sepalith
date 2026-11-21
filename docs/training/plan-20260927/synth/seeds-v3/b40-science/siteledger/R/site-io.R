#' Read site observations from a comma separated file
#'
#' @param path Path to the file.
#' @param site.col Name of the site identifier column.
#' @param time.col Name of the time column, formatted `"%Y-%m-%d %H:%M:%S"`.
#' @param tz Time zone of the times in the file.
#' @return A data frame sorted by site and time, with the time column
#'   parsed to `POSIXct`.
#' @rdname read.site
#' @export
read.site.csv <- function(path, site.col = "site", time.col = "time", tz = "UTC") {
  data <- utils::read.table(path, header = TRUE, sep = ",", dec = ".",
                            stringsAsFactors = FALSE, na.strings = c("", "NA"))
  missing.cols <- setdiff(c(site.col, time.col), names(data))
  if (length(missing.cols) > 0L) {
    stop(sprintf("`%s` lacks columns: %s.", basename(path),
                 paste(missing.cols, collapse = ", ")), call. = FALSE)
  }
  data[[site.col]] <- as.character(data[[site.col]])
  data[[time.col]] <- as.POSIXct(data[[time.col]], format = "%Y-%m-%d %H:%M:%S", tz = tz)
  bad.time <- is.na(data[[time.col]])
  if (any(bad.time)) {
    stop(sprintf("`%s` has %d unparseable times, first in row %d.", basename(path),
                 sum(bad.time), which(bad.time)[1]), call. = FALSE)
  }
  dup <- duplicated(data[c(site.col, time.col)])
  if (any(dup)) {
    warning(sprintf("`%s` has %d duplicated site/time rows; keeping the first.",
                    basename(path), sum(dup)), call. = FALSE)
    data <- data[!dup, , drop = FALSE]
  }

  data <- data[order(data[[site.col]], data[[time.col]]), , drop = FALSE]
  rownames(data) <- NULL
  data
}

#' @rdname read.site
#' @export
read.site.tsv <- function(path, site.col = "site", time.col = "time", tz = "UTC") {
  data <- utils::read.table(path, header = TRUE, sep = "\t", dec = ".",
                            stringsAsFactors = FALSE, na.strings = c("", "NA"))
  missing.cols <- setdiff(c(site.col, time.col), names(data))
  if (length(missing.cols) > 0L) {
    stop(sprintf("`%s` lacks columns: %s.", basename(path),
                 paste(missing.cols, collapse = ", ")), call. = FALSE)
  }
  data[[site.col]] <- as.character(data[[site.col]])
  data[[time.col]] <- as.POSIXct(data[[time.col]], format = "%Y-%m-%d %H:%M:%S", tz = tz)
  bad.time <- is.na(data[[time.col]])
  if (any(bad.time)) {
    stop(sprintf("`%s` has %d unparseable times, first in row %d.", basename(path),
                 sum(bad.time), which(bad.time)[1]), call. = FALSE)
  }
  dup <- duplicated(data[c(site.col, time.col)])
  if (any(dup)) {
    warning(sprintf("`%s` has %d duplicated site/time rows; keeping the first.",
                    basename(path), sum(dup)), call. = FALSE)
    data <- data[!dup, , drop = FALSE]
  }

  data <- data[order(data[[site.col]], data[[time.col]]), , drop = FALSE]
  rownames(data) <- NULL
  data
}

#' @rdname read.site
#' @export
read.site.semicolon <- function(path, site.col = "site", time.col = "time", tz = "UTC") {
  data <- utils::read.table(path, header = TRUE, sep = ";", dec = ",",
                            stringsAsFactors = FALSE, na.strings = c("", "NA"))
  missing.cols <- setdiff(c(site.col, time.col), names(data))
  if (length(missing.cols) > 0L) {
    stop(sprintf("`%s` lacks columns: %s.", basename(path),
                 paste(missing.cols, collapse = ", ")), call. = FALSE)
  }
  data[[site.col]] <- as.character(data[[site.col]])
  data[[time.col]] <- as.POSIXct(data[[time.col]], format = "%Y-%m-%d %H:%M:%S", tz = tz)
  bad.time <- is.na(data[[time.col]])
  if (any(bad.time)) {
    stop(sprintf("`%s` has %d unparseable times, first in row %d.", basename(path),
                 sum(bad.time), which(bad.time)[1]), call. = FALSE)
  }
  dup <- duplicated(data[c(site.col, time.col)])
  if (any(dup)) {
    warning(sprintf("`%s` has %d duplicated site/time rows; keeping the first.",
                    basename(path), sum(dup)), call. = FALSE)
    data <- data[!dup, , drop = FALSE]
  }

  data <- data[order(data[[site.col]], data[[time.col]]), , drop = FALSE]
  rownames(data) <- NULL
  data
}

#' Write site observations to a comma separated file
#'
#' @param data A data frame of observations, one row per reading.
#' @param path Path to the file to create.
#' @param time.col Name of the `POSIXct` time column.
#' @param digits Significant digits kept in numeric columns.
#' @param comment Optional lines written above the header, each prefixed
#'   with `"# "`; the readers skip them.
#' @return `path`, invisibly.
#' @rdname write.site
#' @export
write.site.csv <- function(data, path, time.col = "time", digits = 6, comment = NULL) {
  if (!is.data.frame(data) || nrow(data) == 0L) {
    stop("`data` must be a data frame with at least one row.", call. = FALSE)
  }
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)

  # Format times and round numbers before writing
  out <- data
  if (time.col %in% names(out)) {
    out[[time.col]] <- format(out[[time.col]], "%Y-%m-%d %H:%M:%S")
  }
  num <- vapply(out, is.numeric, logical(1))
  out[num] <- lapply(out[num], signif, digits = digits)

  con <- file(path, open = "w")
  on.exit(close(con))
  if (!is.null(comment)) {
    writeLines(paste("#", comment), con)
  }
  utils::write.table(out, con, sep = ",", dec = ".", row.names = FALSE,
                     quote = TRUE, na = "")
  invisible(path)
}

#' @rdname write.site
#' @export
write.site.tsv <- function(data, path, time.col = "time", digits = 6, comment = NULL) {
  if (!is.data.frame(data) || nrow(data) == 0L) {
    stop("`data` must be a data frame with at least one row.", call. = FALSE)
  }
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)

  # Format times and round numbers before writing
  out <- data
  if (time.col %in% names(out)) {
    out[[time.col]] <- format(out[[time.col]], "%Y-%m-%d %H:%M:%S")
  }
  num <- vapply(out, is.numeric, logical(1))
  out[num] <- lapply(out[num], signif, digits = digits)

  con <- file(path, open = "w")
  on.exit(close(con))
  if (!is.null(comment)) {
    writeLines(paste("#", comment), con)
  }
  utils::write.table(out, con, sep = "\t", dec = ".", row.names = FALSE,
                     quote = FALSE, na = "")
  invisible(path)
}

#' @rdname write.site
#' @export
write.site.semicolon <- function(data, path, time.col = "time", digits = 6, comment = NULL) {
  if (!is.data.frame(data) || nrow(data) == 0L) {
    stop("`data` must be a data frame with at least one row.", call. = FALSE)
  }
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)

  # Format times and round numbers before writing
  out <- data
  if (time.col %in% names(out)) {
    out[[time.col]] <- format(out[[time.col]], "%Y-%m-%d %H:%M:%S")
  }
  num <- vapply(out, is.numeric, logical(1))
  out[num] <- lapply(out[num], signif, digits = digits)

  con <- file(path, open = "w")
  on.exit(close(con))
  if (!is.null(comment)) {
    writeLines(paste("#", comment), con)
  }
  utils::write.table(out, con, sep = ";", dec = ",", row.names = FALSE,
                     quote = TRUE, na = "")
  invisible(path)
}

#' Write one file per site
#'
#' @param data A data frame of observations, one row per reading.
#' @param dir Directory to write into; created if needed.
#' @param format File format: `"csv"`, `"tsv"` or `"semicolon"`.
#' @param site.col Name of the site identifier column.
#' @return A named character vector of written paths, one per site.
#' @export
write.site.split <- function(data, dir, format = c("csv", "tsv", "semicolon"),
                             site.col = "site") {
  format <- match.arg(format)
  if (!site.col %in% names(data)) {
    stop(sprintf("`data` has no site column `%s`.", site.col), call. = FALSE)
  }
  if (anyNA(data[[site.col]])) {
    stop(sprintf("Site column `%s` must not contain missing values.", site.col), call. = FALSE)
  }
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)

  ext <- c(csv = "csv", tsv = "tsv", semicolon = "csv")[[format]]
  sites <- sort(unique(data[[site.col]]))
  paths <- file.path(dir, paste0(sites, ".", ext))
  names(paths) <- sites

  # One file per site, written by the matching writer
  for (s in sites) {
    part <- data[data[[site.col]] == s, , drop = FALSE]
    header <- sprintf("site %s, %d readings", s, nrow(part))
    switch(format,
      csv = write.site.csv(part, paths[[s]], comment = header),
      tsv = write.site.tsv(part, paths[[s]], comment = header),
      semicolon = write.site.semicolon(part, paths[[s]], comment = header)
    )
  }
  paths
}
