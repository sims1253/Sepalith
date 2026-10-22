#' Parse access log lines in combined format
#'
#' Lines that do not match the pattern are dropped and counted in the
#' `"skipped"` attribute of the result.
#'
#' @param lines Character vector of raw log lines.
#' @return A data frame with one row per parsed line.
#' @family log parsers
#' @export
parseAccessLines <- function(lines) {
  pattern <- paste0(
    '^(\\S+) \\S+ (\\S+) \\[([^]]+)\\] "(\\S+) (\\S+) [^"]*" ',
    '(\\d{3}) (\\d+|-)(?: "([^"]*)" "([^"]*)")?'
  )
  m <- regmatches(lines, regexec(pattern, lines, perl = TRUE))
  ok <- lengths(m) > 0L
  parts <- do.call(rbind, m[ok])
  if (is.null(parts)) {
    parts <- matrix(character(0), ncol = 10L)
  }
  out <- data.frame(
    host = parts[, 2],
    user = parts[, 3],
    time = as.POSIXct(parts[, 4], format = "%d/%b/%Y:%H:%M:%S %z", tz = "UTC"),
    method = parts[, 5],
    path = parts[, 6],
    status = as.integer(parts[, 7]),
    bytes = suppressWarnings(as.integer(parts[, 8])),
    stringsAsFactors = FALSE
  )
  attr(out, "skipped") <- sum(!ok)
  return(out)
}

#' Parse syslog lines in RFC 3164 style
#'
#' Lines that do not match the pattern are dropped and counted in the
#' `"skipped"` attribute of the result.
#'
#' @param lines Character vector of raw log lines.
#' @param year Year to assume, since RFC 3164 timestamps omit it.
#' @return A data frame with one row per parsed line.
#' @family log parsers
#' @export
parseSyslogLines <- function(lines, year = format(Sys.Date(), "%Y")) {
  pattern <- "^(\\w{3} [ \\d]\\d \\d{2}:\\d{2}:\\d{2}) (\\S+) ([^:\\[]+)(?:\\[(\\d+)\\])?: (.*)$"
  m <- regmatches(lines, regexec(pattern, lines, perl = TRUE))
  ok <- lengths(m) > 0L
  parts <- do.call(rbind, m[ok])
  if (is.null(parts)) {
    parts <- matrix(character(0), ncol = 6L)
  }
  stamp <- paste(year, gsub("\\s+", " ", parts[, 2]))
  out <- data.frame(
    time = as.POSIXct(stamp, format = "%Y %b %d %H:%M:%S", tz = "UTC"),
    host = parts[, 3],
    program = trimws(parts[, 4]),
    pid = suppressWarnings(as.integer(parts[, 5])),
    message = parts[, 6],
    stringsAsFactors = FALSE
  )
  attr(out, "skipped") <- sum(!ok)
  return(out)
}

#' Parse application log lines with a level prefix
#'
#' Expects lines like `2026-03-01T10:22:01Z WARN [worker-3] queue is full`.
#' Lines that do not match are dropped and counted in the `"skipped"` attribute.
#'
#' @param lines Character vector of raw log lines.
#' @return A data frame with one row per parsed line.
#' @family log parsers
#' @export
parseAppLines <- function(lines) {
  pattern <- "^(\\S+)\\s+(TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\\s+\\[([^]]+)\\]\\s+(.*)$"
  m <- regmatches(lines, regexec(pattern, lines, perl = TRUE))
  ok <- lengths(m) > 0L
  parts <- do.call(rbind, m[ok])
  if (is.null(parts)) {
    parts <- matrix(character(0), ncol = 5L)
  }
  out <- data.frame(
    time = as.POSIXct(parts[, 2], format = "%Y-%m-%dT%H:%M:%OSZ", tz = "UTC"),
    level = factor(parts[, 3], levels = logLevels()),
    thread = parts[, 4],
    message = parts[, 5],
    stringsAsFactors = FALSE
  )
  attr(out, "skipped") <- sum(!ok)
  return(out)
}

#' Read a log file and parse it with the chosen parser
#'
#' Gzipped files are read transparently. Very large files can be limited with
#' `maxLines`.
#'
#' @param path Path to the log file.
#' @param format Log format, selecting one of the parsers.
#' @param maxLines Maximum number of lines to read; `-1` reads everything.
#' @return A data frame as returned by the selected parser.
#' @seealso [parseAccessLines()], [parseSyslogLines()], [parseAppLines()]
#' @export
readLogFile <- function(path, format = c("access", "syslog", "app"), maxLines = -1L) {
  format <- match.arg(format)
  if (!file.exists(path)) {
    stop("Log file not found: ", path, call. = FALSE)
  }
  con <- if (grepl("\\.gz$", path)) gzfile(path, "r") else file(path, "r")
  on.exit(close(con), add = TRUE)
  lines <- readLines(con, n = maxLines, warn = FALSE)
  parsed <- switch(format,
    access = parseAccessLines(lines),
    syslog = parseSyslogLines(lines),
    app = parseAppLines(lines)
  )
  skipped <- attr(parsed, "skipped")
  if (skipped > 0L) {
    message(sprintf("Skipped %d unparseable line(s) in %s", skipped, basename(path)))
  }
  return(parsed)
}

#' Ordered log levels from least to most severe
#'
#' @return A character vector of level names.
#' @keywords internal
logLevels <- function() {
  return(c("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"))
}

#' Keep log records at or above a severity level
#'
#' @param logs A data frame of parsed application log records.
#' @param minLevel Lowest level to keep.
#' @return The filtered data frame.
#' @export
filterByLevel <- function(logs, minLevel = "WARN") {
  levels <- logLevels()
  if (!minLevel %in% levels) {
    stop("`minLevel` must be one of: ", paste(levels, collapse = ", "), call. = FALSE)
  }
  rank <- match(as.character(logs$level), levels)
  keep <- !is.na(rank) & rank >= match(minLevel, levels)
  return(logs[keep, , drop = FALSE])
}

#' Count log records per time bucket and level
#'
#' @param logs A data frame of parsed application log records.
#' @param unit Width of each time bucket.
#' @return A data frame with columns `bucket`, `level` and `n`.
#' @export
countByInterval <- function(logs, unit = c("hour", "minute", "day")) {
  unit <- match.arg(unit)
  fmt <- switch(unit,
    minute = "%Y-%m-%d %H:%M:00",
    hour = "%Y-%m-%d %H:00:00",
    day = "%Y-%m-%d 00:00:00"
  )
  bucket <- format(logs$time, fmt, tz = "UTC")
  counts <- as.data.frame(table(bucket = bucket, level = logs$level), stringsAsFactors = FALSE)
  names(counts)[3] <- "n"
  counts <- counts[counts$n > 0L, , drop = FALSE]
  counts$bucket <- as.POSIXct(counts$bucket, tz = "UTC")
  return(counts[order(counts$bucket, counts$level), , drop = FALSE])
}

#' Group similar log messages into templates and count them
#'
#' Variable parts of each message (numbers, hex ids, quoted strings, paths)
#' are replaced by placeholders so that repeated errors collapse into a single
#' template with a count and the first and last time it was seen.
#'
#' @param logs A data frame of parsed application log records.
#' @param minLevel Lowest level to keep.
#' @param topN Maximum number of templates to return.
#' @return A data frame of templates ordered by decreasing count.
#' @export
summariseMessages <- function(logs, minLevel = "WARN", topN = 20L) {
  logs <- filterByLevel(logs, minLevel)
  if (nrow(logs) == 0L) {
    return(data.frame(
      template = character(0),
      level = character(0),
      n = integer(0),
      first = as.POSIXct(character(0), tz = "UTC"),
      last = as.POSIXct(character(0), tz = "UTC")
    ))
  }
  tpl <- logs$message
  tpl <- gsub("\"[^\"]*\"|'[^']*'", "<str>", tpl)
  tpl <- gsub("(/[A-Za-z0-9._-]+){2,}", "<path>", tpl)
  tpl <- gsub("\\b0x[0-9a-fA-F]+\\b|\\b[0-9a-f]{8,}\\b", "<hex>", tpl)
  tpl <- gsub("\\b[0-9]+(\\.[0-9]+)?\\b", "<num>", tpl)
  key <- paste(logs$level, tpl, sep = " :: ")
  groups <- split(seq_len(nrow(logs)), key)
  out <- data.frame(
    template = vapply(groups, function(i) tpl[i[1]], character(1)),
    level = vapply(groups, function(i) as.character(logs$level[i[1]]), character(1)),
    n = lengths(groups),
    first = do.call(c, lapply(groups, function(i) min(logs$time[i]))),
    last = do.call(c, lapply(groups, function(i) max(logs$time[i]))),
    stringsAsFactors = FALSE
  )
  rownames(out) <- NULL
  out <- out[order(-out$n, out$template), , drop = FALSE]
  return(utils::head(out, topN))
}
