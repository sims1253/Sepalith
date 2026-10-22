.log_levels <- c(DEBUG = 10L, INFO = 20L, WARN = 30L, ERROR = 40L)

#' Create a logger
#'
#' Builds a lightweight logger that writes timestamped lines to one or more
#' appenders. Messages below `threshold` are dropped before formatting.
#'
#' @export
#' @param name A short label printed in every log line.
#' @param threshold The minimum level to emit, one of `"INFO"`, `"DEBUG"`,
#'   `"WARN"` or `"ERROR"`.
#' @param appenders A list of appender functions, see [appender_console()].
#' @return An object of class `taskrelay_logger`.
new_logger <- function(name = "taskrelay",
                       threshold = c("INFO", "DEBUG", "WARN", "ERROR"),
                       appenders = list(appender_console())) {
  threshold <- match.arg(threshold)
  if (!is.character(name) || length(name) != 1L || is.na(name) || !nzchar(name)) {
    stop("`name` must be a single non-empty string.", call. = FALSE)
  }
  if (!is.list(appenders) || !all(vapply(appenders, is.function, logical(1)))) {
    stop("`appenders` must be a list of functions.", call. = FALSE)
  }

  logger <- list(name = name, threshold = threshold, appenders = appenders)
  class(logger) <- "taskrelay_logger"
  return(logger)
}

#' Format a single log line
#'
#' Produces the canonical `timestamp [LEVEL] name: message` layout used by
#' every appender.
#'
#' @keywords internal
#' @param level The level label, such as `"INFO"`.
#' @param name A short label printed in every log line.
#' @param msg The already interpolated message text.
#' @param time The time stamp to print, defaults to now.
format_log_line <- function(level, name, msg, time = Sys.time()) {
  stamp <- format(time, "%Y-%m-%d %H:%M:%S")
  # One record per line, whatever the message contains
  msg <- gsub("[\r\n]+", " ", msg)
  return(sprintf("%s [%-5s] %s: %s", stamp, level, name, msg))
}

#' Log a message at a given level
#'
#' Checks the logger threshold, interpolates `fmt` with [sprintf()] and hands
#' the formatted line to every appender in turn.
#'
#' @export
#' @param logger A logger created by [new_logger()].
#' @param level The level label, such as `"INFO"`.
#' @param fmt A [sprintf()] format string, or a plain message.
#' @param ... Passed to [sprintf()].
#' @return `TRUE` if the line was emitted, `FALSE` otherwise, invisibly.
log_at <- function(logger, level, fmt, ...) {
  if (!inherits(logger, "taskrelay_logger")) {
    stop("`logger` must be created with new_logger().", call. = FALSE)
  }
  if (!level %in% names(.log_levels)) {
    stop("Unknown log level: ", level, call. = FALSE)
  }
  if (.log_levels[[level]] < .log_levels[[logger$threshold]]) {
    return(invisible(FALSE))
  }

  # Without arguments `fmt` is used verbatim, so a stray "%" in a plain
  # message does not trip sprintf()
  msg <- if (...length() > 0L) sprintf(fmt, ...) else fmt
  line <- format_log_line(level, logger$name, msg)
  for (appender in logger$appenders) {
    appender(line, level)
  }
  return(invisible(TRUE))
}

#' Log an informational message
#'
#' Shorthand for `log_at(logger, "INFO", ...)`.
#'
#' @export
#' @inheritParams log_at
#' @return `TRUE` if the line was emitted, `FALSE` otherwise, invisibly.
#' @rdname log_shortcuts
log_info <- function(logger, fmt, ...) {
  return(log_at(logger, "INFO", fmt, ...))
}

#' Log a warning message
#'
#' Shorthand for `log_at(logger, "WARN", ...)`.
#'
#' @export
#' @inheritParams log_at
#' @rdname log_shortcuts
log_warn <- function(logger, fmt, ...) {
  return(log_at(logger, "WARN", fmt, ...))
}

#' Log an error message
#'
#' Shorthand for `log_at(logger, "ERROR", ...)`.
#'
#' @export
#' @inheritParams log_at
#' @rdname log_shortcuts
log_error <- function(logger, fmt, ...) {
  return(log_at(logger, "ERROR", fmt, ...))
}

#' Appender that writes to the console
#'
#' Returns a closure that prints each log line to standard error or
#' standard output.
#'
#' @export
#' @param stream Which connection to write to, `"stderr"` or `"stdout"`.
#' @return A function taking `line` and `level`.
appender_console <- function(stream = c("stderr", "stdout")) {
  stream <- match.arg(stream)
  con <- switch(stream, stderr = stderr(), stdout = stdout())

  return(function(line, level) {
    cat(line, "\n", sep = "", file = con)
    invisible(line)
  })
}

#' Appender that writes to a file
#'
#' Returns a closure that appends each log line to `path`, creating the
#' parent directory on first use.
#'
#' @export
#' @param path Path to the log file.
#' @param append Whether to keep existing content in `path`.
#' @param min_level Only lines at or above this level are written.
#' @return A function taking `line` and `level`.
appender_file <- function(path, append = TRUE, min_level = "DEBUG") {
  # Validate now; a typo would otherwise only surface on the first log call
  if (!min_level %in% names(.log_levels)) {
    stop("Unknown log level: ", min_level, call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (!append && file.exists(path)) {
    file.remove(path)
  }

  # Open and close per line: slower, but nothing is lost if the job dies
  return(function(line, level) {
    if (.log_levels[[level]] < .log_levels[[min_level]]) {
      return(invisible(NULL))
    }
    cat(line, "\n", sep = "", file = path, append = TRUE)
    invisible(line)
  })
}
