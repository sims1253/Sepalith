# new_logger()
lines <- character()
collect <- function(line, level) lines <<- c(lines, line)
lg <- new_logger("unit", threshold = "INFO", appenders = list(collect))
stopifnot(
  inherits(lg, "taskrelay_logger"),
  identical(lg$threshold, "INFO"),
  inherits(try(new_logger(""), silent = TRUE), "try-error"),
  inherits(try(new_logger("x", appenders = list(1)), silent = TRUE), "try-error")
)

# format_log_line()
t0 <- as.POSIXct("2026-03-02 08:15:00", tz = "UTC")
stopifnot(
  identical(format_log_line("INFO", "unit", "a\nb", time = t0),
            "2026-03-02 08:15:00 [INFO ] unit: a b"),
  identical(format_log_line("ERROR", "x", "boom", time = t0),
            "2026-03-02 08:15:00 [ERROR] x: boom")
)

# log_at(), log_info(), log_warn(), log_error()
stopifnot(
  isFALSE(log_at(lg, "DEBUG", "hidden")),
  isTRUE(log_info(lg, "%d file(s)", 3L)),
  isTRUE(log_warn(lg, "100% plain")),
  isTRUE(log_error(lg, "bad %s", "row")),
  length(lines) == 3L,
  grepl("\\[INFO \\] unit: 3 file\\(s\\)$", lines[1]),
  grepl("\\[WARN \\] unit: 100% plain$", lines[2]),
  inherits(try(log_at(lg, "TRACE", "x"), silent = TRUE), "try-error"),
  inherits(try(log_info(list(), "x"), silent = TRUE), "try-error")
)

# appender_console()
con_app <- appender_console("stderr")
res <- con_app("console appender check", "INFO")
stopifnot(identical(res, "console appender check"), is.function(appender_console("stdout")))

# appender_file()
path <- file.path(tempfile(), "sub", "run.log")
file_app <- appender_file(path, min_level = "WARN")
file_app("skip me", "INFO")
file_app("keep me", "ERROR")
stopifnot(
  identical(readLines(path), "keep me"),
  inherits(try(appender_file(path, min_level = "LOUD"), silent = TRUE), "try-error")
)
unlink(dirname(dirname(path)), recursive = TRUE)
