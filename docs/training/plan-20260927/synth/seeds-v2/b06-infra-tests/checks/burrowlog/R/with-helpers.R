# skipped: needs withr (local_quiet, with_quiet, local_burrow_seed,
#   with_burrow_seed, local_burrow_project, with_burrow_project, local_log_file)
path <- tempfile(fileext = ".log")
options(burrowlog.log_file = path)
log_line("info", "started")
line <- log_line("error", "failed")
lines <- readLines(path)
stopifnot(length(lines) == 2L, grepl("\\[INFO\\] started$", lines[1]))
stopifnot(identical(lines[2], line), grepl("\\[ERROR\\] failed$", line))
stopifnot(inherits(try(log_line("debug", "x"), silent = TRUE), "try-error"))

options(burrowlog.log_file = NULL)
stopifnot(is.null(log_line("warn", "nobody hears this")))
unlink(path)
