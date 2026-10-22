# skipped: needs cli (burrow_inform/success/warn when not quiet, the error
#   path of format_summary_bullets, with_step when not quiet) and
#   rlang (burrow_map)
options(burrowlog.quiet = NULL)
Sys.setenv(BURROWLOG_QUIET = "yes")
stopifnot(is_quiet())
Sys.setenv(BURROWLOG_QUIET = "no")
stopifnot(!is_quiet())
options(burrowlog.quiet = TRUE)
stopifnot(is_quiet())

out <- format_summary_bullets(c(rows = 1200, dropped = 3))
stopifnot(identical(names(out), c("*", "*")))
stopifnot(identical(unname(out), c("rows    : 1,200", "dropped : 3")))
stopifnot(identical(unname(format_summary_bullets(c(p = "{x}"), "i")), "p : {{x}}"))
stopifnot(identical(format_summary_bullets(numeric()), character()))

# Quiet mode: no cli calls are reached
stopifnot(is.null(burrow_inform("x")), is.null(burrow_warn("y")))
stopifnot(identical(with_step("Counting", 1 + 1), 2))
