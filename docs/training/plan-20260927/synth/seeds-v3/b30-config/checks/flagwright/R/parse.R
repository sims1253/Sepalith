tok <- split_flag_token("--keep=3")
stopifnot(identical(tok$name, "keep"), identical(tok$value, "3"))
tok <- split_flag_token("--verbose")
stopifnot(identical(tok$name, "verbose"), is.na(tok$value))
stopifnot(identical(split_flag_token("-v", c(v = "verbose"))$name, "verbose"))
stopifnot(is.na(split_flag_token("-5")$name), is.na(split_flag_token("file.txt")$name))

stopifnot(identical(expand_short_flags(c("-vn", "--keep", "x")),
                    c("-v", "-n", "--keep", "x")))
stopifnot(identical(expand_short_flags(c("-v", "-12")), c("-v", "-12")))

stopifnot(isTRUE(parse_flag_value("yes", "logical", "x")))
stopifnot(identical(parse_flag_value("0", "logical", "x"), FALSE))
stopifnot(identical(parse_flag_value("12", "integer", "x"), 12L))
stopifnot(identical(parse_flag_value("1.5", "numeric", "x"), 1.5))
stopifnot(identical(parse_flag_value("abc", "character", "x"), "abc"))
stopifnot(identical(parse_flag_value("/tmp/a", "path", "x"), "/tmp/a"))
stopifnot(identical(parse_flag_value("2024-03-01", "date", "x"), as.Date("2024-03-01")))
stopifnot(identical(parse_flag_value("a, b", "list", "x"), c("a", "b")))
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(parse_flag_value("maybe", "logical", "x")))
stopifnot(fails(parse_flag_value("z", "integer", "x")))
stopifnot(fails(parse_flag_value("z", "numeric", "x")))
stopifnot(fails(parse_flag_value("", "character", "x")))
stopifnot(fails(parse_flag_value("", "path", "x")))
stopifnot(fails(parse_flag_value("soon", "date", "x")))
stopifnot(fails(parse_flag_value("a,,b", "list", "x")))

stopifnot(identical(flag_value(c("--to", "x", "--to=y"), "to"), "y"))
stopifnot(identical(flag_value(c("--a"), "to", "d"), "d"))
stopifnot(fails(flag_value(c("--to"), "to")))

o <- parse_backup_args(c("--source", "/a", "--dest=/b", "--keep", "3",
                         "--compress=xz", "-vn", "extra"))
stopifnot(identical(o$source, "/a"), identical(o$dest, "/b"), identical(o$keep, 3L))
stopifnot(identical(o$compress, "xz"), o$dry_run, o$verbose)
stopifnot(identical(o$positional, "extra"))
d <- parse_backup_args(character())
stopifnot(identical(d$keep, 7L), identical(d$compress, "gzip"), !d$dry_run, !d$verbose)
stopifnot(fails(parse_backup_args("--bogus")), fails(parse_backup_args("--keep")))

spec <- data.frame(name = c("verbose", "level", "out"), short = c("v", "l", NA),
                   type = c("logical", "integer", "character"),
                   default = c(NA, "2", NA), stringsAsFactors = FALSE)
r <- parse_args_with_spec(c("-v", "--level", "5", "pos"), spec)
stopifnot(isTRUE(r$verbose), identical(r$level, 5L), is.null(r$out))
stopifnot(identical(r$positional, "pos"))
r <- parse_args_with_spec(c("--verbose=no", "--out=f.txt"), spec)
stopifnot(identical(r$verbose, FALSE), identical(r$level, 2L), identical(r$out, "f.txt"))
stopifnot(fails(parse_args_with_spec("--nope", spec)), fails(parse_args_with_spec("--out", spec)))

rp <- parse_report_args(c("--input=in.csv", "-o", "out.html", "--format", "pdf",
                          "--from", "2024-01-01", "--to=2024-02-01", "-q", "x"))
stopifnot(identical(rp$input, "in.csv"), identical(rp$output, "out.html"))
stopifnot(identical(rp$format, "pdf"), identical(rp$from, as.Date("2024-01-01")))
stopifnot(identical(rp$to, as.Date("2024-02-01")), rp$quiet, identical(rp$positional, "x"))
rd <- parse_report_args(character())
stopifnot(identical(rd$format, "html"), is.null(rd$from), !rd$quiet)
stopifnot(fails(parse_report_args("--keep=1")), fails(parse_report_args("--from")))
