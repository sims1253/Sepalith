fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
spec <- new_flag_spec(c("verbose", "keep"), short = c("v", NA), type = c("logical", "integer"),
                      default = c(NA, "7"), help = c("Talk more.", "Copies to keep."))
stopifnot(inherits(spec, "flag_spec"), nrow(spec) == 2L, identical(spec$default, c(NA, "7")))
stopifnot(fails(new_flag_spec(c("a", "a"))), fails(new_flag_spec("Bad")))
stopifnot(fails(new_flag_spec(c("a", "b"), short = c("x", "x"))))
stopifnot(fails(new_flag_spec("a", short = "xy")))
stopifnot(fails(new_flag_spec("a", type = "blob")), fails(new_flag_spec("a", "v", "logical", "1")))
stopifnot(fails(new_flag_spec("a", help = NA)))

fl <- flag_spec_from_list(list(verbose = list(short = "v", type = "logical", help = "Talk more."),
                               keep = list(type = "integer", default = 7L,
                                           help = "Copies to keep.")))
stopifnot(identical(fl, spec))
stopifnot(identical(flag_spec_from_list(list(x = list()))$type, "character"))

stopifnot(identical(wrap_help(""), ""), identical(wrap_help("a b", 80L), "a b"))
w <- wrap_help(paste(rep("word", 30), collapse = " "), width = 40L, indent = 4L)
stopifnot(grepl("\n    word", w, fixed = TRUE))

f <- format(spec)
stopifnot(length(f) == 2L, startsWith(f[[1]], "  -v, --verbose"), grepl("Talk more.$", f[[1]]))
stopifnot(grepl("    --keep <integer>", f[[2]], fixed = TRUE))
stopifnot(grepl("[default: 7]", f[[2]], fixed = TRUE))
out <- capture.output(res <- print(spec))
stopifnot(identical(out[[1]], "<flag_spec> 2 flag(s)"), length(out) == 3L, identical(res, spec))

s <- summary(new_flag_spec(c("verbose", "out", "keep"), type = c("logical", "path", "integer"),
                           default = c(NA, NA, "1"), help = c("x", "", "y")))
stopifnot(s$n_flags == 3L, s$n_short == 0L, s$n_switches == 1L, s$n_defaults == 1L)
stopifnot(identical(s$required, "out"), identical(s$undocumented, "out"))
stopifnot(s$types[["integer"]] == 1L, s$types[["date"]] == 0L)

u <- format_usage(spec, "backup", description = "Copy things.",
                  positional = c(target = "Where to go."), env = c(BACKUP_HOME = "Root."))
stopifnot(startsWith(u, "Usage: backup [options] <target>"))
stopifnot(grepl("\n\nCopy things.\n\nOptions:\n", u, fixed = TRUE))
stopifnot(grepl("Arguments:\n  target", u))
stopifnot(grepl("\n\nEnvironment:\n  BACKUP_HOME", u))
u2 <- format_usage(spec, "x")
stopifnot(startsWith(u2, "Usage: x [options]\n\nOptions:\n"), !grepl("Arguments|Environment", u2))
