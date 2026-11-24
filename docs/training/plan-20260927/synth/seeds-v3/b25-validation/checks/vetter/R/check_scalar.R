fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))

stopifnot(identical(fmt_arg("n"), "`n`"))
stopifnot(identical(fmt_arg(c("a", "b")), c("`a`", "`b`")))

stopifnot(identical(check_number(2.5), 2.5))
stopifnot(fails(check_number("a")))
stopifnot(fails(check_number(Inf)))
stopifnot(identical(msg(check_number(5, "k", upper = 3)), "`k` must be at most 3."))
stopifnot(identical(msg(check_number(-1, "k", lower = 0)), "`k` must be at least 0."))

stopifnot(identical(check_count(3), 3L))
stopifnot(fails(check_count(2.5)))
stopifnot(identical(msg(check_count(-2, "n")), "`n` must be at least 0."))
stopifnot(fails(check_count(10, upper = 5)))

stopifnot(identical(check_string("a"), "a"))
stopifnot(fails(check_string(NA_character_)))
stopifnot(identical(msg(check_string("", "s")), "`s` must not be empty."))
stopifnot(identical(check_string("", allow_empty = TRUE), ""))

stopifnot(isTRUE(check_flag(TRUE)))
stopifnot(fails(check_flag(NA)))
stopifnot(identical(msg(check_flag(1, "v")), "`v` must be a single flag."))

stopifnot(identical(check_scalar(1.5, "number"), 1.5))
stopifnot(identical(check_scalar(2, "count"), 2))
stopifnot(identical(check_scalar("x", "string"), "x"))
stopifnot(identical(check_scalar(FALSE, "flag"), FALSE))
stopifnot(identical(check_scalar("b", "choice", choices = c("a", "b")), "b"))
stopifnot(identical(msg(check_scalar("z", "choice", "m", c("a", "b"))),
                    "`m` must be one of `a`, `b`."))

ok <- check_sampler_args(100, 10, 2, 0.5, 1, TRUE, "hmc")
stopifnot(identical(names(ok), c("n_iter", "burn_in", "thin", "step_size", "seed",
                                 "verbose", "method")))
stopifnot(identical(ok$n_iter, 100L), identical(ok$method, "hmc"))
stopifnot(identical(msg(check_sampler_args(100, 100, 2, 0.5, 1, TRUE, "hmc")),
                    "`burn_in` must be smaller than `n_iter`."))
stopifnot(identical(msg(check_sampler_args(100, 10, 0, 0.5, 1, TRUE, "hmc")),
                    "`thin` must be at least 1."))
stopifnot(identical(msg(check_sampler_args(100, 10, 1, 0, 1, TRUE, "hmc")),
                    "`step_size` must be positive."))
stopifnot(fails(check_sampler_args(0, 0, 1, 0.5, 1, TRUE, "hmc")))
stopifnot(fails(check_sampler_args(100, 10, 1, 0.5, 1, NA, "hmc")))
stopifnot(fails(check_sampler_args(100, 10, 1, 0.5, 1, TRUE, "nuts")))

stopifnot(identical(check_number(NA, allow_na = TRUE), NA))
stopifnot(identical(check_count(NA, allow_na = TRUE), NA_integer_))
stopifnot(identical(msg(check_scalar(0, "positive", "p")), "`p` must be positive."))
stopifnot(identical(check_scalar(3, "positive"), 3))
stopifnot(identical(check_scalar(0.2, "probability"), 0.2))
stopifnot(identical(msg(check_scalar(2, "probability", "p")), "`p` must be at most 1."))

opt <- check_optimizer_args(100, 1e-8, 1e-6, 0.1, 10, NA, FALSE)
stopifnot(identical(names(opt), c("max_iter", "abs_tol", "rel_tol", "learning_rate",
                                  "patience", "max_time", "trace")))
stopifnot(identical(opt$max_iter, 100L), identical(opt$patience, 10L), is.na(opt$max_time))
stopifnot(identical(check_optimizer_args(5, 1, 1, 1, 0, 60, TRUE)$max_time, 60))
stopifnot(identical(msg(check_optimizer_args(0, 1, 1, 0.1, 0, NA, TRUE)),
                    "`max_iter` must be at least 1."))
stopifnot(identical(msg(check_optimizer_args(10, -1, 1, 0.1, 0, NA, TRUE)),
                    "`abs_tol` must be positive."))
stopifnot(identical(msg(check_optimizer_args(10, 1, 0, 0.1, 0, NA, TRUE)),
                    "`rel_tol` must be positive."))
stopifnot(identical(msg(check_optimizer_args(10, 1, 1, 1.5, 0, NA, TRUE)),
                    "`learning_rate` must be at most 1."))
stopifnot(identical(msg(check_optimizer_args(10, 1, 1, 0.1, 10, NA, TRUE)),
                    "`patience` must be smaller than `max_iter`."))
stopifnot(identical(msg(check_optimizer_args(10, 1, 1, 0.1, 1, -5, TRUE)),
                    "`max_time` must be at least 0."))
stopifnot(fails(check_optimizer_args(10, 1, 1, 0.1, 1, NA, "yes")))
