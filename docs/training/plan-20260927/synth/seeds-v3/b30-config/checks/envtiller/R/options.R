fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
d <- tiller.option.defaults()
stopifnot(identical(names(d), c("tiller.timeout", "tiller.retries", "tiller.verbose",
                                "tiller.cache.dir", "tiller.log.level")))

stopifnot(identical(tiller.validate.option("tiller.timeout", 5L), 5))
stopifnot(fails(tiller.validate.option("tiller.timeout", 0)))
stopifnot(identical(tiller.validate.option("tiller.retries", 2), 2L))
stopifnot(fails(tiller.validate.option("tiller.retries", -1)))
stopifnot(isTRUE(tiller.validate.option("tiller.verbose", TRUE)))
stopifnot(fails(tiller.validate.option("tiller.verbose", "yes")))
stopifnot(identical(tiller.validate.option("tiller.cache.dir", "/x"), "/x"))
stopifnot(fails(tiller.validate.option("tiller.cache.dir", "")))
stopifnot(identical(tiller.validate.option("tiller.log.level", "warn"), "warn"))
msg <- tryCatch(tiller.validate.option("tiller.log.level", "loud"), error = conditionMessage)
stopifnot(identical(msg, "Option `tiller.log.level` must be debug, info, warn or error."))

options(tiller.timeout = NULL, tiller.retries = NULL, tiller.verbose = NULL,
        tiller.cache.dir = NULL, tiller.log.level = NULL)
stopifnot(identical(tiller.get.option("timeout"), 60))
stopifnot(identical(tiller.get.option("tiller.retries"), 3L))
stopifnot(fails(tiller.get.option("colour")))

old <- tiller.set.options(timeout = 5, tiller.verbose = TRUE)
stopifnot(identical(names(old), c("tiller.timeout", "tiller.verbose")))
stopifnot(is.null(old$tiller.timeout))
stopifnot(identical(tiller.get.option("timeout"), 5), isTRUE(tiller.get.option("verbose")))
stopifnot(fails(tiller.set.options(timeout = -1)), fails(tiller.set.options(colour = 1)))
stopifnot(fails(tiller.set.options(5)), identical(tiller.set.options(), list()))
options(tiller.retries = "x")
stopifnot(fails(tiller.get.option("retries")))
options(tiller.retries = NULL)

old <- tiller.options.from.env(c(TILLER_TIMEOUT = "12", TILLER_RETRIES = "4",
                                  TILLER_VERBOSE = "false", TILLER_CACHE_DIR = "/c",
                                  TILLER_LOG_LEVEL = "WARN", OTHER = "x"))
stopifnot(identical(names(old), c("tiller.timeout", "tiller.retries", "tiller.verbose",
                                  "tiller.cache.dir", "tiller.log.level")))
stopifnot(identical(tiller.get.option("timeout"), 12), identical(tiller.get.option("retries"), 4L))
stopifnot(identical(tiller.get.option("verbose"), FALSE))
stopifnot(identical(tiller.get.option("cache.dir"), "/c"))
stopifnot(identical(tiller.get.option("log.level"), "warn"))
stopifnot(identical(tiller.options.from.env(c(TILLER_TIMEOUT = "")), list()))
stopifnot(fails(tiller.options.from.env(c(TILLER_RETRIES = "many"))))
options(old)

tab <- tiller.options.table()
stopifnot(identical(tab$option, names(d)))
stopifnot(identical(tab$is.default, c(FALSE, TRUE, FALSE, TRUE, TRUE)))
stopifnot(identical(tab$value[[1]], "5"))

options(tiller.timeout = 9)
prev <- tiller.reset.options(only.missing = TRUE)
stopifnot(identical(getOption("tiller.timeout"), 9), "tiller.cache.dir" %in% names(prev))
stopifnot(length(tiller.reset.options(only.missing = TRUE)) == 0L)
tiller.reset.options()
stopifnot(identical(getOption("tiller.timeout"), 60))
stopifnot(identical(getOption("tiller.verbose"), FALSE))
