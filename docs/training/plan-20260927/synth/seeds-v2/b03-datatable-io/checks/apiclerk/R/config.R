# skipped: config_from_env, clerk_config, config_sources (need magrittr)

defaults <- config_defaults()
stopifnot("token" %in% names(defaults), is.null(defaults$token), defaults$max_tries == 4L)

# config_from_file
stopifnot(identical(config_from_file(NULL), list()))
stopifnot(identical(config_from_file(tempfile()), list()))

# coerce_config
cfg <- coerce_config(modifyList(defaults, list(timeout = "10", base_url = "http://x.org//")))
stopifnot(identical(cfg$timeout, 10), identical(cfg$base_url, "http://x.org"))
stopifnot(is.na(coerce_config(modifyList(defaults, list(max_tries = "lots")))$max_tries))

# validate_config
stopifnot(identical(validate_config(cfg), cfg))
bad <- modifyList(cfg, list(base_url = "ftp://x", timeout = 0))
msg <- tryCatch(validate_config(bad), error = conditionMessage)
stopifnot(grepl("base_url", msg), grepl("timeout", msg), !grepl("max_tries", msg))

# show_config masks the token
cfg$token <- "secret-token"
shown <- NULL
invisible(capture.output(shown <- show_config(cfg)))
stopifnot(identical(shown$value[shown$setting == "token"], "secr********"))
stopifnot(nrow(shown) == length(cfg))
