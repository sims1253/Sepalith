# skipped: needs brms/purrr: toBrmsPrior(); needs rstanarm: toRstanarmPrior();
# needs purrr/tibble/dplyr: priorTable()
p <- newPrior("normal", c(mean = 0, sd = 1))
stopifnot(inherits(p, "pcPrior"), identical(p$coef, ""), p$class == "b")
stopifnot(inherits(try(newPrior("normal", c(0, 1)), silent = TRUE), "try-error"))

n <- normalPrior(1, 2.5, coef = "age")
stopifnot(identical(n$params, c(mean = 1, sd = 2.5)), n$coef == "age")
stopifnot(inherits(try(normalPrior(sd = 0), silent = TRUE), "try-error"))
stopifnot(grepl("positive", try(normalPrior(sd = NA), silent = TRUE)))

st <- studentPrior()
stopifnot(st$family == "student_t", identical(names(st$params), c("df", "location", "scale")))
stopifnot(inherits(try(studentPrior(scale = -1), silent = TRUE), "try-error"))
ex <- exponentialPrior(2)
stopifnot(ex$class == "sd", ex$params[["rate"]] == 2)

stopifnot(identical(priorToStan(normalPrior(0, 2.5)), "normal(0, 2.5)"))
stopifnot(identical(priorToStan(st), "student_t(3, 0, 2.5)"))
stopifnot(identical(priorToStan(exponentialPrior(1 / 3), digits = 2), "exponential(0.33)"))
bad <- newPrior("cauchy", c(location = 0))
stopifnot(inherits(try(priorToStan(bad), silent = TRUE), "try-error"))

y <- c(0, 2, 4, NA)
stopifnot(scalePriorToData(normalPrior(0, 1), y)$params[["sd"]] == 2)
stopifnot(scalePriorToData(exponentialPrior(1), y)$params[["rate"]] == 0.5)
flat <- withCallingHandlers(
  scalePriorToData(st, c(1, 1)),
  warning = function(w) invokeRestart("muffleWarning")
)
stopifnot(identical(flat, st))
