# make.rubric
r <- make.rubric(c("ideas", "style"), levels = c(4, 5), weights = c(3, 1))
stopifnot(inherits(r, "rubric"), all.equal(unname(r$weights), c(0.75, 0.25)))
stopifnot(r$levels[["style"]] == 5L)
stopifnot(inherits(try(make.rubric(c("a", "a")), silent = TRUE), "try-error"))
stopifnot(inherits(try(make.rubric("a", weights = 0), silent = TRUE), "try-error"))

# check.rubric.scores
ok <- data.frame(ideas = c(1, 4, NA), style = c(5, 2, 3))
stopifnot(identical(check.rubric.scores(ok, r), ok))
bad <- data.frame(ideas = c(1, 5), style = c(1, 1))
stopifnot(inherits(try(check.rubric.scores(bad, r), silent = TRUE), "try-error"))

# rubric.to.percent / rubric.to.band
sc <- data.frame(ideas = c(1, 4), style = c(1, 5))
stopifnot(all.equal(rubric.to.percent(sc, r), c(0, 100)))
bands <- rubric.to.band(sc, r)
stopifnot(identical(as.character(bands), c("beginning", "exemplary")), is.ordered(bands))

# rater.agreement
stopifnot(rater.agreement(c(1, 2, 3, NA), c(1, 3, 3, 2)) == 2 / 3)
stopifnot(rater.agreement(c(1, 2, 4), c(2, 2, 2), "adjacent") == 2 / 3)
stopifnot(rater.agreement(c(1, 2, 1, 2), c(1, 2, 1, 2), "kappa") == 1)
stopifnot(is.na(rater.agreement(NA, 1)))

# agreement.table
tab <- agreement.table(sc, sc, r)
stopifnot(nrow(tab) == 2, all(tab$exact == 1), identical(tab$criterion, c("ideas", "style")))

# print.rubric
out <- utils::capture.output(res <- print(r))
stopifnot(identical(res, r), length(out) == 3, grepl("75.0%", out[2], fixed = TRUE))
