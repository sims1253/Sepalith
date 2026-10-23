stopifnot(identical(change.direction(c(1, 5, 2, NA), c(2, 3, 2, 1)),
                    c("up", "down", "flat", NA)))
stopifnot(identical(change.direction(1, 1.05, tolerance = 0.1), "flat"))

stopifnot(identical(change.sentence("Sales", 10, 12.5), "Sales rose by 2.5, from 10.0 to 12.5."))
stopifnot(identical(change.sentence("Churn", 5, 3, 0, "%"), "Churn fell by 2%, from 5% to 3%."))
stopifnot(identical(change.sentence("Staff", 4, 4, 0), "Staff was unchanged at 4."))
stopifnot(identical(change.sentence("X", NA, 1), "X cannot be compared."))

stopifnot(identical(extremes.sentence(c(3, 9, 1), c("a", "b", "c")),
                    "b was highest (9.0) and c was lowest (1.0)."))
stopifnot(identical(extremes.sentence(c(NA, NA), c("a", "b")), "No group has an observed value."))
stopifnot(inherits(try(extremes.sentence(1, c("a", "b")), silent = TRUE), "try-error"))

v <- c(5, 9, NA, 7, 1)
lb <- c("N", "S", "E", "W", "C")
stopifnot(identical(rank.sentence(v, lb), "The top 3 were S (9), W (7) and N (5)."))
stopifnot(identical(rank.sentence(v, lb, n = 2, decreasing = FALSE),
                    "The bottom 2 were C (1) and N (5)."))
stopifnot(identical(rank.sentence(v, lb, n = 1), "The top 1 were S (9)."))
stopifnot(inherits(try(rank.sentence(1, lb), silent = TRUE), "try-error"))

stopifnot(identical(threshold.sentence(2, "pH", 4, 9),
                    "pH (2.0) is below the lower bound of 4.0."))
stopifnot(identical(threshold.sentence(10, "pH", 4, 9),
                    "pH (10.0) is above the upper bound of 9.0."))
stopifnot(identical(threshold.sentence(5, "pH", 4, 9, 0), "pH (5) is within range."))
stopifnot(identical(threshold.sentence(NA, "pH"), "pH is missing."))

stopifnot(identical(trend.phrase(c(1, 2, 3, 2, 4)), "rose in 3 of 4 periods"))
stopifnot(identical(trend.phrase(c(3, 1), "month"), "fell in 1 of 1 month"))
stopifnot(identical(trend.phrase(c(1, 2, 1), "week"), "was mixed over 2 weeks"))
stopifnot(identical(trend.phrase(5), "had too few values to show a trend"))

fn <- footnote.registry()
stopifnot(identical(fn$add("Source: survey"), "[1]"), identical(fn$add("Provisional"), "[2]"))
stopifnot(identical(fn$add("Source: survey"), "[1]"), fn$count() == 2L)
stopifnot(identical(fn$render(), c("[1] Source: survey", "[2] Provisional")))
fc <- footnote.registry("caret")
stopifnot(identical(fc$add("x"), "^1"), identical(fc$render(), "^1 x"))

pc <- period.comparison(c(12, 3, 5), c(10, 4, 5), c("Sales", "Churn", "Staff"), 0)
stopifnot(identical(names(pc), c("label", "current", "previous", "change", "direction",
                                 "sentence")))
stopifnot(identical(pc$direction, c("up", "down", "flat")), all.equal(pc$change, c(2, -1, 0)))
stopifnot(identical(pc$sentence[1], "Sales rose by 2, from 10 to 12."))
stopifnot(inherits(try(period.comparison("a", 1, "x"), silent = TRUE), "try-error"))
stopifnot(inherits(try(period.comparison(1, 1:2, "x"), silent = TRUE), "try-error"))
stopifnot(inherits(try(period.comparison(1, 1, c("x", "y")), silent = TRUE), "try-error"))
