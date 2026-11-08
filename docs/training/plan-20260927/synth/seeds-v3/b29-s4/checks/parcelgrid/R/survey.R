p <- survey_point(c(0, 3), c(0, 4), z = c(1, 1), accuracy = c(0.02, 0.1))
stopifnot(is(p, "SurveyPoint"), identical(p@z, c(1, 1)),
  identical(survey_point(1, 2)@accuracy, 0.05))
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(survey_point(1:2, 1)), fails(survey_point("1", 1)),
  fails(survey_point(1:2, 1:2, 1:3)))
stopifnot(fails(survey_point(1:3, 1:3, accuracy = 1:2)))

stopifnot(isTRUE(valid_survey_point(p)))
bad <- p
bad@x <- c(0, NA)
bad@y <- 1
bad@z <- c(1, Inf)
bad@accuracy <- c(-1, 0)
msgs <- valid_survey_point(bad)
stopifnot(length(msgs) == 4L, grepl("^`x`", msgs[1]), grepl("^`y` must contain 2", msgs[2]))
stopifnot(grepl("^`z`", msgs[3]), grepl("^`accuracy`", msgs[4]))

q <- p + 1
stopifnot(identical(q@x, c(1, 4)), identical(q@z, c(2, 2)), identical(q@accuracy, p@accuracy))
stopifnot(identical((p - 1)@y, c(-1, 3)), identical((p * -2)@x, c(0, -6)))
stopifnot(identical((p * -2)@accuracy, c(0.04, 0.2)), identical((p / 2)@accuracy, c(0.01, 0.05)))
stopifnot(identical((p / 2)@y, c(0, 2)), inherits(try(p^2, silent = TRUE), "try-error"))

stopifnot(identical((1 + p)@x, c(1, 4)), identical((10 - p)@y, c(10, 6)))
stopifnot(identical((10 - p)@z, c(9, 9)), identical((2 * p)@accuracy, c(0.04, 0.2)))
stopifnot(identical((2 * p)@y, c(0, 8)), inherits(try(2 / p, silent = TRUE), "try-error"))

stopifnot(identical(p > 0.05, c(FALSE, TRUE)), identical(p <= 0.02, c(TRUE, FALSE)))
stopifnot(identical(point_arith(p, 1, "+"), q),
  inherits(try(point_arith(2, p, "/"), silent = TRUE),
  "try-error"))

t1 <- point_transform(p, scale = 2, angle = pi / 2, dx = 1, dy = -1, dz = 5)
stopifnot(isTRUE(all.equal(t1@x, c(1, -7))), isTRUE(all.equal(t1@y, c(-1, 5))))
stopifnot(identical(t1@z, c(6, 6)), identical(t1@accuracy, c(0.04, 0.2)))
stopifnot(identical(point_transform(p), p))

o <- survey_point(0, 0, 0)
stopifnot(isTRUE(all.equal(point_distance(p, o), c(1, sqrt(26)))))
stopifnot(identical(point_distance(p, o, "manhattan"), c(1, 8)))
stopifnot(identical(point_distance(p, o, "chebyshev"), c(1, 4)))
stopifnot(identical(point_distance(p, o, "planar"), c(0, 5)))

m <- as(p, "matrix")
stopifnot(identical(colnames(m), c("x", "y", "z", "accuracy")), nrow(m) == 2L, m[2, "y"] == 4)
stopifnot(identical(point_to_matrix(p), m), identical(as(m, "SurveyPoint"), p))
p2 <- matrix_to_point(cbind(x = 1:2, y = 3:4))
stopifnot(identical(p2@z, c(0, 0)), identical(p2@accuracy, c(0.05, 0.05)))
stopifnot(inherits(try(matrix_to_point(cbind(x = 1)), silent = TRUE), "try-error"))

out <- capture.output(res <- show_survey_point(p))
stopifnot(length(out) == 3L, grepl("2 points, median accuracy 0.060 m", out[1]), identical(res, p))
many <- survey_point(1:7, 1:7)
out <- capture.output(show_survey_point(many, n = 3))
stopifnot(length(out) == 5L, grepl("and 4 more", out[5]))
stopifnot(length(capture.output(show(many))) == 7L)
