stopifnot(identical(names(heightUnits()), c("m", "cm", "ft")), heightUnits()[["cm"]] == 0.01)

s <- tideSeries("NL042", 1:3, c(1, 2, 1.5))
stopifnot(is(s, "TideSeries"), s@units == "m", identical(s@heights, c(1, 2, 1.5)))
stopifnot(tideSeries("NL042", 1, 2, units = "c")@units == "cm")
stopifnot(inherits(try(tideSeries("NL042", 1:2, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(tideSeries("NL042", "1", 1), silent = TRUE), "try-error"))
stopifnot(identical(tideSeries("NL042", 3:1, c(3, 2, 1))@heights, c(1, 2, 3)))

stopifnot(isTRUE(validTideSeries(s)))
bad <- s
bad@station <- ""
bad@units <- "yd"
bad@times <- c(3, 2, 1)
bad@heights <- 1
msgs <- validTideSeries(bad)
stopifnot(length(msgs) == 4L, grepl("station", msgs[1]), grepl("units", msgs[2]))
stopifnot(grepl("times", msgs[3]), grepl("heights", msgs[4]))

cm <- convertHeights(s, "cm")
stopifnot(cm@units == "cm", isTRUE(all.equal(cm@heights, c(100, 200, 150))))
stopifnot(identical(convertHeights(s, "m"), s))
stopifnot(isTRUE(all.equal(convertHeights(cm, "m")@heights, s@heights)))

stopifnot(identical((s + 1)@heights, c(2, 3, 2.5)), identical((s - 1)@heights, c(0, 1, 0.5)))
stopifnot(identical((s * 2)@heights, c(2, 4, 3)), identical((s / 2)@heights, c(0.5, 1, 0.75)))
stopifnot(inherits(try(s^2, silent = TRUE), "try-error"))
stopifnot(inherits(try(seriesArith(s, 1:2, "+"), silent = TRUE), "try-error"))

stopifnot(identical((1 + s)@heights, c(2, 3, 2.5)), identical((3 - s)@heights, c(2, 1, 1.5)))
stopifnot(identical((2 * s)@heights, c(2, 4, 3)), identical((3 / s)@heights, c(3, 1.5, 2)))
stopifnot(inherits(try(2 %% s, silent = TRUE), "try-error"))
stopifnot(inherits(try(seriesArith(1:2, s, "+"), silent = TRUE), "try-error"))

tot <- s + cm
stopifnot(tot@units == "m", isTRUE(all.equal(tot@heights, c(2, 4, 3))), tot@station == "NL042")
other <- tideSeries("DE007", 1:3, c(0.5, 0.5, 0.5))
d <- s - other
stopifnot(d@station == "combined", identical(d@heights, c(0.5, 1.5, 1)))
stopifnot(inherits(try(s * other, silent = TRUE), "try-error"))
stopifnot(inherits(try(s + tideSeries("NL042", 2:4, 1:3), silent = TRUE), "try-error"))

stopifnot(identical(s > 1.2, c(FALSE, TRUE, TRUE)), identical(1.2 >= s, c(TRUE, FALSE, FALSE)))

w <- tideSeries("NL042", 1:7, c(1, 2, 1, 0, 0.5, 3, 2))
ex <- tideExtremes(w)
stopifnot(identical(ex$type, c("high", "low", "high")), identical(ex$time, c(2, 4, 6)))
stopifnot(identical(ex$height, c(2, 0, 3)), nrow(tideExtremes(tideSeries("X", 1:2, 1:2))) == 0L)
stopifnot(identical(tideExtremes(tideSeries("X", 1:4, c(3, 2, 2, 1)))$type, "low"))

out <- capture.output(res <- showTideSeries(w))
stopifnot(length(out) == 6L, grepl("NL042, 7 predictions in m", out[1]), identical(res, w))
stopifnot(grepl("3.00 m", out[5]), grepl("tides: 2 high, 1 low", out[6]))
stopifnot(grepl("0.000", capture.output(showTideSeries(w, 3))[4]))
stopifnot(length(capture.output(show(tideSeries("X", numeric(), numeric())))) == 1L)
