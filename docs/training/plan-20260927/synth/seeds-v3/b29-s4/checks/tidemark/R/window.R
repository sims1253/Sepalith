w <- timeWindow(2, 5)
stopifnot(is(w, "TimeWindow"), w@start == 2, w@end == 5)
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(timeWindow(5, 2)), fails(timeWindow("a", 2)), fails(timeWindow(1, 2:3)))

stopifnot(isTRUE(validTimeWindow(w)))
bad <- w
bad@start <- NA_real_
bad@end <- c(1, 2)
msgs <- validTimeWindow(bad)
stopifnot(length(msgs) == 2L, grepl("^`start`", msgs[1]), grepl("^`end`", msgs[2]))
bad <- w
bad@end <- 1
stopifnot(identical(validTimeWindow(bad), "`end` must not be before `start`."))

v <- timeWindow(4, 8)
stopifnot(identical(windowIntersect(w, v), timeWindow(4, 5)))
stopifnot(is.null(windowIntersect(w, timeWindow(6, 7))))
stopifnot(identical(windowUnion(w, v), timeWindow(2, 8)))
stopifnot(identical(windowUnion(w, timeWindow(6, 7)), timeWindow(2, 7)))

rel <- function(a, b, c, d) windowRelation(timeWindow(a, b), timeWindow(c, d))
stopifnot(rel(1, 2, 1, 2) == "equals", rel(1, 2, 3, 4) == "before", rel(3, 4, 1, 2) == "after")
stopifnot(rel(1, 2, 2, 4) == "meets", rel(2, 4, 1, 2) == "met-by")
stopifnot(rel(1, 2, 1, 4) == "starts", rel(1, 4, 1, 2) == "started-by")
stopifnot(rel(3, 4, 1, 4) == "finishes", rel(1, 4, 3, 4) == "finished-by")
stopifnot(rel(2, 3, 1, 4) == "during", rel(1, 4, 2, 3) == "contains")
stopifnot(rel(1, 3, 2, 4) == "overlaps", rel(2, 4, 1, 3) == "overlapped-by")

stopifnot(w == timeWindow(2, 5), w != v, w < v, v > w, w <= timeWindow(2, 6), v >= w)
stopifnot(compareWindows(w, timeWindow(2, 4), ">"), !compareWindows(w, v, ">="))
stopifnot(compareWindows(w, w, "<="), !compareWindows(w, w, "<"))

stopifnot(identical(shiftWindow(w, 1), timeWindow(3, 6)))
stopifnot(identical(shiftWindow(w, -1, "start"), timeWindow(1, 5)))
stopifnot(identical(shiftWindow(w, 2, "end"), timeWindow(2, 7)))
stopifnot(inherits(try(shiftWindow(w, 4, "start"), silent = TRUE), "try-error"))
stopifnot(identical(shiftWindow(w, 1, "expand"), timeWindow(1, 6)))
stopifnot(identical(shiftWindow(w, 1, "shrink"), timeWindow(3, 4)))
stopifnot(fails(shiftWindow(w, 2, "shrink")), fails(shiftWindow(w, NA_real_)))
stopifnot(identical(w + 1, timeWindow(3, 6)), identical(w - 2, timeWindow(0, 3)))

stopifnot(identical(as(w, "character"), "[2, 5]"))
stopifnot(identical(characterToWindow(" [ -1.5 , 2e1 ] "), timeWindow(-1.5, 20)))
stopifnot(identical(as("[2, 5]", "TimeWindow"), w))
stopifnot(inherits(try(characterToWindow("2 to 5"), silent = TRUE), "try-error"))
stopifnot(identical(as(w, "numeric"), c(start = 2, end = 5)))
stopifnot(grepl("<TimeWindow> \\[2, 5\\]", capture.output(show(w))))
