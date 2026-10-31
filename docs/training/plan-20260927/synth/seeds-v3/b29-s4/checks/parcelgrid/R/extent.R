e1 <- extent(c(10, 0), c(5, 2), crs = "EPSG:28992")
stopifnot(is(e1, "Extent"), e1@xmin == 0, e1@xmax == 10, e1@ymin == 2, e1@ymax == 5)
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(extent(1:3, 1:2)), fails(extent(1:2, "a")), fails(extent(1:3)))
stopifnot(fails(extent(1:2, 1:2, crs = 4326)))
e4 <- extent(c(0, 10, 4, 2))
stopifnot(e4@xmax == 10, e4@ymin == 2, e4@ymax == 4, e4@crs == "")

stopifnot(isTRUE(valid_extent(e1)))
bad <- e1
bad@xmin <- NA_real_
bad@xmax <- Inf
bad@ymin <- c(1, 2)
bad@ymax <- numeric()
msgs <- valid_extent(bad)
stopifnot(length(msgs) == 4L, grepl("^`xmin`", msgs[1]), grepl("^`xmax`", msgs[2]))
stopifnot(grepl("^`ymin`", msgs[3]), grepl("^`ymax`", msgs[4]))
bad <- e1
bad@xmin <- 20
bad@ymin <- 9
bad@crs <- NA_character_
msgs <- valid_extent(bad)
stopifnot(length(msgs) == 3L, grepl("xmin` must not", msgs[1]), grepl("ymin` must not", msgs[2]))
stopifnot(grepl("^`crs`", msgs[3]))

xmin(e1) <- -1
stopifnot(e1@xmin == -1, inherits(try(set_xmin(e1, 11), silent = TRUE), "try-error"))
stopifnot(inherits(try(set_xmin(e1, "a"), silent = TRUE), "try-error"))
xmax(e1) <- 12
stopifnot(e1@xmax == 12, inherits(try(set_xmax(e1, -2), silent = TRUE), "try-error"))
stopifnot(inherits(try(set_xmax(e1, 1:2), silent = TRUE), "try-error"))
ymin(e1) <- 0
stopifnot(e1@ymin == 0, inherits(try(set_ymin(e1, 6), silent = TRUE), "try-error"))
stopifnot(inherits(try(set_ymin(e1, NULL), silent = TRUE), "try-error"))
ymax(e1) <- 4
stopifnot(e1@ymax == 4, inherits(try(set_ymax(e1, -1), silent = TRUE), "try-error"))
stopifnot(inherits(try(set_ymax(e1, TRUE), silent = TRUE), "try-error"))

e2 <- extent(c(0, 10), c(0, 4))
b <- extent_buffer(e2, 1)
stopifnot(b@xmin == -1, b@xmax == 11, b@ymin == -1, b@ymax == 5)
b <- extent_buffer(e2, -1, -3)
stopifnot(b@xmin == 1, b@xmax == 9, b@ymin == 2, b@ymax == 2)
b <- extent_buffer(e2, -6, 0)
stopifnot(b@xmin == 5, b@xmax == 5, b@ymax == 4)

e3 <- extent(c(0.4, 9.6), c(1.2, 3.7))
stopifnot(identical(extent_snap(e3, 1), extent(c(0, 10), c(1, 4))))
stopifnot(identical(extent_snap(e3, 1, "in"), extent(c(1, 9), c(2, 3))))
stopifnot(identical(extent_snap(e3, 1, "near"), extent(c(0, 10), c(1, 4))))
stopifnot(identical(extent_snap(e3, 5), extent(c(0, 10), c(0, 5))))
stopifnot(fails(extent_snap(e3, 5, "in")), fails(extent_snap(e3, 0)), fails(extent_snap(e3, NA)))

out <- capture.output(res <- show(e2))
stopifnot(length(out) == 3L, grepl("unknown crs", out[1]))
stopifnot(grepl("x: \\[0, 10\\]  width:  10", out[2]), grepl("height: 4", out[3]))
stopifnot(grepl("EPSG:28992", capture.output(show(e1))[1]))
