b <- bbox_from_points(c(1, 5, NA, 3), c(10, 20, 0, 15), pad = 1)
stopifnot(identical(b, c(xmin = 0, ymin = 9, xmax = 6, ymax = 21)))
stopifnot(bbox_from_points(179.5, 89.5, pad = 2)[["xmax"]] == 180)
stopifnot(bbox_from_points(-179.5, -89.5, pad = 2)[["ymin"]] == -90)
stopifnot(inherits(try(bbox_from_points(NA, 1), silent = TRUE), "try-error"))

a <- c(xmin = 0, ymin = 0, xmax = 10, ymax = 10)
z <- c(xmin = 5, ymin = -5, xmax = 15, ymax = 5)
stopifnot(identical(bbox_combine(a, z, "union"), c(xmin = 0, ymin = -5, xmax = 15, ymax = 10)))
stopifnot(identical(bbox_combine(a, z, "intersection"), c(xmin = 5, ymin = 0, xmax = 10,
  ymax = 5)))
far <- c(xmin = 20, ymin = 20, xmax = 30, ymax = 30)
stopifnot(is.null(bbox_combine(a, far, "intersection")))
stopifnot(inherits(try(bbox_combine(1, a), silent = TRUE), "try-error"))
stopifnot(inherits(try(bbox_combine(a, c(xmin = 2, ymin = 0, xmax = 1, ymax = 1)),
  silent = TRUE), "try-error"))

stopifnot(identical(bbox_expand(a, 1, "degrees"), c(xmin = -1, ymin = -1, xmax = 11, ymax = 11)))
stopifnot(identical(bbox_expand(a, 0.1, "fraction"), c(xmin = -1, ymin = -1, xmax = 11,
  ymax = 11)))
stopifnot(identical(bbox_expand(a, 10, "percent"), bbox_expand(a, 0.1, "fraction")))
m <- bbox_expand(c(xmin = 0, ymin = -1, xmax = 1, ymax = 1), 6371008.8 * pi / 180, "meters")
stopifnot(isTRUE(all.equal(m[["ymax"]], 2)), isTRUE(all.equal(m[["xmax"]], 2)))
stopifnot(inherits(try(bbox_expand(a, -6, "degrees"), silent = TRUE), "try-error"))

stopifnot(identical(bbox_contains(a, c(0, 5, 11, NA), c(5, 5, 5, 5)), c(TRUE, TRUE, FALSE,
  FALSE)))
stopifnot(!bbox_contains(a, 0, 5, edges = FALSE))
wrap <- c(xmin = 170, ymin = -10, xmax = -170, ymax = 10)
stopifnot(identical(bbox_contains(wrap, c(175, -175, 0), c(0, 0, 0)), c(TRUE, TRUE, FALSE)))
stopifnot(inherits(try(bbox_contains(c(xmin = 0, ymin = 5, xmax = 1, ymax = 1), 0, 0),
  silent = TRUE), "try-error"))

s <- bbox_split(a, 2, 5)
stopifnot(nrow(s) == 10, s$xmax[1] == 5, s$ymax[1] == 2, s$col[2] == 2, s$row[3] == 2)
stopifnot(s$xmin[10] == 5, s$ymin[10] == 8, s$ymax[10] == 10)

cl <- bbox_clip(c(xmin = -200, ymin = -10, xmax = 190, ymax = 95), quiet = TRUE)
stopifnot(identical(attr(cl, "clipped"), c("xmin", "xmax", "ymax")), cl[["ymin"]] == -10)
stopifnot(cl[["xmin"]] == -180, cl[["xmax"]] == 180, cl[["ymax"]] == 90)
cl2 <- bbox_clip(c(xmin = 0, ymin = -95, xmax = 1, ymax = 1), quiet = TRUE)
stopifnot(identical(attr(cl2, "clipped"), "ymin"))
stopifnot(length(attr(bbox_clip(a, quiet = TRUE), "clipped")) == 0)
