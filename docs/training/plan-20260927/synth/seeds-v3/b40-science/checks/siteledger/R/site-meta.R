fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
d0 <- as.Date("2020-01-01")

stopifnot(isTRUE(site.validate("A1", 50, 8, 100, d0)))
stopifnot(fails(site.validate("", 50, 8, 100, d0)))
stopifnot(fails(site.validate("A1", 95, 8, 100, d0)), fails(site.validate("A1", "x", 8, 1, d0)))
stopifnot(fails(site.validate("A1", 50, 190, 100, d0)), fails(site.validate("A1", 50, NA, 1, d0)))
stopifnot(fails(site.validate("A1", 50, 8, 9500, d0)), fails(site.validate("A1", 50, 8, NA, d0)))
stopifnot(fails(site.validate("A1", 50, 8, 100, "2020")))
stopifnot(fails(site.validate("A1", 50, 8, 100, d0, d0 - 1)))

meta <- site.register(c("A1", "B2", "C3"), c(50, 51, -10), c(8, 9, -60), c(100, 200, 50),
                      rep(d0, 3), as.Date(c(NA, "2022-01-01", NA)))
stopifnot(identical(names(meta), c("id", "lat", "lon", "elevation", "start", "end", "active")))
stopifnot(identical(meta$active, c(TRUE, FALSE, TRUE)))
stopifnot(fails(site.register(c("A", "A"), c(1, 2), c(1, 2), c(1, 2), rep(d0, 2))))

stopifnot(abs(site.distance(0, 0, 0, 1) - 111.195) < 0.01)
stopifnot(abs(site.distance(0, 0, 0, 1, "m") - 111195) < 10)
stopifnot(abs(site.distance(0, 0, 0, 1, "mi") - 69.09) < 0.01)
stopifnot(abs(site.distance(0, 0, 0, 1, "nmi") - 60.04) < 0.01)

bb <- site.bbox(meta, pad = 1)
stopifnot(identical(names(bb), c("south", "north", "west", "east")))
stopifnot(bb[["south"]] == -11, bb[["north"]] == 52, bb[["west"]] == -61, bb[["east"]] == 10)
stopifnot(site.bbox(meta, pad = 100)[["north"]] == 90, fails(site.bbox(meta[0, ])))

nn <- site.nearest(meta, 51, 9)
stopifnot(nn$id == "A1", "distance_km" %in% names(nn))
stopifnot(site.nearest(meta, 51, 9, active.only = FALSE)$id == "B2")
stopifnot(identical(site.nearest(meta, 0, 0, n = 2)$id, c("A1", "C3")))

dd <- site.describe(meta, digits = 1)
stopifnot(dd[1] == "A1: 50.0N 8.0E, 100 m, 2020-01-01 to now")
stopifnot(dd[2] == "B2: 51.0N 9.0E, 200 m, 2020-01-01 to 2022-01-01")
stopifnot(grepl("10.0S 60.0W", dd[3]))
