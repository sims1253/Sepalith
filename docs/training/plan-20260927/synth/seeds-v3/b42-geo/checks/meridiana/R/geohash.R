near <- function(a, b) isTRUE(all.equal(a, b, tolerance = 1e-9))
stopifnot(identical(geohash.encode(c(-5.6, 10.40744), c(42.6, 57.64911), 5L),
  c("ezs42", "u4pru")))
stopifnot(identical(geohash.encode(10.40744, 57.64911, 11L), "u4pruydqqvj"))
stopifnot(identical(geohash.encode(0, 0, 1L), "s"))
stopifnot(inherits(try(geohash.encode(0, 0, 13L), silent = TRUE), "try-error"))

b <- geohash.bounds(c("s", "EZS42"))
stopifnot(identical(unlist(b[1, ]), c(xmin = 0, ymin = 0, xmax = 45, ymax = 45)))
stopifnot(b$xmin[2] < -5.6, b$xmax[2] > -5.6, b$ymin[2] < 42.6, b$ymax[2] > 42.6)
stopifnot(inherits(try(geohash.bounds("abc"), silent = TRUE), "try-error"))

d <- geohash.decode("s")
stopifnot(d$lon == 22.5, d$lat == 22.5, d$lon.err == 22.5, d$lat.err == 22.5)
d2 <- geohash.decode("u4pruydqqvj")
stopifnot(abs(d2$lon - 10.40744) < 1e-5, abs(d2$lat - 57.64911) < 1e-5)

stopifnot(geohash.neighbour("ezs42", "north") == "ezs48")
stopifnot(geohash.neighbour("ezs42", "south") == "ezs40")
stopifnot(geohash.neighbour("ezs42", "east") == "ezs43")
stopifnot(geohash.neighbour("ezs42", "west") == "ezefr")
stopifnot(geohash.neighbour("ezs42", "northeast") == "ezs49")
stopifnot(geohash.neighbour("ezs42", "northwest") == "ezefx")
stopifnot(geohash.neighbour("ezs42", "southeast") == "ezs41")
stopifnot(geohash.neighbour("ezs42", "southwest") == "ezefp")
stopifnot(is.na(geohash.neighbour("u", "north")))
stopifnot(geohash.neighbour("r", "east") == geohash.encode(-157.5, -22.5, 1L))

s <- geohash.cell.size(5)
stopifnot(identical(names(s), c("bits", "lon.bits", "lat.bits", "width.deg", "height.deg",
  "width.m", "height.m")))
stopifnot(s$bits == 25, s$lon.bits == 13, s$lat.bits == 12, near(s$width.deg, 360 / 8192))
stopifnot(near(s$width.m, s$height.m), near(geohash.cell.size(1, lat = 60)$width.m,
  geohash.cell.size(1)$width.m / 2))

cv <- geohash.cover(c(xmin = -5.62, ymin = 42.58, xmax = -5.58, ymax = 42.62), 5)
stopifnot(identical(cv, c("ezs40", "ezs41", "ezs42", "ezs43")))
stopifnot(inherits(try(geohash.cover(c(xmin = -10, ymin = -10, xmax = 10, ymax = 10), 6,
  max.cells = 10), silent = TRUE), "try-error"))
