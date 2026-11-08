near <- function(a, b) isTRUE(all.equal(a, b, tolerance = 1e-9))
stopifnot(identical(names(coord.patterns), c("decimal", "dms", "dm", "hemisphere")))

raw <- c(" 52.5 ", "52 deg 31' 12\" north", "13d 24.5' e", "45.25 South",
  "1 degree 2 minutes 3 seconds west", "10o 5'' S", "12 30 W")
cl <- coord.clean(raw)
stopifnot(identical(cl[1:4], c("52.5", "52D 31M 12S N", "13D 24.5M E", "45.25 S")))
stopifnot(identical(cl[5], "1D 2M 3S W"), identical(cl[6], "10D 5S S"))
stopifnot(identical(coord.clean("7 EAST"), "7 E"))

stopifnot(near(coord.parse(c("52.5", "-3.5", "x"), "decimal")[1:2], c(52.5, -3.5)))
stopifnot(is.na(coord.parse("x", "decimal")))
stopifnot(near(coord.parse("52D 31M 12S N", "dms"), 52.52))
stopifnot(near(coord.parse("1D 2M 3S W", "dms"), -(1 + 2 / 60 + 3 / 3600)))
stopifnot(near(coord.parse("13D 24.5M E", "dm"), 13 + 24.5 / 60))
stopifnot(near(coord.parse("13D 30M S", "dm"), -13.5))
stopifnot(near(coord.parse(c("45.25 S", "45.25N"), "hemisphere"), c(-45.25, 45.25)))
stopifnot(is.na(coord.parse("45.25", "hemisphere")))

stopifnot(identical(coord.detect(cl), c("decimal", "dms", "dm", "hemisphere", "dms",
  NA, NA)))
stopifnot(identical(coord.detect(c("1D 2M", "3 W")), c("dm", "hemisphere")))

pp <- coord.parse.pair(c("52.5, 13.4", "52d 30m N; 13d 24m E", "100, 5"))
stopifnot(near(pp$lat[1:2], c(52.5, 52.5)), near(pp$lon, c(13.4, 13.4, 5)), is.na(pp$lat[3]))
pl <- coord.parse.pair("13.4;52.5", order = "lonlat")
stopifnot(pl$lon == 13.4, pl$lat == 52.5)
stopifnot(inherits(try(coord.parse.pair("1, 2, 3"), silent = TRUE), "try-error"))

rep <- coord.format.report(raw)
stopifnot(identical(rep$format, c("decimal", "dms", "dm", "hemisphere", "unparsed")))
stopifnot(identical(rep$n, c(1L, 2L, 1L, 1L, 2L)), isTRUE(all.equal(sum(rep$share), 1)))
stopifnot(identical(rep$example[1:4], c(" 52.5 ", "52 deg 31' 12\" north", "13d 24.5' e",
  "45.25 South")), rep$example[5] == "10o 5'' S")
stopifnot(is.na(coord.format.report("1.5")$example[2]))
