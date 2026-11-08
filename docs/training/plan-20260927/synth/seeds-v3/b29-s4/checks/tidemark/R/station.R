st <- station("nl042", "Hoek van Holland", 51.9775, 4.12, datum = "NAP")
stopifnot(is(st, "Station"), st@id == "NL042", st@datum == "NAP", isTRUE(st@active))
stopifnot(inherits(try(station("NL042", "x", 95, 4), silent = TRUE), "try-error"))

stopifnot(isTRUE(validStation(st)))
two <- st
two@lat <- c(1, 2)
two@datum <- character()
stopifnot(identical(validStation(two),
  c("`lat` must have length 1.", "`datum` must have length 1.")))
bad <- st
bad@id <- "nl42"
bad@name <- "  "
bad@lat <- -91
bad@lon <- 181
bad@datum <- ""
bad@active <- NA
msgs <- validStation(bad)
stopifnot(length(msgs) == 6L, grepl("^`id`", msgs[1]), grepl("^`name`", msgs[2]))
stopifnot(grepl("^`lat`", msgs[3]), grepl("^`lon`", msgs[4]), grepl("^`datum`", msgs[5]))
stopifnot(grepl("^`active`", msgs[6]))

stationName(st) <- "  Hoek  "
stopifnot(setStationName(st, "Hoek   van\tHolland")@name == "Hoek van Holland")
stopifnot(inherits(try(setStationName(st, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(setStationName(st, NA_character_), silent = TRUE), "try-error"))
stopifnot(st@name == "Hoek",
  inherits(try(setStationName(st, c("a", "b")), silent = TRUE), "try-error"))
stopifnot(inherits(try(setStationName(st, ""), silent = TRUE), "try-error"))
stationLat(st) <- 51.12345678
stopifnot(st@lat == 51.123457, inherits(try(setStationLat(st, 100), silent = TRUE), "try-error"))
stopifnot(inherits(try(setStationLat(st, 1:2), silent = TRUE), "try-error"))
stopifnot(inherits(try(setStationLat(st, "1"), silent = TRUE), "try-error"))
stationLon(st) <- -4.5
stopifnot(st@lon == -4.5, inherits(try(setStationLon(st, 200), silent = TRUE), "try-error"))
stopifnot(inherits(try(setStationLon(st, numeric()), silent = TRUE), "try-error"))
stopifnot(inherits(try(setStationLon(st, NA_real_), silent = TRUE), "try-error"))

l <- as(st, "list")
stopifnot(identical(names(l), c("id", "name", "lat", "lon", "datum", "active")), l$lon == -4.5)
stopifnot(identical(stationToList(st), l))
back <- as(l, "Station")
stopifnot(identical(back, st))
d <- listToStation(list(id = "DE007", name = "Cuxhaven", lat = 53.87, lon = 8.72))
stopifnot(d@datum == "LAT", isTRUE(d@active))
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(listToStation(list(name = "x", lat = 1, lon = 1))))
stopifnot(fails(listToStation(list(id = "DE007", lat = 1, lon = 1))))
stopifnot(fails(listToStation(list(id = "DE007", name = "x", lon = 1))))
stopifnot(fails(listToStation(list(id = "DE007", name = "x", lat = 1))))
df <- as(st, "data.frame")
stopifnot(is.data.frame(df), nrow(df) == 1L, ncol(df) == 6L, df$id == "NL042")

out <- capture.output(res <- showStation(st))
stopifnot(length(out) == 4L, grepl("NL042 Hoek \\(active\\)", out[1]), identical(res, st))
stopifnot(grepl("51.1235 N", out[2]), grepl("4.5000 W", out[3]), grepl("NAP", out[4]))
stopifnot(grepl("51.12 N", capture.output(showStation(st, 2))[2]))
d@active <- FALSE
stopifnot(grepl("inactive", capture.output(show(d))[1]))
