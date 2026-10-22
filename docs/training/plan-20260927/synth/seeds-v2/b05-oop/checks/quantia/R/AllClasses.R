# lookupUnit() lives in convert.R; a tiny stand-in is enough here
lookupUnit <- function(symbol) unit(symbol, c(length = 1), scale = 1000)

stopifnot(length(baseDimensions()) == 7L, baseDimensions()[[5]] == "temperature")
d <- makeDims(c(length = 1, time = -2))
stopifnot(identical(names(d), baseDimensions()), d[["time"]] == -2, sum(d != 0) == 2)
stopifnot(inherits(try(makeDims(c(colour = 1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(makeDims(1), silent = TRUE), "try-error"))

km <- unit("km", c(length = 1), scale = 1000)
stopifnot(is(km, "Unit"), km@scale == 1000, validObject(km, test = TRUE))
stopifnot(inherits(try(unit("x", scale = -1), silent = TRUE), "try-error"))

q <- quantity(c(1.5, 2), km)
stopifnot(is(q, "Quantity"), identical(q@.Data, c(1.5, 2)), q@unit@symbol == "km")
stopifnot(quantity(3, "km")@unit@scale == 1000)
stopifnot(inherits(try(quantity("a", km), silent = TRUE), "try-error"))
stopifnot(inherits(try(new("Quantity", 1, unit = unit("bad", c(length = 1), offset = 2)),
  silent = TRUE), "try-error"))

stopifnot(isDimensionless(unit("1")), !isDimensionless(km))
