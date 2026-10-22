# Classes and helpers live in AllClasses.R / accessors.R / convert.R; minimal copies
setClass("Unit", representation(symbol = "character", dims = "numeric",
  scale = "numeric", offset = "numeric"))
setClass("Quantity", contains = "numeric", representation(unit = "Unit"))
dimNames <- c("length", "mass", "time", "current", "temperature", "amount", "luminosity")
unit <- function(symbol, dims = numeric(), scale = 1, offset = 0) {
  full <- setNames(numeric(7), dimNames)
  full[names(dims)] <- dims
  new("Unit", symbol = symbol, dims = full, scale = scale, offset = offset)
}
isDimensionless <- function(unit) all(unit@dims == 0)
magnitude <- function(object) object@.Data
convertTo <- function(x, to) {
  new("Quantity", (x@.Data * x@unit@scale + x@unit@offset - to@offset) / to@scale, unit = to)
}

m <- unit("m", c(length = 1))
km <- unit("km", c(length = 1), scale = 1000)
s <- unit("s", c(time = 1))
degC <- unit("degC", c(temperature = 1), offset = 273.15)

stopifnot(sameDims(m, km), !sameDims(m, s))
v <- combineUnits(km, s, "/")
stopifnot(v@symbol == "km/s", v@dims[["time"]] == -1, v@scale == 1000)
stopifnot(inherits(try(combineUnits(degC, s), silent = TRUE), "try-error"))
stopifnot(powerUnit(m, 2)@dims[["length"]] == 2, powerUnit(km, 2)@scale == 1e6)

a <- new("Quantity", 1, unit = km)
b <- new("Quantity", 500, unit = m)
stopifnot(all.equal((a + b)@.Data, 1.5), (a + b)@unit@symbol == "km")
stopifnot(all.equal((a - b)@.Data, 0.5))
stopifnot(inherits(try(a + new("Quantity", 1, unit = s), silent = TRUE), "try-error"))
stopifnot((a * 2)@.Data == 2, (a^2)@unit@dims[["length"]] == 2)
stopifnot(inherits(try(a + 1, silent = TRUE), "try-error"))
stopifnot((2 / new("Quantity", 4, unit = s))@unit@dims[["time"]] == -1)

stopifnot(identical(a > b, TRUE), identical(b < a, TRUE))
tot <- sum(a, b)
stopifnot(all.equal(tot@.Data, 1.5), tot@unit@symbol == "km")
stopifnot(inherits(try(prod(a), silent = TRUE), "try-error"))
