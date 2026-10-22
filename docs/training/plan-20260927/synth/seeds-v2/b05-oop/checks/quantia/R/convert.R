# Classes and helpers live in AllClasses.R / accessors.R / arith.R; minimal copies
setClass("Unit", representation(symbol = "character", dims = "numeric",
  scale = "numeric", offset = "numeric"))
setClass("Quantity", contains = "numeric", representation(unit = "Unit"))
baseDimensions <- function() {
  c("length", "mass", "time", "current", "temperature", "amount", "luminosity")
}
makeDims <- function(dims = numeric()) {
  out <- setNames(numeric(7), baseDimensions())
  out[names(dims)] <- dims
  out
}
unit <- function(symbol, dims = numeric(), scale = 1, offset = 0) {
  new("Unit", symbol = symbol, dims = makeDims(dims), scale = scale, offset = offset)
}
formatDims <- function(unit) paste(names(unit@dims)[unit@dims != 0], collapse = " ")
sameDims <- function(u1, u2) isTRUE(all.equal(u1@dims, u2@dims, check.attributes = FALSE))

stopifnot(registerDefaultUnits() == 15L)
stopifnot(lookupUnit("km")@scale == 1000)
stopifnot(inherits(try(lookupUnit("furlong"), silent = TRUE), "try-error"))
stopifnot(inherits(try(registerUnit(unit("km", c(length = 1))), silent = TRUE), "try-error"))

lens <- listUnits("length")
stopifnot(identical(lens$symbol, c("cm", "km", "m", "mi")), nrow(listUnits()) == 15L)
stopifnot(unregisterUnit("mi"), !unregisterUnit("mi"), nrow(listUnits()) == 14L)

q <- new("Quantity", 5, unit = lookupUnit("km"))
stopifnot(all.equal(convertTo(q, "m")@.Data, 5000))
t1 <- convertTo(new("Quantity", 100, unit = lookupUnit("degC")), "K")
stopifnot(all.equal(t1@.Data, 373.15))
stopifnot(inherits(try(convertTo(q, "kg"), silent = TRUE), "try-error"))
