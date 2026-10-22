# Classes and constructors live in AllClasses.R; minimal copies for this check
dimsOf <- function(...) {
  out <- setNames(numeric(7), c("length", "mass", "time", "current", "temperature",
    "amount", "luminosity"))
  d <- c(...)
  out[names(d)] <- d
  out
}
setClass("Unit", representation(symbol = "character", dims = "numeric",
  scale = "numeric", offset = "numeric"))
setClass("Quantity", contains = "numeric", representation(unit = "Unit"))
km <- new("Unit", symbol = "km", dims = dimsOf(length = 1), scale = 1000, offset = 0)
m <- new("Unit", symbol = "m", dims = dimsOf(length = 1), scale = 1, offset = 0)
lookupUnit <- function(symbol) m

q <- new("Quantity", c(1, 2, 3), unit = km)
stopifnot(identical(magnitude(q), c(1, 2, 3)), identical(unitOf(q)@symbol, "km"))
magnitude(q) <- c(4, 5, 6)
stopifnot(identical(magnitude(q), c(4, 5, 6)))
stopifnot(inherits(try(magnitude(q) <- 1, silent = TRUE), "try-error"))

unitOf(q) <- "m"
stopifnot(unitOf(q)@symbol == "m", identical(magnitude(q), c(4, 5, 6)))

stopifnot(identical(formatDims(km), "L"))
stopifnot(identical(formatDims(new("Unit", symbol = "N", scale = 1, offset = 0,
  dims = dimsOf(mass = 1, length = 1, time = -2))), "L M T^-2"))
stopifnot(identical(formatDims(new("Unit", symbol = "1", dims = dimsOf(), scale = 1,
  offset = 0)), "1"))

sub <- q[2:3]
stopifnot(is(sub, "Quantity"), identical(magnitude(sub), c(5, 6)), length(q[]) == 3L)
stopifnot(any(grepl("3 value", capture.output(show(q)))))
stopifnot(any(grepl("<Unit> km \\[L\\] x 1000", capture.output(show(km)))))
