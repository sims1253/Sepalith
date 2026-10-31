unitFactor = function(unit) {
  factors = c(mm = 1e-3, cm = 1e-2, m = 1, km = 1e3, `in` = 0.0254)
  factors[[match.arg(unit, names(factors))]]
}
newReading = function(value = double(), unit = "m") {
  structure(value, unit = unit, class = "gm_reading")
}
newRange = function(lower = double(), upper = double(), unit = "m") {
  structure(list(lower = lower, upper = upper), unit = unit, class = "gm_range")
}

r = newReading(c(150, 20), "cm")
stopifnot(all.equal(as.double(unclass(convertUnit(r, "m"))), c(1.5, 0.2)))
stopifnot(identical(attr(convertUnit(r, "mm"), "unit"), "mm"), identical(convertUnit(r, "cm"), r))
stopifnot(inherits(try(convertUnit(r, "ft"), silent = TRUE), "try-error"))

g = newRange(c(1, 2), c(3, 4), "km")
gm = convertUnit(g, "m")
stopifnot(identical(unclass(gm)$lower, c(1000, 2000)), identical(unclass(gm)$upper, c(3000, 4000)))
stopifnot(identical(attr(gm, "unit"), "m"), identical(convertUnit(g, "km"), g))

stopifnot(all.equal(convertUnit(c(1, 2), "cm", from = "in"), c(2.54, 5.08)))
stopifnot(inherits(try(convertUnit(1, "cm"), silent = TRUE), "try-error"))
stopifnot(inherits(try(convertUnit("a", "cm", from = "m"), silent = TRUE), "try-error"))

df = as.data.frame(r)
stopifnot(identical(names(df), c("value", "unit")), identical(df$unit, c("cm", "cm")))
df = as.data.frame(r, row.names = c("a", "b"), metres = TRUE)
stopifnot(all.equal(df$metres, c(1.5, 0.2)), identical(rownames(df), c("a", "b")))

dg = as.data.frame(g)
stopifnot(identical(names(dg), c("lower", "upper", "unit")), identical(dg$upper, c(3, 4)))
dg = as.data.frame(g, metres = TRUE, row.names = c("x", "y"))
stopifnot(identical(dg$lower_metres, c(1000, 2000)), identical(dg$upper_metres, c(3000, 4000)))
stopifnot(identical(rownames(dg), c("x", "y")))
