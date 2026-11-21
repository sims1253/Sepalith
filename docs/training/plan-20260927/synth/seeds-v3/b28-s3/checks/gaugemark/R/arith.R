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

a = newReading(c(1, 2), "m")
b = newReading(c(50, 300), "cm")
stopifnot(identical(inUnit(a, "cm"), c(100, 200)), identical(inUnit(3L, "km"), 3))
stopifnot(identical(inUnit(newRange(1, 2, "m"), "mm"), list(lower = 1000, upper = 2000)))
stopifnot(identical(commonUnit(a, b), "cm"), identical(commonUnit(b, a), "cm"))
stopifnot(identical(commonUnit(2, a), "m"), identical(commonUnit(a, 2), "m"))

s = a + b
stopifnot(identical(attr(s, "unit"), "cm"), all.equal(as.double(unclass(s)), c(150, 500)))
stopifnot(all.equal(as.double(unclass(b - a)), c(-50, 100)))
stopifnot(identical(as.double(unclass(a * 2)), c(2, 4)), identical(attr(3 * a, "unit"), "m"))
stopifnot(identical(as.double(unclass(a / 2)), c(0.5, 1)), all.equal(a / b, c(2, 2 / 3)))
stopifnot(identical(a > b, c(TRUE, FALSE)))
stopifnot(identical(a == newReading(c(100, 1), "cm"), c(TRUE, FALSE)))
stopifnot(identical(a <= 1, c(TRUE, FALSE)))
stopifnot(inherits(try(a + 1, silent = TRUE), "try-error"))
stopifnot(inherits(try(a * b, silent = TRUE), "try-error"))
stopifnot(inherits(try(2 / a, silent = TRUE), "try-error"))
stopifnot(inherits(try(a^2, silent = TRUE), "try-error"))
stopifnot(inherits(try(-a, silent = TRUE), "try-error"))

g = newRange(c(1, 5), c(2, 6), "m")
h = newRange(c(300, 100), c(400, 600), "cm")
stopifnot(identical(g < h, c(TRUE, FALSE)), identical(g > h, c(FALSE, FALSE)))
stopifnot(identical(g == g, c(TRUE, TRUE)), identical(g != h, c(TRUE, TRUE)))
stopifnot(identical(g >= newRange(0, 1, "m"), c(TRUE, TRUE)), identical(g <= h, c(TRUE, FALSE)))
stopifnot(inherits(try(g == 1, silent = TRUE), "try-error"))
stopifnot(identical(unclass(g + 1)$lower, c(2, 6)), identical(unclass(1 + g)$upper, c(3, 7)))
stopifnot(identical(unclass(g - 0.5)$lower, c(0.5, 4.5)))
stopifnot(identical(unclass(g * 2)$upper, c(4, 12)), identical(unclass(g / 2)$lower, c(0.5, 2.5)))
stopifnot(inherits(try(1 - g, silent = TRUE), "try-error"))
stopifnot(inherits(try(g + h, silent = TRUE), "try-error"))
stopifnot(inherits(try(g * -2, silent = TRUE), "try-error"))
stopifnot(inherits(try(g + "a", silent = TRUE), "try-error"))
stopifnot(inherits(try(g %/% 2, silent = TRUE), "try-error"))
stopifnot(inherits(try(-g, silent = TRUE), "try-error"))

x = newReading(c(1.26, 2.71), "mm")
stopifnot(identical(as.double(unclass(floor(x))), c(1, 2)))
stopifnot(identical(attr(round(x), "unit"), "mm"))
stopifnot(identical(as.double(unclass(round(x, 1))), c(1.3, 2.7)))
stopifnot(identical(as.double(unclass(abs(newReading(-1, "m")))), 1))
stopifnot(inherits(try(log(x), silent = TRUE), "try-error"))
y = newRange(c(1.2, 2.5), c(3.7, 4.1), "km")
stopifnot(identical(unclass(ceiling(y))$lower, c(2, 3)))
stopifnot(identical(unclass(trunc(y))$upper, c(3, 4)))
stopifnot(identical(attr(signif(y, 1), "unit"), "km"))
stopifnot(inherits(try(exp(y), silent = TRUE), "try-error"))

stopifnot(identical(as.double(unclass(min(a, b))), 0.5), identical(attr(max(a, b), "unit"), "m"))
stopifnot(all.equal(as.double(unclass(max(a, b))), 3), all.equal(as.double(unclass(sum(a))), 3))
stopifnot(identical(as.double(unclass(range(a, 5))), c(1, 5)))
stopifnot(identical(as.double(unclass(min(newReading(c(NA, 2), "m"), na.rm = TRUE))), 2))
stopifnot(inherits(try(prod(a), silent = TRUE), "try-error"))

stopifnot(identical(as.double(unclass(min(g, h))), 1), identical(as.double(unclass(max(g, h))), 6))
rg = range(g, h)
stopifnot(inherits(rg, "gm_range"), identical(unclass(rg)$lower, 1))
stopifnot(identical(unclass(rg)$upper, 6))
stopifnot(inherits(try(any(g), silent = TRUE), "try-error"))
