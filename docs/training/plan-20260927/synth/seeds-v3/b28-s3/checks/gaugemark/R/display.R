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
registerS3method("format", "gm_reading", format.gm_reading)
registerS3method("format", "gm_range", format.gm_range)

r = newReading(c(1.5, NA, 1234.5), "cm")
stopifnot(identical(format(r), c("1.5 cm", "NA", "1230 cm")))
stopifnot(identical(format(r, digits = 5), c("1.5 cm", "NA", "1234.5 cm")))
g = newRange(c(1, NA), c(2.25, 3), "m")
stopifnot(identical(format(g), c("1-2.25 m", "NA")))
stopifnot(identical(format(g, digits = 2), c("1-2.2 m", "NA")))

out = capture.output(res <- print(r))
stopifnot(identical(res, r), identical(out[1], "<gm_reading[3]>"), grepl("1.5 cm", out[2]))
stopifnot(identical(capture.output(print(newReading())), "<gm_reading[0]>"))
out = capture.output(res <- print(g))
stopifnot(identical(res, g), identical(out[1], "<gm_range[2]>"), grepl("1-2.25 m", out[2]))
stopifnot(identical(capture.output(print(newRange())), "<gm_range[0]>"))

stopifnot(length(g) == 2L, length(newRange()) == 0L)

stopifnot(identical(r[c(1, 3)], newReading(c(1.5, 1234.5), "cm")), identical(r[], r))
stopifnot(identical(attr(r[2], "unit"), "cm"))
stopifnot(identical(g[2], newRange(NA_real_, 3, "m")), identical(g[], g))
stopifnot(identical(unclass(g[c(TRUE, FALSE)])$upper, 2.25))
stopifnot(identical(format(r, notation = "fixed", digits = 1), c("1.5 cm", "NA", "1234.5 cm")))
stopifnot(identical(format(r, notation = "scientific", digits = 1)[1], "1.5e+00 cm"))
stopifnot(inherits(try(format(r, notation = "roman"), silent = TRUE), "try-error"))
stopifnot(identical(format(g, notation = "fixed", digits = 1), c("1.0-2.2 m", "NA")))
stopifnot(identical(format(g, notation = "scientific", digits = 0), c("1e+00-2e+00 m", "NA")))
