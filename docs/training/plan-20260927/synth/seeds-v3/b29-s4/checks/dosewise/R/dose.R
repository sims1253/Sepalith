stopifnot(identical(names(mass.factors()), c("g", "mg", "mcg", "ng")),
  mass.factors()[["mcg"]] == 0.001)
stopifnot(identical(dose.routes, c("oral", "iv", "sc", "im")))

d = dose(c(250, 500), route = "IV")
stopifnot(is(d, "Dose"), d@route == "iv", d@unit == "mg", identical(d@amount, c(250, 500)))
fails = function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(dose(-1)), fails(dose("1")), fails(dose(1, unit = c("mg", "g"))),
  fails(dose(1, route = 2)))
stopifnot(dose(1, "Grams")@unit == "g", dose(1, "milligram")@unit == "mg",
  dose(1, "ug")@unit == "mcg")
stopifnot(dose(1, "nanograms")@unit == "ng", fails(dose(1, "pound")))
stopifnot(dose(1, route = "PO")@route == "oral", dose(1, route = "intravenous")@route == "iv")
stopifnot(dose(1, route = "subcutaneous")@route == "sc",
  dose(1, route = "intramuscular")@route == "im")

stopifnot(isTRUE(valid.dose(d)))
bad = d
bad@amount = c(1, NA)
bad@unit = "lb"
bad@route = "nasal"
msgs = valid.dose(bad)
stopifnot(length(msgs) == 3L, grepl("^`amount`", msgs[1]), grepl("^`unit`", msgs[2]))
stopifnot(grepl("^`route`", msgs[3]))

g = dose.convert(d, "g")
stopifnot(g@unit == "g", isTRUE(all.equal(g@amount, c(0.25, 0.5))))
stopifnot(isTRUE(all.equal(dose.convert(d, "mcg")@amount, c(250000, 5e5))))
stopifnot(identical(dose.convert(d)@amount, d@amount))

s = d + dose(0.5, "g", "iv")
stopifnot(s@unit == "mg", identical(s@amount, c(750, 1000)))
stopifnot(identical((d - dose(100, route = "iv"))@amount, c(150, 400)))
stopifnot(inherits(try(d + dose(1), silent = TRUE), "try-error"))
stopifnot(inherits(try(d * dose(1, route = "iv"), silent = TRUE), "try-error"))
stopifnot(inherits(try(d - dose(300, route = "iv"), silent = TRUE), "try-error"))
stopifnot(identical((d * 2)@amount, c(500, 1000)), identical((d / 5)@amount, c(50, 100)))
stopifnot(inherits(try(d + 1, silent = TRUE), "try-error"))
stopifnot(identical((3 * d)@amount, c(750, 1500)),
  inherits(try(3 - d, silent = TRUE), "try-error"))

stopifnot(identical(d == dose(c(0.25, 1), "g"), c(TRUE, FALSE)),
  identical(d != g, c(FALSE, FALSE)))
stopifnot(identical(d < dose(300), c(TRUE, FALSE)), identical(d > dose(300), c(FALSE, TRUE)))
stopifnot(identical(d <= dose(250), c(TRUE, FALSE)), identical(d >= dose(500), c(FALSE, TRUE)))

stopifnot(identical(format.dose(d), c("250 mg iv", "500 mg iv")))
stopifnot(identical(format(dose(1 / 3, "g")), "0.333 g oral"))
stopifnot(identical(format.dose(dose(1234.5), 2), "1200 mg oral"))
stopifnot(identical(format.dose(dose(c(2500, 0.5, 0)), scale = TRUE),
  c("2.5 g oral", "500 mcg oral", "0 ng oral")))
out = capture.output(show(d))
stopifnot(grepl("<Dose>", out[1]), grepl("250 mg iv", out[2]), length(out) == 3L)
stopifnot(any(grepl("empty", capture.output(show(new("Dose"))))))
