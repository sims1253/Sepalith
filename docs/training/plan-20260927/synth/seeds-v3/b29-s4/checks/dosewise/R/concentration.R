stopifnot(identical(names(conc.units()), c("mg/L", "ug/mL", "ng/mL", "ug/L")),
  conc.units()[["ng/mL"]] == 0.001)

x = concentration("vancomycin", c(4, 0, 1, 3, 2), c(4, 0, 32, 8, 16))
stopifnot(is(x, "Concentration"), identical(x@value, c(0, 32, 16, 8, 4)))
stopifnot(identical(concentration("v", c(4, 0), c(5, 0))@time.h, c(0, 4)))
stopifnot(inherits(try(concentration("v", 1:2, 1, unit = "mmol"), silent = TRUE), "try-error"))

stopifnot(isTRUE(valid.concentration(x)))
bad = x
bad@analyte = ""
bad@unit = "g"
bad@time.h = c(0, 0, 1, 2, 3)
bad@value = c(1, -1, 2, 3, 4)
msgs = valid.concentration(bad)
stopifnot(length(msgs) == 4L, grepl("^`analyte`", msgs[1]), grepl("^`unit`", msgs[2]))
stopifnot(grepl("^`time.h`", msgs[3]), grepl("^`value`", msgs[4]))
neg = x
neg@time.h = neg@time.h - 1
stopifnot(identical(valid.concentration(neg), "`time.h` must not be negative."))

ng = conc.convert(x, "ng/mL")
stopifnot(ng@unit == "ng/mL", isTRUE(all.equal(ng@value, c(0, 32000, 16000, 8000, 4000))))
stopifnot(isTRUE(all.equal(conc.convert(ng, "mg/L")@value, x@value)),
  identical(conc.convert(x)@value, x@value))

s = conc.summary(x)
stopifnot(identical(names(s), c("n", "cmax", "tmax", "cmin", "tlast", "clast", "auc", "aumc",
  "mrt", "lambda.z", "half.life")))
stopifnot(s$n == 5L, s$cmax == 32, s$tmax == 1, s$cmin == 0, s$tlast == 4, s$clast == 4)
stopifnot(s$auc == 58, s$aumc == 96, isTRUE(all.equal(s$mrt, 96 / 58)))
stopifnot(isTRUE(all.equal(s$lambda.z, log(2))), isTRUE(all.equal(s$half.life, 1)))
y = concentration("v", 1:3, c(NA, 2, NA))
sy = conc.summary(y)
stopifnot(sy$n == 1L, sy$auc == 0, sy$tmax == 2, is.na(sy$mrt), is.na(sy$lambda.z))
stopifnot(inherits(try(conc.summary(concentration("v", 1, NA)), silent = TRUE), "try-error"))

df = as(x, "data.frame")
stopifnot(is.data.frame(df), identical(names(df), c("time.h", "value")),
  attr(df, "unit") == "mg/L")
m = as(x, "matrix")
stopifnot(is.matrix(m), identical(colnames(m), c("time.h", "value")), attr(m, "unit") == "mg/L")
num = as(x, "numeric")
stopifnot(identical(unname(num), x@value), identical(names(num), c("0", "1", "2", "3", "4")))
ch = conc.as(concentration("v", 2, 1.5), "character")
stopifnot(identical(ch, "1.5 mg/L @ 2h"), length(as(x, "character")) == 5L)
li = as(x, "list")
stopifnot(identical(names(li), c("analyte", "unit", "time.h", "value")),
  identical(li$value, x@value))

out = capture.output(res <- show.concentration(x))
stopifnot(length(out) == 4L, grepl("vancomycin, 5 of 5 samples", out[1]), identical(res, x))
stopifnot(grepl("cmax: 32 mg/L at 1 h", out[2]), grepl("auc:  58", out[3]))
stopifnot(grepl("t1/2: 1 h", out[4]))
stopifnot(length(capture.output(show(y))) == 3L)
stopifnot(length(capture.output(show(concentration("v", 1, NA)))) == 1L)
