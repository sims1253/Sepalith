r = regimen("amoxicillin", 500, 8, 6, start.h = 2)
stopifnot(is(r, "Regimen"), identical(r@n.doses, 6L), r@route == "oral")
fails = function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(regimen("x", 1, 0, 1)), fails(regimen(1, 1, 8, 1)), fails(regimen("x", "1", 8, 1)))
stopifnot(fails(regimen("x", 1, 1:2, 1)), fails(regimen("x", 1, "weekly", 1)),
  fails(regimen("x", 1, 8, 1.5)))
stopifnot(regimen("X", 1, "QD", 1)@interval.h == 24, regimen("x", 1, "bid", 1)@interval.h == 12)
stopifnot(regimen("x", 1, "tid", 1)@interval.h == 8, regimen("x", 1, "qid", 1)@interval.h == 6)
stopifnot(regimen("Amoxicillin", 1, 8, 1)@drug == "amoxicillin")

stopifnot(isTRUE(valid.regimen(r)))
two = r
two@amount = c(1, 2)
two@route = character()
stopifnot(identical(valid.regimen(two),
  c("`amount` must be a single value.", "`route` must be a single value.")))
bad = r
bad@drug = ""
bad@amount = -1
bad@interval.h = NA_real_
bad@n.doses = 0L
bad@start.h = -2
bad@route = "nasal"
msgs = valid.regimen(bad)
stopifnot(length(msgs) == 6L, grepl("^`drug`", msgs[1]), grepl("^`amount`", msgs[2]))
stopifnot(grepl("^`interval.h`", msgs[3]), grepl("^`n.doses`", msgs[4]),
  grepl("^`start.h`", msgs[5]))
stopifnot(grepl("^`route`", msgs[6]))

amount(r) = "250"
stopifnot(r@amount == 250, inherits(try(set.regimen.amount(r, 1:2), silent = TRUE), "try-error"))
stopifnot(inherits(try(set.regimen.amount(r, 0), silent = TRUE), "try-error"))
stopifnot(fails(set.regimen.amount(r, TRUE)), fails(set.regimen.interval(r, "x")))
stopifnot(fails(set.regimen.n.doses(r, list(2))), fails(set.regimen.n.doses(r, NA)))
interval(r) = 12
stopifnot(r@interval.h == 12,
  inherits(try(set.regimen.interval(r, -1), silent = TRUE), "try-error"))
stopifnot(inherits(try(set.regimen.interval(r, c(6, 8)), silent = TRUE), "try-error"))
n.doses(r) = 4.0
stopifnot(identical(r@n.doses, 4L),
  inherits(try(set.regimen.n.doses(r, 0), silent = TRUE), "try-error"))
stopifnot(inherits(try(set.regimen.n.doses(r, 1:3), silent = TRUE), "try-error"))

s = as(r, "data.frame")
stopifnot(identical(names(s), c("dose", "time.h", "amount", "cumulative")), nrow(s) == 4L)
stopifnot(identical(s$time.h, c(2, 14, 26, 38)), identical(s$cumulative, c(250, 500, 750, 1000)))
stopifnot(identical(regimen.schedule(r), s))

tot = regimen.totals(r)
stopifnot(identical(names(tot), c("total.mg", "daily.mg", "duration.h", "last.dose.h",
  "total.mg.kg", "daily.mg.kg")))
stopifnot(tot$total.mg == 1000, tot$daily.mg == 500, tot$duration.h == 48, tot$last.dose.h == 38)
stopifnot(is.na(tot$total.mg.kg), is.na(tot$daily.mg.kg), fails(regimen.totals(r, 0)))
tk = regimen.totals(r, weight.kg = 50)
stopifnot(tk$total.mg.kg == 20, tk$daily.mg.kg == 10)

out = capture.output(res <- show.regimen(r))
stopifnot(length(out) == 8L, grepl("amoxicillin 250 mg oral q12h x 4", out[1]), identical(res, r))
stopifnot(grepl("dose 3 at 26 h", out[7]), grepl("1 more, last at 38 h", out[8]))
stopifnot(length(capture.output(show.regimen(r, n.show = 4L))) == 8L)
stopifnot(grepl("daily:    500 mg", out[3]), identical(capture.output(show(r)), out))
