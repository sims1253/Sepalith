# present_value / future_value
stopifnot(all.equal(present_value(110, 0.1, 1), 100))
stopifnot(all.equal(future_value(100, 0.1, 1), 110))
stopifnot(inherits(try(present_value(1, 0.1, -1), silent = TRUE), "try-error"))

# annuity_pv / annuity_fv
stopifnot(all.equal(annuity_pv(100, 0, 12), 1200))
stopifnot(all.equal(round(annuity_pv(100, 0.05, 10), 2), 772.17))
stopifnot(all.equal(annuity_fv(100, 0.05, 10, due = TRUE), annuity_fv(100, 0.05, 10) * 1.05))
stopifnot(all.equal(round(annuity_fv(100, 0.05, 10), 2), 1257.79))

# annuity_payment
pmt <- annuity_payment(10000, 0.01, 24)
stopifnot(all.equal(annuity_pv(pmt, 0.01, 24), 10000))
stopifnot(all.equal(annuity_payment(1200, 0, 12), 100))
stopifnot(annuity_payment(10000, 0.01, 24, balloon = 2000) < pmt)

# net_present_value
stopifnot(all.equal(net_present_value(c(-1000, 300, 400, 500), 0), 200))
stopifnot(net_present_value(numeric(0), 0.1) == 0)

# internal_rate
flows <- c(-1000, 300, 400, 500)
irr <- internal_rate(flows)
stopifnot(abs(net_present_value(flows, irr)) < 1e-4, irr > 0.08, irr < 0.09)
stopifnot(is.na(suppressWarnings(internal_rate(c(100, 200)))))
stopifnot(is.na(suppressWarnings(internal_rate(c(-100, NA, 110)))))
stopifnot(abs(internal_rate(c(-100, 0, 121)) - 0.1) < 1e-6)

# check_rate
stopifnot(identical(check_rate(0.05), 0.05))
stopifnot(inherits(try(check_rate("a"), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_rate(-2), silent = TRUE), "try-error"))
