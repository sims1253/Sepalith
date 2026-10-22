# yearfrac_act360 / yearfrac_act365
stopifnot(all.equal(yearfrac_act360("2024-01-01", "2024-07-01"), 182 / 360))
stopifnot(all.equal(yearfrac_act365("2023-01-01", "2024-01-01"), 1))

# yearfrac_30360
stopifnot(all.equal(yearfrac_30360("2024-01-31", "2024-03-31"), 60 / 360))
stopifnot(all.equal(yearfrac_30360("2023-02-28", "2023-08-31"), 0.5))
stopifnot(all.equal(yearfrac_30360("2024-01-15", "2025-01-15"), 1))

# year_fraction
stopifnot(all.equal(year_fraction("2023-01-01", "2024-01-01", "act360"), 365 / 360))
stopifnot(all.equal(year_fraction("2023-01-01", "2024-01-01"), 1))

# add_months / add_one_month
stopifnot(add_months(as.Date("2024-01-31"), 1) == as.Date("2024-02-29"))
stopifnot(add_months(as.Date("2024-03-15"), -3) == as.Date("2023-12-15"))
stopifnot(is.na(add_months(as.Date(NA), 1)))
stopifnot(add_one_month(as.Date("2024-12-01")) == as.Date("2025-01-01"))

# payment_dates
pd <- payment_dates("2024-01-15", "2026-01-15", per_year = 2)
stopifnot(length(pd) == 4, pd[4] == as.Date("2026-01-15"), pd[1] == as.Date("2024-07-15"))
stopifnot(inherits(try(payment_dates("2024-01-01", "2023-01-01"), silent = TRUE), "try-error"))

# accrued_interest
stopifnot(all.equal(accrued_interest(1000, 0.0365, "2024-01-01", "2024-01-11"), 1))
stopifnot(inherits(try(accrued_interest(1, 1, "2024-02-01", "2024-01-01"), silent = TRUE),
                   "try-error"))
