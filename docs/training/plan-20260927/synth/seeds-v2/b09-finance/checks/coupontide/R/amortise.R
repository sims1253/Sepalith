# skipped: amortisation_schedule(), remaining_balance() and refinance_comparison()
# need annuity_payment() and friends from R/tvm.R
sched <- new_amort_schedule(
  data.frame(
    period = 1:3, payment = c(40, 40, 40), interest = c(3, 2, 1),
    principal = c(37, 38, 39), balance = c(77, 39, 0)
  ),
  principal = 114, annual_rate = 0.12, per_year = 2
)

# new_amort_schedule
stopifnot(inherits(sched, "amort_schedule"), attr(sched, "per_year") == 2)

# total_interest
stopifnot(total_interest(sched) == 6)

# annual_summary
yrs <- annual_summary(sched)
stopifnot(nrow(yrs) == 2, all(yrs$interest == c(5, 1)), all(yrs$balance == c(39, 0)))

# format / print / summary methods
fmt <- format(sched, digits = 1)
stopifnot(fmt$payment[1] == "40.0", is.character(fmt$balance))
out <- capture.output(res <- print(sched, n = 2))
stopifnot(identical(res, sched), any(grepl("1 more payments", out)))
s <- summary(sched)
stopifnot(s[["total_paid"]] == 120, all.equal(s[["interest_share"]], 0.05))
