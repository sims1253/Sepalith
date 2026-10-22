# skipped: needs dplyr (tenure_years, hires_in_period, leavers_in_period,
# attrition_by_group)
# skipped: needs lubridate (monthly_attrition, cohort_retention)

# tenure_band
b <- tenure_band(c(0, 0.5, 1, 4, 12))
stopifnot(is.ordered(b), identical(as.character(b), c("<1y", "<1y", "1-3y", "3-5y", "10y+")))
stopifnot(identical(levels(tenure_band(1, breaks = c(2, 4))), c("<2y", "2-4y", "4y+")))

# headcount_at
staff <- data.frame(
  employee_id = 1:4,
  hire_date = as.Date(c("2020-01-01", "2021-06-01", NA, "2023-01-01")),
  term_date = as.Date(c(NA, "2022-01-01", NA, NA))
)
stopifnot(headcount_at(staff, as.Date("2021-12-31")) == 2L)
stopifnot(headcount_at(staff, as.Date("2022-01-01")) == 1L)
stopifnot(headcount_at(staff, as.Date("2024-01-01")) == 2L)
