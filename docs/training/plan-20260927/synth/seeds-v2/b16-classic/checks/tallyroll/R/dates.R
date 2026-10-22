d <- function(x) as.Date(x)

# parseDates
pd <- parseDates(c("2024-03-15", "15.03.2024", "03/15/2024", NA))
stopifnot(identical(pd[1:3], rep(d("2024-03-15"), 3)), is.na(pd[4]))
w <- tryCatch(parseDates(c("2024-01-01", "soon")), warning = function(w) conditionMessage(w))
stopifnot(grepl("^1 value", w))

# daysBetween / weeksBetween
stopifnot(daysBetween("2024-01-01", "2024-01-29") == 28)
stopifnot(daysBetween("2024-01-29", "2024-01-01") == -28)
stopifnot(weeksBetween("2024-01-01", "2024-01-28") == 3)
stopifnot(weeksBetween("2024-01-29", "2024-01-01") == -4)

# monthsBetween: complete months in both directions
stopifnot(identical(monthsBetween("2024-01-15", c("2024-03-14", "2024-03-15")), c(1L, 2L)))
stopifnot(monthsBetween("2023-11-30", "2024-02-29") == 2L)
stopifnot(monthsBetween("2024-03-14", "2024-01-15") == -1L)
stopifnot(monthsBetween("2024-03-15", "2024-01-15") == -2L)

# monthStarts
ms <- monthStarts("2024-01-20", "2024-04-02")
stopifnot(length(ms) == 4, ms[1] == d("2024-01-01"), ms[4] == d("2024-04-01"))
stopifnot(inherits(try(monthStarts("2024-05-01", "2024-01-01"), silent = TRUE), "try-error"))

# workingDays
wd <- workingDays("2024-12-23", "2024-12-29", holidays = d(c("2024-12-25", "2024-12-26")))
stopifnot(length(wd) == 3, all(wd == d(c("2024-12-23", "2024-12-24", "2024-12-27"))))

# periodLabel
pl <- d(c("2024-02-10", "2024-11-30"))
stopifnot(identical(periodLabel(pl, "quarter"), c("2024-Q1", "2024-Q4")))
stopifnot(identical(periodLabel(pl), c("2024-02", "2024-11")))
stopifnot(identical(periodLabel(pl, "year"), c("2024", "2024")))

# addMonths: clamps month ends and keeps NA
stopifnot(addMonths(d("2024-01-31"), 1) == d("2024-02-29"))
stopifnot(addMonths(d("2023-01-31"), 1) == d("2023-02-28"))
stopifnot(addMonths(d("2024-03-15"), -3) == d("2023-12-15"))
am <- addMonths(d(c("2024-12-31", NA)), 2)
stopifnot(am[1] == d("2025-02-28"), is.na(am[2]))
