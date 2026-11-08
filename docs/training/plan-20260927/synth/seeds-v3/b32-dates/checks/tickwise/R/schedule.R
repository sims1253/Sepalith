stopifnot(months.add(as.Date("2024-01-31"), 1L, 31L) == as.Date("2024-02-29"))
stopifnot(identical(months.add(as.Date("2024-11-10"), 0:2, 31L),
  as.Date(c("2024-11-30", "2024-12-31", "2025-01-31"))))

stopifnot(length(sched.daily("2024-01-01", "2024-01-10")) == 10L)
stopifnot(identical(sched.daily("2024-01-01", "2024-01-10", step = 4L),
  as.Date(c("2024-01-01", "2024-01-05", "2024-01-09"))))
stopifnot(length(sched.daily("2024-01-10", "2024-01-01")) == 0L)

w <- sched.weekly("2024-01-03", "2024-01-31", weekday = 1L)
stopifnot(w[1] == as.Date("2024-01-08"), length(w) == 4L)
stopifnot(length(sched.weekly("2024-01-03", "2024-01-31")) == 5L)
stopifnot(length(sched.weekly("2024-01-30", "2024-01-31", weekday = 1L)) == 0L)

m <- sched.monthly("2024-01-31", "2024-05-15")
stopifnot(identical(m, as.Date(c("2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"))))
stopifnot(identical(sched.monthly("2024-01-10", "2024-03-20", day = 15L),
  as.Date(c("2024-01-15", "2024-02-15", "2024-03-15"))))

q <- sched.quarterly("2024-01-31", "2025-01-31")
stopifnot(identical(q, as.Date(c("2024-01-31", "2024-04-30", "2024-07-31", "2024-10-31",
  "2025-01-31"))))
stopifnot(length(sched.quarterly("2024-01-10", "2024-03-20", day = 5L)) == 0L)

y <- sched.yearly("2024-02-29", "2027-03-01")
stopifnot(identical(y, as.Date(c("2024-02-29", "2025-02-28", "2026-02-28", "2027-02-28"))))
stopifnot(length(sched.yearly("2024-06-01", "2026-05-31")) == 2L)

s <- make.schedule("2024-01-01", "2024-03-31", "monthly")
stopifnot(inherits(s, "tw_schedule"), attr(s, "freq") == "monthly", length(s) == 3L)
stopifnot(length(make.schedule("2024-01-01", "2024-01-07")) == 7L)
stopifnot(length(make.schedule("2024-01-01", "2024-01-31", "weekly", anchor = 5L)) == 4L)
stopifnot(length(make.schedule("2024-01-01", "2024-12-31", "quarterly")) == 4L)
stopifnot(length(make.schedule("2020-01-01", "2024-12-31", "yearly")) == 5L)
stopifnot(inherits(try(make.schedule("2024-01-01", "2030-01-01", max.n = 10L),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(make.schedule(Sys.Date() + 0:1, Sys.Date()), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(make.schedule(Sys.Date(), NA), silent = TRUE), "try-error"))
stopifnot(inherits(try(make.schedule(Sys.Date(), Sys.Date(), "weekly", anchor = 8L),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(make.schedule(Sys.Date(), Sys.Date(), "daily", anchor = 1L),
  silent = TRUE), "try-error"))

p <- schedule.periods(s)
stopifnot(identical(names(p), c("start", "end", "days", "fraction")))
stopifnot(identical(p$days, c(31L, 29L)), all.equal(p$fraction, c(31, 29) / 360))
stopifnot(all.equal(schedule.periods(s, "act365")$fraction, c(31, 29) / 365))
p3 <- schedule.periods(as.Date(c("2024-01-31", "2024-02-29", "2024-03-31")), "thirty360")
stopifnot(all.equal(p3$fraction, c(29, 32) / 360))
stopifnot(all.equal(schedule.periods(p3$start, "thirtyE360")$fraction, 29 / 360))
e360 <- schedule.periods(as.Date(c("2024-01-31", "2024-03-31")), "thirtyE360")
stopifnot(all.equal(e360$fraction, 60 / 360))
stopifnot(all.equal(schedule.periods(s, "actact")$fraction, c(31, 29) / 366))
stopifnot(inherits(try(schedule.periods(Sys.Date()), silent = TRUE), "try-error"))
