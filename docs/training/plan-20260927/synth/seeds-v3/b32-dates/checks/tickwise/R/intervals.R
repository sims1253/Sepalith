iv <- interval.new(c("2024-03-01 08:00", "2024-03-01 12:00"), "2024-03-01 18:30")
stopifnot(inherits(iv, "tw_interval"), nrow(iv) == 2L, identical(names(iv), c("start", "end")))
stopifnot(attr(iv$start, "tzone") == "UTC")
stopifnot(inherits(try(interval.new(NA, "2024-03-01"), silent = TRUE), "try-error"))
stopifnot(inherits(try(interval.new("2024-03-01", NA), silent = TRUE), "try-error"))
stopifnot(inherits(try(interval.new("2024-03-02", "2024-03-01"), silent = TRUE), "try-error"))

stopifnot(identical(as.numeric(interval.length(iv)), c(10.5, 6.5)))
stopifnot(as.numeric(interval.length(iv, "mins"))[2] == 390)
stopifnot(units(interval.length(iv, "days")) == "days")

ex <- interval.extend(iv, before = 3600, after = as.difftime(30, units = "mins"))
stopifnot(identical(as.numeric(interval.length(ex)), c(12, 8)))
stopifnot(inherits(try(interval.extend(iv, after = -10 * 3600), silent = TRUE), "try-error"))

a <- interval.new(c("2024-03-01 08:00", "2024-03-01 08:00"), c("2024-03-01 10:00",
  "2024-03-01 09:00"))
b <- interval.new(c("2024-03-01 09:00", "2024-03-01 09:30"), c("2024-03-01 11:00",
  "2024-03-01 10:00"))
ov <- interval.overlap(a, b)
stopifnot(identical(ov$overlap, c(3600, 0)), is.na(ov$start[2]), is.na(ov$end[2]))
stopifnot(ov$start[1] == as.POSIXct("2024-03-01 09:00", tz = "UTC"))
stopifnot(ov$end[1] == as.POSIXct("2024-03-01 10:00", tz = "UTC"))
stopifnot(inherits(try(interval.overlap(a, b[1, ]), silent = TRUE), "try-error"))

one <- interval.new("2024-03-01 22:30", "2024-03-03 01:15")
sh <- interval.split(one, "hour")
stopifnot(nrow(sh) == 28L, sum(as.numeric(interval.length(sh, "mins"))) == 26 * 60 + 45)
sd <- interval.split(one, "day")
stopifnot(nrow(sd) == 3L, identical(as.numeric(interval.length(sd)), c(1.5, 24, 1.25)))
sw <- interval.split(interval.new("2024-03-01", "2024-03-20"), "week")
stopifnot(nrow(sw) == 4L, sw$start[2] == as.POSIXct("2024-03-04", tz = "UTC"))
sm <- interval.split(interval.new("2024-01-15", "2024-04-01"), "month")
stopifnot(nrow(sm) == 3L, sm$end[3] == as.POSIXct("2024-04-01", tz = "UTC"))
stopifnot(inherits(try(interval.split(iv, "day"), silent = TRUE), "try-error"))

many <- interval.new(c("2024-03-01 08:00", "2024-03-01 09:00", "2024-03-01 12:00",
  "2024-03-01 13:05"), c("2024-03-01 10:00", "2024-03-01 11:00", "2024-03-01 13:00",
  "2024-03-01 14:00"))
g <- interval.gaps(many)
stopifnot(nrow(g) == 2L, as.numeric(interval.length(g, "mins"))[1] == 60)
stopifnot(nrow(interval.gaps(many, min.gap = 600)) == 1L)
stopifnot(nrow(interval.gaps(many[1, ])) == 0L)
