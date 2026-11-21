fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
days <- as.Date("2026-09-01") + 0:9
reg <- data.frame(
  student = rep(c("a", "b"), each = 10), date = rep(days, 2), session = "am",
  status = c(
    "present", "present", "present", "absent", "absent", "absent", "late", "present",
    "present", "present",
    "present", "late", "late", "late", "left", "left", "excused", "present", "present",
    "present"
  )
)

ss <- student.summary(reg)
stopifnot(identical(ss$student, c("a", "b")), identical(ss$sessions, c(10L, 10L)))
stopifnot(identical(ss$present, c(6L, 4L)), identical(ss$present.rate, c(0.6, 0.4)))
stopifnot(identical(ss$late, c(1L, 3L)), identical(ss$late.rate, c(0.1, 0.3)))
stopifnot(identical(ss$absent, c(3L, 0L)), identical(ss$absent.rate, c(0.3, 0)))
stopifnot(identical(ss$excused, c(0L, 1L)), identical(ss$excused.rate, c(0, 0.1)))
stopifnot(identical(ss$left, c(0L, 2L)), identical(ss$left.rate, c(0, 0.2)))
stopifnot(identical(ss$attended.rate, c(0.7, 0.9)))

r <- attendance.rate(reg)
stopifnot(identical(r$group, c("a", "b")))
stopifnot(identical(r$attended, c(7L, 9L)))
stopifnot(identical(r$rate, c(0.7, 0.9)))
wk <- attendance.rate(reg, "week")
stopifnot(sum(wk$sessions) == 20L, !is.unsorted(wk$group), all(grepl("^2026-W3[67]$", wk$group)))
mo <- attendance.rate(reg, "month")
stopifnot(identical(mo$group, "2026-09"), mo$sessions == 20L, mo$rate == 0.8)
wd <- attendance.rate(reg, "weekday")
stopifnot(identical(wd$group, c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))
stopifnot(sum(wd$attended) == 16L)
se <- attendance.rate(reg, "session")
stopifnot(identical(se$group, "am"), se$attended == 16L)

cp <- compare.periods(reg, days[6])
stopifnot(identical(cp$sessions.before, c(5L, 5L)), identical(cp$sessions.after, c(5L, 5L)))
stopifnot(identical(cp$rate.before, c(0.6, 1)), identical(cp$absent.before, c(2L, 0L)))
stopifnot(identical(cp$rate.after, c(0.8, 0.8)), identical(cp$absent.after, c(1L, 0L)))
stopifnot(identical(cp$rate.change, c(0.2, -0.2)), identical(cp$trend, c("up", "down")))
joiner <- data.frame(student = "c", date = days[9], session = "am", status = "present")
late.joiner <- rbind(reg, joiner)
cp2 <- compare.periods(late.joiner, days[6])
stopifnot(is.na(cp2$rate.before[3]), cp2$absent.before[3] == 0L, is.na(cp2$trend[3]))

nat <- attendance.band(c(0.3, 0.6, 0.92, 1))
stopifnot(identical(as.character(nat), c("severe", "persistent", "concern", "expected")))
stopifnot(identical(levels(nat), c("severe", "persistent", "concern", "expected")))
stopifnot(identical(as.character(attendance.band(0.85, "strict")), "persistent"))
stopifnot(identical(as.character(attendance.band(0.85, "lenient")), "concern"))
tl <- attendance.band(c(0.5, 0.92, 0.99, NA), "traffic")
stopifnot(identical(as.character(tl), c("red", "amber", "green", NA)))
stopifnot(fails(attendance.band(1.2)))

st <- streaks(reg[20:1, ])
stopifnot(identical(st$best.attended, c(4L, 6L)), identical(st$worst.missed, c(3L, 1L)))
stopifnot(identical(st$current, c("attended x4", "attended x3")))
stopifnot(identical(streaks(reg[1:4, ])$current, "missed x1"))
