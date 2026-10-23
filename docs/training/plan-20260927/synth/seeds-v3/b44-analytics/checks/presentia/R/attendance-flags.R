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

stopifnot(identical(check.attendance(reg), reg), fails(check.attendance(reg[-4])))
bad <- reg; bad$student[1] <- ""
stopifnot(fails(check.attendance(bad)))
bad <- reg; bad$date <- as.character(bad$date)
stopifnot(fails(check.attendance(bad)))
bad <- reg; bad$session[3] <- NA
stopifnot(fails(check.attendance(bad)))
bad <- reg; bad$status[3] <- "sick"
stopifnot(fails(check.attendance(bad)))
stopifnot(fails(check.attendance(rbind(reg, reg[1, ]))))

stopifnot(identical(normalize.codes(c("p", " A", "l", "E", "d")),
                    c("present", "absent", "late", "excused", "left")))
stopifnot(identical(normalize.codes(c(1, 0, 4), "numeric"), c("present", "absent", "left")))
stopifnot(identical(normalize.codes(c("Late", "LEFT EARLY"), "word"), c("late", "left")))
rg <- normalize.codes(c("/", "o", "M"), "register")
stopifnot(identical(rg, c("present", "absent", "excused")))
w <- tryCatch(normalize.codes(c("P", "X")), warning = function(w) conditionMessage(w))
stopifnot(identical(w, "1 code(s) not recognised for scheme \"letter\"."))
stopifnot(identical(suppressWarnings(normalize.codes(c("P", "X", NA))), c("present", NA, NA)))

stopifnot(identical(rule.thresholds("primary")[["late.rate"]], 0.15))
stopifnot(identical(names(rule.thresholds()), c("absence.rate", "consecutive", "late.rate",
                                                "unexcused", "early.leave")))
stopifnot(identical(rule.thresholds("secondary")[["unexcused"]], 5))
stopifnot(identical(rule.thresholds("college")[["consecutive"]], 4))
stopifnot(identical(rule.thresholds("adult")[["early.leave"]], 6))

fl <- attendance.flags(reg[20:1, ])
stopifnot(identical(fl$student, c("a", "b")), identical(fl$sessions, c(10L, 10L)))
stopifnot(identical(fl$absence.rate, c(TRUE, TRUE)), identical(fl$absence.rate.value, c(0.3, 0.1)))
stopifnot(identical(fl$consecutive, c(TRUE, FALSE)), identical(fl$consecutive.value, c(3L, 0L)))
stopifnot(identical(fl$late.rate, c(FALSE, TRUE)), identical(fl$late.rate.value, c(0.1, 0.3)))
stopifnot(identical(fl$unexcused, c(FALSE, FALSE)), identical(fl$unexcused.value, c(3L, 0L)))
stopifnot(identical(fl$early.leave, c(FALSE, FALSE)), identical(fl$early.leave.value, c(0L, 2L)))
stopifnot(identical(fl$n.flags, c(2, 2)))
th <- c(absence.rate = 0.5, consecutive = 4, late.rate = 0.5, unexcused = 3, early.leave = 2)
fl2 <- attendance.flags(reg, th)
stopifnot(identical(fl2$unexcused, c(TRUE, FALSE)), identical(fl2$early.leave, c(FALSE, TRUE)))
stopifnot(identical(fl2$n.flags, c(1, 1)))

fs <- flag.summary(fl)
stopifnot(identical(fs$rule, c("absence.rate", "consecutive", "late.rate", "early.leave",
                               "unexcused")))
stopifnot(identical(fs$students, c(2L, 1L, 1L, 0L, 0L)), identical(fs$share, c(1, 0.5, 0.5, 0, 0)))
stopifnot(identical(fs$names, c("a, b", "a", "b", "", "")))
stopifnot(identical(flag.summary(fl2)$names[1:2], c("b", "a")))
