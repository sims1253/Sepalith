near = function(a, b) isTRUE(all.equal(a, b))
h = c(0, 5, 30, 80)

stopifnot(identical(latePenalty(80, h)$final, rep(80, 4)))
stopifnot(identical(latePenalty(80, h, "flat")$final, c(80, 72, 72, 72)))
stopifnot(identical(latePenalty(80, h, "perDay")$final, c(80, 72, 64, 48)))
stopifnot(identical(latePenalty(80, h, "perHour")$final, c(80, 78.33, 70, 53.33)))
stopifnot(identical(latePenalty(80, h, "tiered")$final, c(80, 72, 56, 24)))
ct = latePenalty(80, h, "cutoff")
stopifnot(identical(ct$final, c(80, 72, 64, 0)), identical(ct$deduction, c(0, 10, 20, 100)))
stopifnot(identical(latePenalty(50, c(NA, -3))$hoursLate, c(0, 0)))
stopifnot(identical(latePenalty(50, 24 * 30, "perDay")$final, 0))

due = as.POSIXct("2026-03-10 12:00:00", tz = "UTC")
req = data.frame(
  student = c("s1", "s1", "s1", "s2"),
  requested = due - c(48, 10, 48, 30) * 3600,
  due = rep(due, 4), days = c(2, 5, 20, 4), documented = c(FALSE, TRUE, TRUE, FALSE)
)
rv = reviewExtensions(req)
stopifnot(identical(rv$lateRequest, c(FALSE, TRUE, FALSE, FALSE)))
stopifnot(identical(rv$tooLong, c(FALSE, FALSE, TRUE, FALSE)))
stopifnot(identical(rv$tooMany, c(FALSE, FALSE, TRUE, FALSE)))
stopifnot(identical(rv$undocumented, c(FALSE, FALSE, FALSE, TRUE)))
stopifnot(identical(rv$approved, c(TRUE, FALSE, FALSE, FALSE)))
stopifnot(identical(rv$reasons, c("", "lateRequest", "tooLong;tooMany", "undocumented")))
stopifnot(near(as.numeric(difftime(rv$newDue, req$due, units = "days")), c(2, 0, 0, 0)))
stopifnot(all(reviewExtensions(req, maxDays = 30, maxPerStudent = 3, noticeHours = 1,
                               docDays = 5)$approved))

sub = due + c(-60, 600, 3600, NA, 7200)
cl = classifySubmissions(rep(due, 5), sub, excused = c(FALSE, FALSE, FALSE, FALSE, TRUE))
stopifnot(identical(cl$status, c("onTime", "grace", "late", "missing", "excused")))
stopifnot(identical(cl$minutesLate, c(0, 10, 60, NA, 120)))
stopifnot(identical(unname(cl$counts), rep(1L, 5)))
stopifnot(identical(classifySubmissions(due, due + 600, graceMinutes = 5)$status, "late"))
stopifnot(identical(classifySubmissions(due, NA, excused = TRUE)$counts[["excused"]], 1L))

ls = lateSummary(c("b", "a", "a", "b", "a"), cl$status)
stopifnot(identical(ls$student, c("a", "b")), identical(ls$onTime, c(0L, 1L)))
stopifnot(identical(ls$grace, c(1L, 0L)), identical(ls$late, c(1L, 0L)))
stopifnot(identical(ls$missing, c(0L, 1L)), identical(ls$excused, c(1L, 0L)))
stopifnot(identical(ls$total, c(3L, 2L)), identical(ls$lateShare, c(0.5, 0.5)))

pv = penaltyPreview(hours = c(5, 80), score = 80)
stopifnot(identical(dim(pv), c(2L, 6L)), identical(rownames(pv), c("5h", "80h")))
stopifnot(identical(pv[, "tiered"], c(`5h` = 72, `80h` = 24)), pv["80h", "cutoff"] == 0)
