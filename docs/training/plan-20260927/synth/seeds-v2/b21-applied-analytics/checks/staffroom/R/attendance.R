# skipped: needs dplyr (attendance_status, attendance_summary,
# flag_chronic_absence, flag_persistent_lateness, flag_absence_runs,
# attendance_alerts, weekly_attendance, monthly_attendance)

# longest_absence_run
st <- c("present", "excused", "unexcused", "present", "unexcused")
stopifnot(longest_absence_run(st) == 2L)
stopifnot(longest_absence_run(c("present", "late", NA)) == 0L)
stopifnot(longest_absence_run(rep("unexcused", 4)) == 4L)
