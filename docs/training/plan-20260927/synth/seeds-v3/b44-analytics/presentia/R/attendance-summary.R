#' Attendance counts per student
#'
#' One block per status `present`, `late`, `absent`, `excused` and `left`; a
#' session counts as attended when the status is `present`, `late` or `left`.
#'
#' @param records A data frame with columns `student`, `date`, `session` and
#'   `status`, one row per student and session.
#' @param digits Number of decimal places kept in rates.
#' @return A data frame with columns `student` and `sessions`, a count and a
#'   `.rate` column per status, and `attended.rate`.
#' @export
student.summary <- function(records, digits = 3) {
  students <- sort(unique(records$student))
  s <- factor(records$student, levels = students)
  sessions <- as.vector(table(s))
  out <- data.frame(student = students, sessions = sessions)

  # Status `present`
  n <- as.vector(tapply(records$status == "present", s, sum))
  out$present <- n
  out$present.rate <- round(n / sessions, digits)
  # Status `late`
  n <- as.vector(tapply(records$status == "late", s, sum))
  out$late <- n
  out$late.rate <- round(n / sessions, digits)
  # Status `absent`
  n <- as.vector(tapply(records$status == "absent", s, sum))
  out$absent <- n
  out$absent.rate <- round(n / sessions, digits)
  # Status `excused`
  n <- as.vector(tapply(records$status == "excused", s, sum))
  out$excused <- n
  out$excused.rate <- round(n / sessions, digits)
  # Status `left`
  n <- as.vector(tapply(records$status == "left", s, sum))
  out$left <- n
  out$left.rate <- round(n / sessions, digits)

  out$attended.rate <- round((out$present + out$late + out$left) / sessions, digits)
  out
}

#' Attendance rate by group
#'
#' @inheritParams student.summary
#' @param by One of `"student"`, `"week"`, `"month"`, `"weekday"` or `"session"`.
#' @return A data frame with columns `group`, `sessions`, `attended` and `rate`.
#' @export
attendance.rate <- function(records, by = c("student", "week", "month", "weekday", "session"),
                            digits = 3) {
  by <- match.arg(by)
  attended <- records$status %in% c("present", "late", "left")
  switch(by,
    student = {
      # One group per student
      key <- records$student
      groups <- sort(unique(key))
    },
    week = {
      # ISO weeks such as 2026-W36
      key <- format(records$date, "%G-W%V")
      groups <- sort(unique(key))
    },
    month = {
      # Calendar months such as 2026-09
      key <- format(records$date, "%Y-%m")
      groups <- sort(unique(key))
    },
    weekday = {
      # Monday first, whatever the locale
      day.names <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
      key <- day.names[as.integer(format(records$date, "%u"))]
      groups <- day.names[day.names %in% key]
    },
    session = {
      # One group per session label
      key <- records$session
      groups <- sort(unique(key))
    }
  )

  f <- factor(key, levels = groups)
  sessions <- as.vector(table(f))
  hits <- as.vector(tapply(attended, f, sum, default = 0L))
  data.frame(group = groups, sessions = sessions, attended = hits,
             rate = round(hits / sessions, digits))
}

#' Compare attendance before and after a date
#'
#' @inheritParams student.summary
#' @param split.date A `Date`; sessions on or after it count as after.
#' @return A data frame with columns `student`, `sessions.before`,
#'   `rate.before`, `absent.before`, the same three `.after` columns,
#'   `rate.change` and `trend`.
#' @export
compare.periods <- function(records, split.date) {
  students <- sort(unique(records$student))
  attended <- records$status %in% c("present", "late", "left")
  out <- data.frame(student = students)

  # Before `split.date`
  rows <- records$date < split.date
  s <- factor(records$student[rows], levels = students)
  out$sessions.before <- as.vector(table(s))
  out$rate.before <- as.vector(tapply(attended[rows], s, mean))
  out$absent.before <- as.vector(tapply(records$status[rows] == "absent", s, sum, default = 0L))
  # From `split.date` on
  rows <- records$date >= split.date
  s <- factor(records$student[rows], levels = students)
  out$sessions.after <- as.vector(table(s))
  out$rate.after <- as.vector(tapply(attended[rows], s, mean))
  out$absent.after <- as.vector(tapply(records$status[rows] == "absent", s, sum, default = 0L))

  out$rate.change <- round(out$rate.after - out$rate.before, 3)
  out$trend <- ifelse(out$rate.change > 0, "up", ifelse(out$rate.change < 0, "down", "flat"))
  out
}

#' Band attendance rates
#'
#' @param rate Attendance rates between 0 and 1.
#' @param scheme One of `"national"`, `"strict"`, `"lenient"` or `"traffic"`.
#' @return A factor of bands, lowest band first.
#' @export
attendance.band <- function(rate, scheme = c("national", "strict", "lenient", "traffic")) {
  scheme <- match.arg(scheme)
  if (any(rate < 0 | rate > 1, na.rm = TRUE)) {
    stop("`rate` must lie between 0 and 1.", call. = FALSE)
  }
  switch(scheme,
    national = {
      cuts <- c(0, 0.5, 0.9, 0.95)
      labels <- c("severe", "persistent", "concern", "expected")
    },
    strict = {
      cuts <- c(0, 0.8, 0.95, 0.98)
      labels <- c("severe", "persistent", "concern", "expected")
    },
    lenient = {
      cuts <- c(0, 0.4, 0.8, 0.9)
      labels <- c("severe", "persistent", "concern", "expected")
    },
    traffic = {
      cuts <- c(0, 0.9, 0.95)
      labels <- c("red", "amber", "green")
    }
  )
  band <- labels[findInterval(rate, cuts)]
  factor(band, levels = labels)
}

#' Attendance streaks per student
#'
#' @inheritParams student.summary
#' @return A data frame with columns `student`, `best.attended` (longest run
#'   attended), `worst.missed` (longest run missed) and `current`.
#' @export
streaks <- function(records) {
  records <- records[order(records$student, records$date, records$session), ]
  students <- sort(unique(records$student))
  out <- data.frame(student = students, best.attended = 0L, worst.missed = 0L, current = "")
  for (i in seq_along(students)) {
    status <- records$status[records$student == students[i]]
    runs <- rle(status %in% c("present", "late", "left"))
    # Longest run attended
    out$best.attended[i] <- max(c(0L, runs$lengths[runs$values]))
    # Longest run missed
    out$worst.missed[i] <- max(c(0L, runs$lengths[!runs$values]))
    # Run still going at the last session
    last <- length(runs$lengths)
    state <- if (runs$values[last]) "attended" else "missed"
    out$current[i] <- sprintf("%s x%d", state, runs$lengths[last])
  }
  out
}
