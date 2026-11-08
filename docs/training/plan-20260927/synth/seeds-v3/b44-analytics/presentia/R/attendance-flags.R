#' Check an attendance register
#'
#' Validates the columns `student`, `date`, `session` and `status` in turn.
#'
#' @param records A data frame with columns `student`, `date` (class `Date`),
#'   `session` and `status`, one row per student and session.
#' @param statuses Allowed values of the `status` column.
#' @return `records`, invisibly, when the register is valid.
#' @keywords internal
check.attendance <- function(records,
                             statuses = c("present", "absent", "late", "excused", "left")) {
  needed <- c("student", "date", "session", "status")
  absent.cols <- setdiff(needed, names(records))
  if (length(absent.cols) > 0L) {
    stop("`records` lacks column(s): ", toString(absent.cols), call. = FALSE)
  }

  # Column `student`
  bad <- is.na(records$student) | !nzchar(records$student)
  if (any(bad)) {
    stop("Column `student` is invalid in ", sum(bad), " row(s).", call. = FALSE)
  }
  # Column `date`
  bad <- !inherits(records$date, "Date") | is.na(records$date)
  if (any(bad)) {
    stop("Column `date` is invalid in ", sum(bad), " row(s).", call. = FALSE)
  }
  # Column `session`
  bad <- is.na(records$session) | !nzchar(records$session)
  if (any(bad)) {
    stop("Column `session` is invalid in ", sum(bad), " row(s).", call. = FALSE)
  }
  # Column `status`
  bad <- !records$status %in% statuses
  if (any(bad)) {
    stop("Column `status` is invalid in ", sum(bad), " row(s).", call. = FALSE)
  }

  key <- paste(records$student, records$date, records$session)
  if (anyDuplicated(key) > 0L) {
    stop("Each student may appear once per date and session.", call. = FALSE)
  }
  invisible(records)
}

#' Normalise attendance codes
#'
#' Maps the codes of one register scheme onto the statuses `present`,
#' `absent`, `late`, `excused` and `left`.
#'
#' @param code A vector of raw codes.
#' @param scheme One of `"letter"`, `"numeric"`, `"word"` or `"register"`.
#' @return A character vector of statuses, `NA` for codes not recognised.
#' @export
normalize.codes <- function(code, scheme = c("letter", "numeric", "word", "register")) {
  scheme <- match.arg(scheme)
  statuses <- c("present", "absent", "late", "excused", "left")
  code <- trimws(as.character(code))
  switch(scheme,
    letter = {
      # P, A, L, E and D for departed
      from <- c("P", "A", "L", "E", "D")
      key <- toupper(code)
    },
    numeric = {
      # 1 for present, 0 for absent, then 2 to 4
      from <- c("1", "0", "2", "3", "4")
      key <- code
    },
    word = {
      # Full words, any case
      from <- c("present", "absent", "late", "excused", "left early")
      key <- tolower(code)
    },
    register = {
      # Paper register marks
      from <- c("/", "O", "L", "M", "E")
      key <- toupper(code)
    }
  )

  out <- statuses[match(key, from)]
  unknown <- sum(is.na(out) & !is.na(code))
  if (unknown > 0L) {
    warning(unknown, " code(s) not recognised for scheme \"", scheme, "\".", call. = FALSE)
  }
  out
}

#' Default flag thresholds per school phase
#'
#' @param phase One of `"primary"`, `"secondary"`, `"college"` or `"adult"`.
#' @return A named numeric vector with elements `absence.rate`, `consecutive`,
#'   `late.rate`, `unexcused` and `early.leave`.
#' @export
rule.thresholds <- function(phase = c("primary", "secondary", "college", "adult")) {
  phase <- match.arg(phase)
  switch(phase,
    primary = {
      absence.rate <- 0.1
      consecutive <- 3
      late.rate <- 0.15
      unexcused <- 4
      early.leave <- 3
    },
    secondary = {
      absence.rate <- 0.1
      consecutive <- 3
      late.rate <- 0.2
      unexcused <- 5
      early.leave <- 4
    },
    college = {
      absence.rate <- 0.15
      consecutive <- 4
      late.rate <- 0.25
      unexcused <- 6
      early.leave <- 5
    },
    adult = {
      absence.rate <- 0.2
      consecutive <- 5
      late.rate <- 0.3
      unexcused <- 8
      early.leave <- 6
    }
  )
  c(
    absence.rate = absence.rate, consecutive = consecutive, late.rate = late.rate,
    unexcused = unexcused, early.leave = early.leave
  )
}

#' Flag students against attendance rules
#'
#' One block per rule: `absence.rate` and `late.rate` are shares of sessions,
#' `consecutive` is the longest run of absences, `unexcused` and `early.leave`
#' are counts. A rule fires when its value reaches the threshold.
#'
#' @inheritParams check.attendance
#' @param thresholds Named thresholds, as returned by [rule.thresholds()].
#' @return A data frame with columns `student`, `sessions`, one logical column
#'   and one `.value` column per rule, and `n.flags`.
#' @export
attendance.flags <- function(records, thresholds = rule.thresholds("secondary")) {
  records <- records[order(records$student, records$date, records$session), ]
  students <- sort(unique(records$student))
  s <- factor(records$student, levels = students)
  longest.run <- function(x) {
    runs <- rle(x)
    max(c(0L, runs$lengths[runs$values]))
  }
  out <- data.frame(student = students, sessions = as.vector(table(s)))

  # Rule `absence.rate`
  hit <- records$status %in% c("absent", "excused")
  value <- as.vector(tapply(hit, s, mean))
  out$absence.rate <- value >= thresholds[["absence.rate"]]
  out$absence.rate.value <- round(value, 3)
  # Rule `consecutive`
  hit <- records$status == "absent"
  value <- as.vector(tapply(hit, s, longest.run))
  out$consecutive <- value >= thresholds[["consecutive"]]
  out$consecutive.value <- value
  # Rule `late.rate`
  hit <- records$status == "late"
  value <- as.vector(tapply(hit, s, mean))
  out$late.rate <- value >= thresholds[["late.rate"]]
  out$late.rate.value <- round(value, 3)
  # Rule `unexcused`
  hit <- records$status == "absent"
  value <- as.vector(tapply(hit, s, sum))
  out$unexcused <- value >= thresholds[["unexcused"]]
  out$unexcused.value <- value
  # Rule `early.leave`
  hit <- records$status == "left"
  value <- as.vector(tapply(hit, s, sum))
  out$early.leave <- value >= thresholds[["early.leave"]]
  out$early.leave.value <- value

  rules <- c("absence.rate", "consecutive", "late.rate", "unexcused", "early.leave")
  out$n.flags <- rowSums(out[rules])
  out
}

#' Summarise attendance flags per rule
#'
#' @param flags A data frame as returned by [attendance.flags()].
#' @return A data frame with columns `rule`, `students`, `share` and `names`,
#'   one row per rule.
#' @export
flag.summary <- function(flags) {
  rules <- c("absence.rate", "consecutive", "late.rate", "unexcused", "early.leave")
  out <- data.frame(rule = rules, students = 0L, share = 0, names = "")

  # Rule `absence.rate`
  hit <- flags$absence.rate
  out$students[1] <- sum(hit)
  out$share[1] <- round(mean(hit), 3)
  out$names[1] <- paste(flags$student[hit], collapse = ", ")
  # Rule `consecutive`
  hit <- flags$consecutive
  out$students[2] <- sum(hit)
  out$share[2] <- round(mean(hit), 3)
  out$names[2] <- paste(flags$student[hit], collapse = ", ")
  # Rule `late.rate`
  hit <- flags$late.rate
  out$students[3] <- sum(hit)
  out$share[3] <- round(mean(hit), 3)
  out$names[3] <- paste(flags$student[hit], collapse = ", ")
  # Rule `unexcused`
  hit <- flags$unexcused
  out$students[4] <- sum(hit)
  out$share[4] <- round(mean(hit), 3)
  out$names[4] <- paste(flags$student[hit], collapse = ", ")
  # Rule `early.leave`
  hit <- flags$early.leave
  out$students[5] <- sum(hit)
  out$share[5] <- round(mean(hit), 3)
  out$names[5] <- paste(flags$student[hit], collapse = ", ")

  out[order(-out$students, out$rule), ]
}
