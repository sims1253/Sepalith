#' Standardise raw attendance codes
#'
#' Maps the many codes used by registers onto four statuses: present, late,
#' excused and unexcused.
#'
#' @param code Character vector of raw register codes.
#' @return A factor with levels `present`, `late`, `excused` and `unexcused`.
#' @keywords internal
attendance_status <- function(code) {
  code <- toupper(trimws(code))
  status <- dplyr::case_when(
    code %in% c("P", "/", "\\", "V") ~ "present",
    code %in% c("L", "T", "U") ~ "late",
    code %in% c("M", "I", "E", "C", "R") ~ "excused",
    code %in% c("O", "N", "A", "G") ~ "unexcused",
    TRUE ~ NA_character_
  )
  factor(status, levels = c("present", "late", "excused", "unexcused"))
}

#' Attendance summary per student
#'
#' Counts sessions by status and computes the attendance rate, treating late
#' arrivals as present.
#'
#' @param register A data frame of register marks with columns `student_id`,
#'   `date` and `code`.
#' @return A data frame with one row per student.
#' @export
#' @importFrom dplyr mutate group_by summarise
attendance_summary <- function(register) {
  register |>
    mutate(status = attendance_status(code)) |>
    group_by(student_id) |>
    summarise(
      sessions = sum(!is.na(status)),
      present = sum(status == "present", na.rm = TRUE),
      late = sum(status == "late", na.rm = TRUE),
      excused = sum(status == "excused", na.rm = TRUE),
      unexcused = sum(status == "unexcused", na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(rate = (present + late) / sessions)
}

#' Longest run of consecutive absences
#'
#' Uses run-length encoding over the student's sessions in date order.
#'
#' @param status A factor of attendance statuses in date order.
#' @return A single integer.
#' @keywords internal
longest_absence_run <- function(status) {
  absent <- status %in% c("excused", "unexcused")
  if (!any(absent)) {
    return(0L)
  }
  runs <- rle(absent)
  max(runs$lengths[runs$values])
}

#' Flag chronic absence
#'
#' Students missing at least `threshold` of their sessions, for any reason,
#' are flagged as chronically absent.
#'
#' @param register A data frame of register marks with columns `student_id`,
#'   `date` and `code`.
#' @param threshold Share of missed sessions that triggers the flag.
#' @return A data frame with columns `student_id`, `missed_share` and
#'   `chronic`.
#' @family attendance flags
#' @export
flag_chronic_absence <- function(register, threshold = 0.1) {
  attendance_summary(register) |>
    mutate(
      missed_share = (excused + unexcused) / sessions,
      chronic = missed_share >= threshold
    ) |>
    dplyr::select(student_id, missed_share, chronic)
}

#' Flag repeated lateness
#'
#' Students arriving late in at least `threshold` of their sessions are
#' flagged.
#'
#' @param register A data frame of register marks with columns `student_id`,
#'   `date` and `code`.
#' @param threshold Share of late sessions that triggers the flag.
#' @return A data frame with columns `student_id`, `late_share` and
#'   `persistent_late`.
#' @family attendance flags
#' @export
flag_persistent_lateness <- function(register, threshold = 0.1) {
  attendance_summary(register) |>
    mutate(
      late_share = late / sessions,
      persistent_late = late_share >= threshold
    ) |>
    dplyr::select(student_id, late_share, persistent_late)
}

#' Flag long runs of consecutive absence
#'
#' Students with `min_run` or more consecutive missed sessions are flagged,
#' which often catches extended absences hidden in a healthy overall rate.
#'
#' @param register A data frame of register marks with columns `student_id`,
#'   `date` and `code`.
#' @param min_run Length of absence run that triggers the flag.
#' @return A data frame with columns `student_id`, `longest_run` and
#'   `extended_absence`.
#' @family attendance flags
#' @export
flag_absence_runs <- function(register, min_run = 5L) {
  register |>
    mutate(status = attendance_status(code)) |>
    dplyr::arrange(student_id, date) |>
    group_by(student_id) |>
    summarise(longest_run = longest_absence_run(status), .groups = "drop") |>
    mutate(extended_absence = longest_run >= min_run)
}

#' Combine attendance flags into an alert list
#'
#' Joins the individual flags and keeps students with at least one, ordered
#' so that those with the most flags come first.
#'
#' @param register A data frame of register marks with columns `student_id`,
#'   `date` and `code`.
#' @param chronic_threshold Share of missed sessions for the chronic flag.
#' @param late_threshold Share of late sessions for the lateness flag.
#' @param min_run Length of absence run for the extended-absence flag.
#' @return A data frame with one row per flagged student and an `n_flags`
#'   column.
#' @export
attendance_alerts <- function(register, chronic_threshold = 0.1, late_threshold = 0.1,
                              min_run = 5L) {
  chronic <- flag_chronic_absence(register, chronic_threshold)
  lateness <- flag_persistent_lateness(register, late_threshold)
  runs <- flag_absence_runs(register, min_run)
  chronic |>
    dplyr::inner_join(lateness, by = "student_id") |>
    dplyr::inner_join(runs, by = "student_id") |>
    mutate(n_flags = chronic + persistent_late + extended_absence) |>
    dplyr::filter(n_flags > 0) |>
    dplyr::arrange(dplyr::desc(n_flags), dplyr::desc(missed_share))
}

#' Weekly attendance rate per student
#'
#' Groups the register by ISO week and reports the share of sessions attended.
#'
#' @param register A data frame of register marks with columns `student_id`,
#'   `date` and `code`.
#' @return A data frame with columns `student_id`, `period` and `rate`.
#' @family attendance trends
#' @export
weekly_attendance <- function(register) {
  register |>
    mutate(
      status = attendance_status(code),
      period = format(as.Date(date), "%G-W%V")
    ) |>
    group_by(student_id, period) |>
    summarise(rate = mean(status %in% c("present", "late")), .groups = "drop")
}

#' Monthly attendance rate per student
#'
#' Groups the register by calendar month and reports the share of sessions
#' attended.
#'
#' @param register A data frame of register marks with columns `student_id`,
#'   `date` and `code`.
#' @return A data frame with columns `student_id`, `period` and `rate`.
#' @family attendance trends
#' @export
monthly_attendance <- function(register) {
  register |>
    mutate(
      status = attendance_status(code),
      period = format(as.Date(date), "%Y-%m")
    ) |>
    group_by(student_id, period) |>
    summarise(rate = mean(status %in% c("present", "late")), .groups = "drop")
}
