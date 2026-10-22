#' Parse visit dates recorded in mixed formats
#'
#' Site case report forms arrive with dates as "2026-03-14", "14/03/2026"
#' or "14MAR2026". All of these are parsed into `Date`; anything else
#' becomes `NA` with a warning listing the offending values.
#'
#' @param x A character vector of raw visit dates.
#' @param orders Date orders tried by [lubridate::parse_date_time()].
#' @return A `Date` vector the same length as `x`.
#' @export
parse.visit.date = function(x, orders = c("Ymd", "dmY", "dbY")) {
  x = stringr::str_squish(as.character(x))
  x[x %in% c("", "UNK", "ND")] = NA_character_
  parsed = suppressWarnings(lubridate::parse_date_time(x, orders = orders, exact = FALSE))
  failed = !is.na(x) & is.na(parsed)
  if (any(failed)) {
    warning("Could not parse ", sum(failed), " visit date(s): ",
      paste(utils::head(unique(x[failed]), 5), collapse = ", "), call. = FALSE)
  }
  lubridate::as_date(parsed)
}

#' Derive the study day of each visit
#'
#' Follows the usual convention that the reference date is day 1 and the
#' day before it is day -1; there is no day 0.
#'
#' @param data A data frame of visit records, one row per subject visit.
#' @param date <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the visit date.
#' @param ref <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the reference date.
#' @return `data` with an added integer `study_day` column.
#' @family visit timing
#' @export
derive.study.day = function(data, date = visit_date, ref = first_dose_date) {
  data |>
    dplyr::mutate(
      .offset = as.integer({{ date }} - {{ ref }}),
      study_day = dplyr::if_else(.offset >= 0L, .offset + 1L, .offset)
    ) |>
    dplyr::select(-".offset")
}

#' Derive the days since each subject's previous visit
#'
#' @param data A data frame of visit records, one row per subject visit.
#' @param date <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the visit date.
#' @param subject <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the subject.
#' @return `data` with an added integer `days_since_prev` column.
#' @family visit timing
#' @export
derive.days.since.prev = function(data, date = visit_date, subject = subject_id) {
  data |>
    dplyr::arrange({{ subject }}, {{ date }}) |>
    dplyr::mutate(
      days_since_prev = as.integer({{ date }} - dplyr::lag({{ date }})),
      .by = {{ subject }}
    )
}

#' Build a protocol visit schedule
#'
#' @param labels Character vector of visit labels in protocol order.
#' @param target_days Integer vector of target study days, one per label.
#' @param tolerance Allowed deviation in days, recycled to all visits.
#' @return A tibble with `visit`, `target_day`, `window_start` and
#'   `window_end` columns.
#' @export
visit.schedule = function(labels, target_days, tolerance = 3L) {
  if (length(labels) != length(target_days)) {
    stop("`labels` and `target_days` must have the same length.", call. = FALSE)
  }
  if (is.unsorted(target_days, strictly = TRUE)) {
    stop("`target_days` must be strictly increasing.", call. = FALSE)
  }
  tolerance = rep_len(as.integer(tolerance), length(labels))
  tibble::tibble(
    visit = factor(labels, levels = labels),
    target_day = as.integer(target_days),
    window_start = .window.start(target_days, tolerance),
    window_end = as.integer(target_days) + tolerance
  )
}

#' Assign each record to the nearest scheduled visit
#'
#' Records are matched to the scheduled visit whose target day is closest
#' to their study day. Ties go to the earlier visit.
#'
#' @param data A data frame of visit records with a `study_day` column.
#' @param schedule A visit schedule as returned by [visit.schedule()].
#' @return `data` with `visit`, `target_day` and `deviation` columns.
#' @export
assign.visit.window = function(data, schedule) {
  if (!"study_day" %in% names(data)) {
    stop("`data` needs a `study_day` column; see derive.study.day().", call. = FALSE)
  }
  nearest = vapply(data$study_day, function(day) {
    if (is.na(day)) {
      return(NA_integer_)
    }
    which.min(abs(schedule$target_day - day))
  }, integer(1))
  data |>
    dplyr::mutate(
      visit = schedule$visit[nearest],
      target_day = schedule$target_day[nearest],
      deviation = .data$study_day - .data$target_day
    )
}

#' Flag visits that happened before their window opened
#'
#' @param data A data frame of visits with `study_day` and `visit` columns.
#' @param schedule A visit schedule as returned by [visit.schedule()].
#' @return `data` with an added logical `early` column.
#' @family window flags
#' @export
flag.early.visits = function(data, schedule) {
  bounds = dplyr::select(schedule, "visit", "window_start")
  data |>
    dplyr::left_join(bounds, by = "visit") |>
    dplyr::mutate(early = !is.na(.data$study_day) & .data$study_day < .data$window_start) |>
    dplyr::select(-"window_start")
}

#' Flag visits that happened after their window closed
#'
#' @param data A data frame of visits with `study_day` and `visit` columns.
#' @param schedule A visit schedule as returned by [visit.schedule()].
#' @return `data` with an added logical `late` column.
#' @family window flags
#' @export
flag.late.visits = function(data, schedule) {
  bounds = dplyr::select(schedule, "visit", "window_end")
  data |>
    dplyr::left_join(bounds, by = "visit") |>
    dplyr::mutate(late = !is.na(.data$study_day) & .data$study_day > .data$window_end) |>
    dplyr::select(-"window_end")
}

#' Compute window starts without crossing study day 1
#'
#' @param target_days Integer vector of target study days.
#' @param tolerance Integer vector of allowed deviations.
#' @return An integer vector of window start days.
#' @noRd
.window.start = function(target_days, tolerance) {
  start = as.integer(target_days) - tolerance
  crosses = target_days > 0 & start <= 0
  start[crosses] = 1L
  start
}
