#' Derive the baseline value for each subject and parameter
#'
#' The baseline is the last non-missing value on or before study day 1.
#' Subjects without such a value get `NA` as their baseline.
#'
#' @param data A long data frame of measurements with `subject_id`,
#'   `param` and `study_day` columns.
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return `data` with added `base` and `is_baseline` columns.
#' @export
derive.baseline = function(data, value = aval) {
  data %>%
    dplyr::arrange(.data$subject_id, .data$param, .data$study_day) %>%
    dplyr::group_by(.data$subject_id, .data$param) %>%
    dplyr::mutate(
      .eligible = .data$study_day <= 1L & !is.na({{ value }}),
      .base_row = if (any(.eligible)) max(which(.eligible)) else NA_integer_,
      is_baseline = dplyr::row_number() == .base_row & !is.na(.base_row),
      base = if (any(.eligible)) {{ value }}[max(which(.eligible))] else NA_real_
    ) %>%
    dplyr::ungroup() %>%
    dplyr::select(-".eligible", -".base_row")
}

#' Derive the absolute change from baseline
#'
#' @param data A long data frame of measurements with a `base` column, as
#'   returned by [derive.baseline()].
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return `data` with an added `chg` column; `NA` on and before baseline.
#' @family change derivations
#' @export
derive.change = function(data, value = aval) {
  data %>%
    dplyr::mutate(
      chg = dplyr::if_else(.data$study_day > 1L, {{ value }} - .data$base, NA_real_)
    )
}

#' Derive the percent change from baseline
#'
#' @param data A long data frame of measurements with a `base` column, as
#'   returned by [derive.baseline()].
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return `data` with an added `pchg` column; `NA` on and before baseline.
#' @family change derivations
#' @export
derive.pct.change = function(data, value = aval) {
  data %>%
    dplyr::mutate(
      pchg = dplyr::if_else(
        .data$study_day > 1L & .data$base != 0,
        100 * ({{ value }} - .data$base) / .data$base,
        NA_real_
      )
    )
}

#' Derive the fold change from baseline
#'
#' @param data A long data frame of measurements with a `base` column, as
#'   returned by [derive.baseline()].
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return `data` with an added `fold` column; `NA` on and before baseline.
#' @family change derivations
#' @export
derive.fold.change = function(data, value = aval) {
  data %>%
    dplyr::mutate(
      fold = dplyr::if_else(
        .data$study_day > 1L & .data$base > 0,
        {{ value }} / .data$base,
        NA_real_
      )
    )
}

#' Categorise change from baseline into response classes
#'
#' @param data A long data frame of measurements with a `pchg` column, as
#'   returned by [derive.pct.change()].
#' @param improve Percent change at or below which a value counts as improved.
#' @param worsen Percent change at or above which a value counts as worsened.
#' @return `data` with an added factor `response` column.
#' @export
categorise.change = function(data, improve = -20, worsen = 20) {
  stopifnot(improve < worsen)
  data %>%
    dplyr::mutate(
      response = dplyr::case_when(
        is.na(.data$pchg) ~ NA_character_,
        .data$pchg <= improve ~ "improved",
        .data$pchg >= worsen ~ "worsened",
        .default = "stable"
      ),
      response = factor(.data$response, levels = c("improved", "stable", "worsened"))
    )
}

#' Find the worst post-baseline value per subject and parameter
#'
#' @param data A long data frame of measurements with `subject_id`,
#'   `param` and `study_day` columns.
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return A tibble with one row per subject and parameter and a `max_post` column.
#' @family post-baseline extremes
#' @export
post.baseline.max = function(data, value = aval) {
  data %>%
    dplyr::filter(.data$study_day > 1L, !is.na({{ value }})) %>%
    dplyr::group_by(.data$subject_id, .data$param) %>%
    dplyr::slice_max({{ value }}, n = 1L, with_ties = FALSE) %>%
    dplyr::ungroup() %>%
    dplyr::select("subject_id", "param", "study_day", max_post = {{ value }})
}

#' Find the best post-baseline value per subject and parameter
#'
#' @param data A long data frame of measurements with `subject_id`,
#'   `param` and `study_day` columns.
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @return A tibble with one row per subject and parameter and a `min_post` column.
#' @family post-baseline extremes
#' @export
post.baseline.min = function(data, value = aval) {
  data %>%
    dplyr::filter(.data$study_day > 1L, !is.na({{ value }})) %>%
    dplyr::group_by(.data$subject_id, .data$param) %>%
    dplyr::slice_min({{ value }}, n = 1L, with_ties = FALSE) %>%
    dplyr::ungroup() %>%
    dplyr::select("subject_id", "param", "study_day", min_post = {{ value }})
}

#' Derive analysis flags for an efficacy dataset
#'
#' Combines the baseline and change derivations, marks the record used for
#' analysis at each visit (the latest non-missing value within the visit)
#' and flags subjects without a baseline.
#'
#' @param data A long data frame of measurements with `subject_id`,
#'   `param`, `visit` and `study_day` columns.
#' @param value <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the measurement.
#' @param improve Percent change at or below which a value counts as improved.
#' @param worsen Percent change at or above which a value counts as worsened.
#' @return `data` with baseline, change, response and flag columns.
#' @export
derive.analysis.flags = function(data, value = aval, improve = -20, worsen = 20) {
  required = c("subject_id", "param", "visit", "study_day")
  absent = setdiff(required, names(data))
  if (length(absent) > 0) {
    stop("Missing columns: ", paste(absent, collapse = ", "), call. = FALSE)
  }
  derived = data %>%
    derive.baseline({{ value }}) %>%
    derive.change({{ value }}) %>%
    derive.pct.change({{ value }}) %>%
    categorise.change(improve = improve, worsen = worsen)
  # one analysis record per subject, parameter and visit
  derived = derived %>%
    dplyr::group_by(.data$subject_id, .data$param, .data$visit) %>%
    dplyr::mutate(
      .rank = dplyr::if_else(
        is.na({{ value }}),
        NA_integer_,
        dplyr::min_rank(dplyr::desc(.data$study_day))
      ),
      anl_flag = !is.na(.rank) & .rank == 1L
    ) %>%
    dplyr::ungroup() %>%
    dplyr::select(-".rank")
  no.base = derived %>%
    dplyr::group_by(.data$subject_id, .data$param) %>%
    dplyr::summarise(no_baseline = all(is.na(.data$base)), .groups = "drop")
  n.missing = sum(no.base$no_baseline)
  if (n.missing > 0) {
    message(n.missing, " subject/parameter combination(s) have no baseline value.")
  }
  derived %>%
    dplyr::left_join(no.base, by = c("subject_id", "param")) %>%
    dplyr::relocate("base", "chg", "pchg", "response", "anl_flag", .after = {{ value }})
}
