#' @title Pivot measurements to one column per parameter
#' @description `widen.labs()` and `widen.vitals()` differ only in the prefix
#'   they give the new columns. Repeated measurements within a visit are
#'   averaged.
#' @param data A long data frame of results with `subject_id`, `visit`,
#'   `param` and `aval` columns.
#' @param names_prefix Prefix added to each new column name.
#' @return A wide tibble with one row per subject visit.
#' @export
widen.labs = function(data, names_prefix = "lb_") {
  # Parameter names become column names, so make them syntactic first.
  data |>
    dplyr::mutate(
      param = stringr::str_to_lower(stringr::str_replace_all(.data$param, "\\W+", "_"))
    ) |>
    tidyr::pivot_wider(
      id_cols = c("subject_id", "visit"),
      names_from = "param",
      values_from = "aval",
      names_prefix = names_prefix,
      values_fn = mean
    )
}

#' @rdname widen.labs
#' @export
widen.vitals = function(data, names_prefix = "vs_") {
  data |>
    dplyr::mutate(
      param = stringr::str_to_lower(stringr::str_replace_all(.data$param, "\\W+", "_"))
    ) |>
    tidyr::pivot_wider(
      id_cols = c("subject_id", "visit"),
      names_from = "param",
      values_from = "aval",
      names_prefix = names_prefix,
      values_fn = mean
    )
}

#' @title Pivot wide measurement columns back to long form
#' @param data A wide data frame with one row per subject visit.
#' @param names_prefix Prefix identifying the measurement columns.
#' @return A long tibble with `subject_id`, `visit`, `param` and `aval`.
#' @export
lengthen.labs = function(data, names_prefix = "lb_") {
  data |>
    tidyr::pivot_longer(
      cols = dplyr::starts_with(names_prefix),
      names_to = "param",
      names_prefix = names_prefix,
      values_to = "aval",
      values_drop_na = TRUE
    )
}

#' @rdname lengthen.labs
#' @export
lengthen.vitals = function(data, names_prefix = "vs_") {
  data |>
    tidyr::pivot_longer(
      cols = dplyr::starts_with(names_prefix),
      names_to = "param",
      names_prefix = names_prefix,
      values_to = "aval",
      values_drop_na = TRUE
    )
}

#' @title Split blood pressure readings into systolic and diastolic values
#' @description Readings recorded as "120/80" are split into two numeric columns.
#'   Readings that do not match the pattern give `NA` in both columns.
#' @param data A data frame of vital signs.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the
#'   combined reading.
#' @return `data` with `sysbp` and `diabp` columns replacing `col`.
#' @export
bp.to.columns = function(data, col = bp) {
  # str_match() returns a matrix: the full match, then one column per group.
  data |>
    dplyr::mutate(
      .bp = stringr::str_match({{ col }}, "^\\s*(\\d{2,3})\\s*/\\s*(\\d{2,3})\\s*$"),
      sysbp = as.numeric(.bp[, 2]),
      diabp = as.numeric(.bp[, 3])
    ) |>
    dplyr::select(-".bp", -{{ col }})
}

#' @title Nest visit records by subject
#' @description Produces one row per subject with the visits in a list-column, which
#'   is convenient for per-subject derivations with purrr.
#' @param data A data frame of visit records, one row per subject visit.
#' @param subject <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the subject.
#' @return A tibble with the subject column, `visits` and `n_visits`.
#' @export
visits.by.subject = function(data, subject = subject_id) {
  data |>
    tidyr::nest(visits = -{{ subject }}) |>
    dplyr::mutate(n_visits = purrr::map_int(.data$visits, nrow))
}

#' @title Fill missing values forward within each subject
#' @description Last observation carried forward, restricted to rows of the same
#'   subject and ordered by study day.
#' @param data A data frame of visit records, one row per subject visit.
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Columns to fill.
#' @param subject <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the subject.
#' @return `data` with missing values in `cols` filled forward.
#' @export
carry.forward = function(data, cols, subject = subject_id) {
  data |>
    dplyr::arrange({{ subject }}, .data$study_day) |>
    dplyr::group_by({{ subject }}) |>
    tidyr::fill({{ cols }}, .direction = "down") |>
    dplyr::ungroup()
}
