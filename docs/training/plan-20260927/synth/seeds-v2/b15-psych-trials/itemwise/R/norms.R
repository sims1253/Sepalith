#' Convert raw scores to standard norm scales
#'
#' `to_z_score()` standardises raw scores against a reference mean and
#' standard deviation, `to_t_score()` rescales them to a mean of 50 and a
#' standard deviation of 10, and `to_stanine()` maps them onto the nine-point
#' stanine scale, clamped to 1-9.
#'
#' @param x A numeric vector of raw scale scores.
#' @param ref_mean Mean of the reference (norm) sample.
#' @param ref_sd Standard deviation of the reference (norm) sample.
#' @return A numeric vector of z scores, T scores or (integer) stanines.
#' @export
to_z_score <- function(x, ref_mean, ref_sd) {
  stopifnot(ref_sd > 0)
  (x - ref_mean) / ref_sd
}

#' @rdname to_z_score
#' @export
to_t_score <- function(x, ref_mean, ref_sd) {
  z <- to_z_score(x, ref_mean, ref_sd)
  50 + 10 * z
}

#' @rdname to_z_score
#' @export
to_stanine <- function(x, ref_mean, ref_sd) {
  z <- to_z_score(x, ref_mean, ref_sd)
  stanine <- round(5 + 2 * z)
  as.integer(pmin(pmax(stanine, 1), 9))
}

#' Percentile ranks against a reference sample
#'
#' Uses the mid-point definition: the percentage of reference scores below a
#' value plus half the percentage equal to it.
#'
#' @param x A numeric vector of raw scale scores.
#' @param reference A numeric vector of raw scores from the norm sample.
#' @return A numeric vector of percentile ranks between 0 and 100.
#' @export
percentile_rank <- function(x, reference) {
  reference <- reference[!is.na(reference)]
  if (length(reference) == 0) {
    stop("`reference` has no non-missing values.", call. = FALSE)
  }

  n <- length(reference)
  vapply(x, function(value) {
    if (is.na(value)) {
      return(NA_real_)
    }
    below <- sum(reference < value)
    equal <- sum(reference == value)
    100 * (below + 0.5 * equal) / n
  }, numeric(1))
}

#' Build a raw-to-norm conversion table
#'
#' Tabulates every observed raw score with its T score and percentile rank,
#' optionally within groups such as age bands or sex.
#'
#' @param data A data frame with one row per respondent of the norm sample.
#' @param score Name of the column holding raw scale scores.
#' @param group Optional character vector of grouping column names.
#' @return A data frame with columns for the groups, `raw`, `n`, `t_score`
#'   and `percentile`.
#' @export
build_norm_table <- function(data, score, group = NULL) {
  # Norm statistics are computed within each group before collapsing to one
  # row per distinct raw score.
  data |>
    dplyr::filter(!is.na(.data[[score]])) |>
    dplyr::group_by(dplyr::across(dplyr::all_of(group))) |>
    dplyr::mutate(
      ref_mean = mean(.data[[score]]),
      ref_sd = stats::sd(.data[[score]]),
      percentile = percentile_rank(.data[[score]], .data[[score]])
    ) |>
    dplyr::group_by(raw = .data[[score]], .add = TRUE) |>
    dplyr::summarise(
      n = dplyr::n(),
      t_score = round(to_t_score(
        dplyr::first(raw), dplyr::first(ref_mean), dplyr::first(ref_sd)
      )),
      percentile = dplyr::first(percentile),
      .groups = "drop"
    ) |>
    dplyr::arrange(dplyr::across(dplyr::all_of(c(group, "raw"))))
}

#' Attach norm scores to new respondents
#'
#' Joins a conversion table built by [build_norm_table()] onto new data by
#' raw score and any grouping columns.
#'
#' @param data A data frame with one row per respondent to be normed.
#' @param norms A conversion table from [build_norm_table()].
#' @param score Name of the column holding raw scale scores.
#' @param group Optional character vector of grouping column names.
#' @return `data` with `t_score` and `percentile` columns added.
#' @export
apply_norms <- function(data, norms, score, group = NULL) {
  by <- c(group, stats::setNames("raw", score))
  out <- data |>
    dplyr::left_join(dplyr::select(norms, -"n"), by = by)

  # Raw scores never seen in the norm sample come back without a T score
  unmatched <- sum(is.na(out$t_score) & !is.na(out[[score]]))
  if (unmatched > 0) {
    warning(unmatched, " score(s) not found in the norm table.", call. = FALSE)
  }
  out
}

#' Standard error of measurement
#'
#' Classical-test-theory SEM from the score SD and a reliability estimate.
#'
#' @param ref_sd Standard deviation of the reference (norm) sample.
#' @param rel Reliability of the scale, e.g. from [cronbach_alpha()].
#' @return The standard error of measurement.
#' @export
sem_classical <- function(ref_sd, rel) {
  stopifnot(rel >= 0, rel <= 1)
  ref_sd * sqrt(1 - rel)
}

#' Reliable change index
#'
#' Jacobson-Truax style index for pre/post change, classifying each person
#' as improved, deteriorated or unchanged.
#'
#' @param pre,post Numeric vectors of raw scores before and after treatment.
#' @param ref_sd Standard deviation of the reference (norm) sample.
#' @param rel Reliability of the scale, e.g. from [cronbach_alpha()].
#' @param level Confidence level used for the cut-off.
#' @param higher_is_better If `TRUE`, increases count as improvement.
#' @return A data frame with columns `rci` and `status`.
#' @export
reliable_change <- function(pre, post, ref_sd, rel, level = 0.95,
                            higher_is_better = FALSE) {
  stopifnot(length(pre) == length(post))

  # Standard error of a difference between two fallible scores
  s_diff <- sqrt(2) * sem_classical(ref_sd, rel)
  rci <- (post - pre) / s_diff
  crit <- stats::qnorm(1 - (1 - level) / 2)

  signed <- if (higher_is_better) rci else -rci
  status <- ifelse(signed >= crit, "improved",
    ifelse(signed <= -crit, "deteriorated", "unchanged")
  )
  data.frame(rci = rci, status = status)
}
