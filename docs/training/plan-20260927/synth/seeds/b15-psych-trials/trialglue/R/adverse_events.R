#' Number of participants per arm in the safety population
#'
#' @param adsl A subject-level data frame with one row per participant.
#' @param arm Name of the treatment arm column.
#' @return A tibble with columns `arm` and `n_arm`.
#' @keywords internal
arm_totals <- function(adsl, arm) {
  adsl %>%
    dplyr::filter(.data$saffl == "Y") %>%
    dplyr::count(arm = .data[[arm]], name = "n_arm")
}

#' Flag treatment-emergent adverse events
#'
#' An event is treatment-emergent when it starts on or after the first dose
#' and no later than `window` days after the last dose.
#'
#' @param ae An event-level adverse event data frame, one row per event.
#' @param adsl A subject-level data frame with one row per participant.
#' @param window Number of days after the last dose still counted as on
#'   treatment.
#' @return `ae` with a logical `trtemfl` column added.
#' @export
flag_treatment_emergent <- function(ae, adsl, window = 30) {
  dosing <- adsl %>%
    dplyr::select("usubjid", "trtsdt", "trtedt")
  ae %>%
    dplyr::left_join(dosing, by = "usubjid") %>%
    dplyr::mutate(
      trtemfl = !is.na(.data$trtsdt) &
        .data$aestdt >= .data$trtsdt &
        (is.na(.data$trtedt) | .data$aestdt <= .data$trtedt + window)
    ) %>%
    dplyr::select(-"trtsdt", -"trtedt")
}

#' Participants with events by system organ class
#'
#' Counts each participant at most once per system organ class and expresses
#' the count as a percentage of the safety population in each arm.
#'
#' @param ae An event-level adverse event data frame, one row per event.
#' @param adsl A subject-level data frame with one row per participant.
#' @param arm Name of the treatment arm column.
#' @return A tibble with columns `aesoc`, `arm`, `n`, `n_arm` and `pct`.
#' @family ae counts
#' @export
ae_by_soc <- function(ae, adsl, arm = "trt01a") {
  totals <- arm_totals(adsl, arm)
  ae %>%
    dplyr::filter(.data$trtemfl) %>%
    dplyr::distinct(.data$usubjid, .data$aesoc, arm = .data[[arm]]) %>%
    dplyr::count(.data$aesoc, .data$arm, name = "n") %>%
    dplyr::left_join(totals, by = "arm") %>%
    dplyr::mutate(pct = 100 * .data$n / .data$n_arm)
}

#' Participants with events by preferred term
#'
#' Counts each participant at most once per preferred term within its system
#' organ class and expresses the count as a percentage of the safety
#' population in each arm.
#'
#' @param ae An event-level adverse event data frame, one row per event.
#' @param adsl A subject-level data frame with one row per participant.
#' @param arm Name of the treatment arm column.
#' @return A tibble with columns `aesoc`, `aedecod`, `arm`, `n`, `n_arm` and
#'   `pct`.
#' @family ae counts
#' @export
ae_by_pt <- function(ae, adsl, arm = "trt01a") {
  totals <- arm_totals(adsl, arm)
  ae %>%
    dplyr::filter(.data$trtemfl) %>%
    dplyr::distinct(.data$usubjid, .data$aesoc, .data$aedecod, arm = .data[[arm]]) %>%
    dplyr::count(.data$aesoc, .data$aedecod, .data$arm, name = "n") %>%
    dplyr::left_join(totals, by = "arm") %>%
    dplyr::mutate(pct = 100 * .data$n / .data$n_arm)
}

#' Participants by worst severity
#'
#' Keeps the most severe event per participant and counts participants by
#' that worst grade in each arm.
#'
#' @param ae An event-level adverse event data frame, one row per event.
#' @param adsl A subject-level data frame with one row per participant.
#' @param arm Name of the treatment arm column.
#' @param levels Severity levels from mildest to most severe.
#' @return A tibble with columns `aesev`, `arm`, `n`, `n_arm` and `pct`.
#' @family ae counts
#' @export
ae_by_worst_severity <- function(ae, adsl, arm = "trt01a",
                                 levels = c("MILD", "MODERATE", "SEVERE")) {
  totals <- arm_totals(adsl, arm)
  ae %>%
    dplyr::filter(.data$trtemfl) %>%
    dplyr::mutate(aesev = factor(.data$aesev, levels = levels, ordered = TRUE)) %>%
    dplyr::group_by(.data$usubjid, arm = .data[[arm]]) %>%
    dplyr::summarise(aesev = max(.data$aesev), .groups = "drop") %>%
    dplyr::count(.data$aesev, .data$arm, name = "n") %>%
    dplyr::left_join(totals, by = "arm") %>%
    dplyr::mutate(pct = 100 * .data$n / .data$n_arm)
}

#' Overview of adverse event categories
#'
#' Produces the opening table of a safety section: participants with any
#' treatment-emergent event, any serious event, any related event, any
#' event leading to discontinuation and any fatal event.
#'
#' @param ae An event-level adverse event data frame, one row per event.
#' @param adsl A subject-level data frame with one row per participant.
#' @param arm Name of the treatment arm column.
#' @return A tibble with columns `category`, `arm`, `n`, `n_arm` and `pct`.
#' @export
ae_overview <- function(ae, adsl, arm = "trt01a") {
  totals <- arm_totals(adsl, arm)
  teae <- ae %>%
    dplyr::filter(.data$trtemfl)
  categories <- list(
    "Any TEAE" = rep(TRUE, nrow(teae)),
    "Serious TEAE" = teae$aeser == "Y",
    "Related TEAE" = teae$aerel %in% c("PROBABLE", "POSSIBLE", "RELATED"),
    "TEAE leading to discontinuation" = teae$aeacn == "DRUG WITHDRAWN",
    "Fatal TEAE" = teae$aeout == "FATAL"
  )
  rows <- vector("list", length(categories))
  for (i in seq_along(categories)) {
    keep <- categories[[i]] & !is.na(categories[[i]])
    rows[[i]] <- teae[keep, , drop = FALSE] %>%
      dplyr::distinct(.data$usubjid, arm = .data[[arm]]) %>%
      dplyr::count(.data$arm, name = "n") %>%
      dplyr::right_join(totals, by = "arm") %>%
      dplyr::mutate(
        category = names(categories)[i],
        n = dplyr::coalesce(.data$n, 0L)
      )
  }
  dplyr::bind_rows(rows) %>%
    dplyr::mutate(
      pct = 100 * .data$n / .data$n_arm,
      category = factor(.data$category, levels = names(categories))
    ) %>%
    dplyr::select("category", "arm", "n", "n_arm", "pct") %>%
    dplyr::arrange(.data$category, .data$arm)
}

#' Risk differences between an active arm and control
#'
#' Computes the difference in incidence proportions with a Wald confidence
#' interval for every row of a count table.
#'
#' @param counts A count table from [ae_by_soc()] or [ae_by_pt()].
#' @param active Label of the active arm.
#' @param control Label of the control arm.
#' @param conf_level Confidence level for the intervals.
#' @return A tibble with the term columns, `rd`, `rd_low` and `rd_high`
#'   (all in percentage points).
#' @export
ae_risk_difference <- function(counts, active, control, conf_level = 0.95) {
  z <- stats::qnorm(1 - (1 - conf_level) / 2)
  keys <- setdiff(names(counts), c("arm", "n", "n_arm", "pct"))
  wide <- counts %>%
    dplyr::filter(.data$arm %in% c(active, control)) %>%
    dplyr::mutate(arm = ifelse(.data$arm == active, "act", "ctl")) %>%
    dplyr::select(dplyr::all_of(keys), "arm", "n", "n_arm") %>%
    tidyr::pivot_wider(names_from = "arm", values_from = c("n", "n_arm"))
  wide %>%
    dplyr::mutate(
      dplyr::across(c("n_act", "n_ctl"), ~ dplyr::coalesce(.x, 0L)),
      p_act = .data$n_act / .data$n_arm_act,
      p_ctl = .data$n_ctl / .data$n_arm_ctl,
      se = sqrt(.data$p_act * (1 - .data$p_act) / .data$n_arm_act +
        .data$p_ctl * (1 - .data$p_ctl) / .data$n_arm_ctl),
      rd = 100 * (.data$p_act - .data$p_ctl),
      rd_low = .data$rd - 100 * z * .data$se,
      rd_high = .data$rd + 100 * z * .data$se
    ) %>%
    dplyr::select(dplyr::all_of(keys), "rd", "rd_low", "rd_high")
}
