#' Estimated marginal means by arm at each visit
#'
#' Wraps [emmeans::emmeans()] for the usual treatment-by-visit layout of a
#' repeated-measures trial.
#'
#' @param fit A fitted model supported by emmeans, e.g. from [fit_trial_lmm()].
#' @param treatment Name of the treatment arm column.
#' @param time Name of the visit or time column.
#' @param type Scale of the estimates, `"link"` or `"response"`.
#' @return An `emmGrid` object.
#' @importFrom emmeans emmeans
#' @export
arm_means <- function(fit, treatment, time, type = c("link", "response")) {
  type <- match.arg(type)
  specs <- stats::as.formula(paste("~", treatment, "|", time))
  emmeans(fit, specs = specs, type = type)
}

#' Treatment-versus-control contrasts at each visit
#'
#' Compares every active arm with the reference arm, within each visit.
#'
#' @inheritParams arm_means
#' @param ref Index of the reference (control) arm among the factor levels.
#' @param adjust Multiplicity adjustment passed to emmeans.
#' @return An `emmGrid` object of contrasts.
#' @importFrom emmeans contrast
#' @export
arm_vs_control <- function(fit, treatment, time, ref = 1, adjust = "dunnett") {
  emm <- arm_means(fit, treatment, time)
  contrast(emm, method = "trt.vs.ctrl", ref = ref, adjust = adjust)
}

#' All pairwise arm contrasts at each visit
#'
#' @inheritParams arm_means
#' @param adjust Multiplicity adjustment passed to emmeans.
#' @return An `emmGrid` object of contrasts.
#' @export
arm_pairwise <- function(fit, treatment, time, adjust = "tukey") {
  emm <- arm_means(fit, treatment, time)
  contrast(emm, method = "pairwise", adjust = adjust)
}

#' Contrast of change from first to last visit between arms
#'
#' Builds a difference-in-differences contrast: the change from the first to
#' the last visit in each active arm minus the same change in control.
#'
#' @param fit A fitted model supported by emmeans, e.g. from [fit_trial_lmm()].
#' @param treatment Name of the treatment arm column.
#' @param time Name of the visit or time column.
#' @param control Label of the control arm.
#' @return An `emmGrid` object with one contrast per active arm.
#' @export
change_contrast <- function(fit, treatment, time, control) {
  emm <- emmeans(fit, specs = c(treatment, time))
  grid <- emm@grid
  arms <- unique(as.character(grid[[treatment]]))
  visits <- unique(as.character(grid[[time]]))
  if (!control %in% arms) {
    stop("Control arm '", control, "' not found.", call. = FALSE)
  }

  # Visits follow the factor level order, so first/last are baseline/end
  first <- visits[1]
  last <- visits[length(visits)]

  # One weight vector per active arm over the arm-by-visit grid:
  # (arm_last - arm_first) - (control_last - control_first)
  coefs <- list()
  for (arm in setdiff(arms, control)) {
    w <- numeric(nrow(grid))
    w[grid[[treatment]] == arm & grid[[time]] == last] <- 1
    w[grid[[treatment]] == arm & grid[[time]] == first] <- -1
    w[grid[[treatment]] == control & grid[[time]] == last] <- -1
    w[grid[[treatment]] == control & grid[[time]] == first] <- 1
    coefs[[paste(arm, "-", control)]] <- w
  }
  contrast(emm, method = coefs)
}

#' Tidy a set of emmeans contrasts
#'
#' Converts an `emmGrid` of contrasts to a data frame with confidence limits
#' and p-values, in the column names used across the reporting code.
#'
#' @param con An `emmGrid` object of contrasts.
#' @param conf_level Confidence level for the intervals.
#' @return A tibble with one row per contrast.
#' @importFrom emmeans infer
#' @export
tidy_contrasts <- function(con, conf_level = 0.95) {
  summ <- summary(infer(con, infer = c(TRUE, TRUE), level = conf_level))

  # Column names depend on the model scale (t vs z, difference vs ratio)
  est_col <- intersect(c("estimate", "ratio", "odds.ratio"), names(summ))[1]
  lcl_col <- intersect(c("lower.CL", "asymp.LCL"), names(summ))[1]
  ucl_col <- intersect(c("upper.CL", "asymp.UCL"), names(summ))[1]
  summ <- as.data.frame(summ)

  tibble::tibble(
    contrast = as.character(summ$contrast),
    estimate = summ[[est_col]],
    conf_low = summ[[lcl_col]],
    conf_high = summ[[ucl_col]],
    p_value = summ$p.value
  ) %>%
    dplyr::bind_cols(summ[setdiff(names(summ), c(
      "contrast", est_col, lcl_col, ucl_col, "SE", "df", "t.ratio", "z.ratio", "p.value"
    ))])
}

#' Format a contrast table for a clinical study report
#'
#' Combines estimate and confidence limits into a single string and formats
#' p-values, optionally flagging those below `alpha`.
#'
#' @param tidy A tibble from [tidy_contrasts()].
#' @param digits Number of decimal places for estimates.
#' @param alpha Significance level used for flagging.
#' @return A tibble of character columns ready for printing.
#' @export
format_contrasts <- function(tidy, digits = 2, alpha = 0.05) {
  fmt <- function(x) formatC(x, format = "f", digits = digits)

  tidy %>%
    dplyr::mutate(
      `Estimate (95% CI)` = sprintf(
        "%s (%s, %s)", fmt(.data$estimate), fmt(.data$conf_low), fmt(.data$conf_high)
      ),
      `P-value` = format.pval(.data$p_value, digits = 3, eps = 0.001),
      `P-value` = ifelse(.data$p_value < alpha, paste0(.data$`P-value`, "*"), .data$`P-value`)
    ) %>%
    dplyr::select(-dplyr::any_of(c("estimate", "conf_low", "conf_high", "p_value")))
}
