#' Build a mixed-model formula for a repeated-measures trial
#'
#' Assembles the fixed part (treatment, visit, their interaction and any
#' covariates) and a random intercept, optionally with a random slope.
#'
#' @param response Name of the outcome column.
#' @param treatment Name of the treatment arm column.
#' @param time Name of the visit or time column.
#' @param subject Name of the participant identifier column.
#' @param covariates Optional character vector of baseline covariate names.
#' @param random_slope If `TRUE`, add a random slope for `time` by subject.
#' @return A formula suitable for [lme4::lmer()] or [lme4::glmer()].
#' @export
#' @examples
#' trial_formula("hamd", "arm", "week", "pid", covariates = "hamd_bl")
trial_formula <- function(response, treatment, time, subject,
                          covariates = character(), random_slope = FALSE) {
  fixed <- c(paste(treatment, time, sep = " * "), covariates)
  random <- if (random_slope) {
    sprintf("(1 + %s | %s)", time, subject)
  } else {
    sprintf("(1 | %s)", subject)
  }

  rhs <- paste(c(fixed, random), collapse = " + ")
  stats::as.formula(paste(response, "~", rhs), env = parent.frame())
}

#' Fit a linear mixed model for a continuous endpoint
#'
#' Thin wrapper around [lme4::lmer()] that drops rows with a missing outcome
#' and records the formula used.
#'
#' @param data A data frame in long format, one row per participant and visit.
#' @param formula A model formula, e.g. from [trial_formula()].
#' @param reml If `TRUE`, fit by restricted maximum likelihood.
#' @param ... Passed to [lme4::lmer()].
#' @return A fitted `merMod` object.
#' @seealso [fit_trial_glmm()] for binary and count endpoints.
#' @export
fit_trial_lmm <- function(data, formula, reml = TRUE, ...) {
  outcome <- all.vars(formula)[1]
  data <- data %>% dplyr::filter(!is.na(.data[[outcome]]))

  fit <- lme4::lmer(formula, data = data, REML = reml, ...)
  attr(fit, "trial_formula") <- formula
  fit
}

#' Fit a generalised linear mixed model for a binary or count endpoint
#'
#' Thin wrapper around [lme4::glmer()] that drops rows with a missing outcome
#' and records the formula used.
#'
#' @inheritParams fit_trial_lmm
#' @param family A GLM family, e.g. [stats::binomial()].
#' @param ... Passed to [lme4::glmer()].
#' @return A fitted `merMod` object.
#' @export
fit_trial_glmm <- function(data, formula, family = stats::binomial(), ...) {
  outcome <- all.vars(formula)[1]
  data <- data %>% dplyr::filter(!is.na(.data[[outcome]]))

  fit <- lme4::glmer(formula, data = data, family = family, ...)
  attr(fit, "trial_formula") <- formula
  fit
}

#' Tidy variance components of a mixed model
#'
#' Extracts the random-effect variances, standard deviations and
#' correlations from [lme4::VarCorr()] as a data frame.
#'
#' @param fit A fitted `merMod` object.
#' @return A data frame with columns `group`, `term`, `variance`, `sd` and
#'   `share` (proportion of total variance).
#' @export
variance_components <- function(fit) {
  vc <- as.data.frame(lme4::VarCorr(fit))

  # Rows with a second variable are covariances; keep only the variances
  out <- vc %>%
    dplyr::filter(is.na(.data$var2)) %>%
    dplyr::transmute(
      group = .data$grp,
      term = dplyr::coalesce(.data$var1, "(residual)"),
      variance = .data$vcov,
      sd = .data$sdcor
    )
  out %>% dplyr::mutate(share = .data$variance / sum(.data$variance))
}

#' Intraclass correlation from a random-intercept model
#'
#' Ratio of the subject-level intercept variance to the total variance. For
#' binomial models the residual variance is fixed at `pi^2 / 3`.
#'
#' @param fit A fitted `merMod` object.
#' @param group Name of the grouping factor holding the random intercept.
#' @return A single numeric value between 0 and 1.
#' @export
icc_from_fit <- function(fit, group) {
  vc <- variance_components(fit)
  between <- vc %>%
    dplyr::filter(.data$group == !!group, .data$term == "(Intercept)") %>%
    dplyr::pull(.data$variance)
  if (length(between) != 1) {
    stop("No random intercept found for `", group, "`.", call. = FALSE)
  }

  # Latent-scale residual variance of the logistic distribution for GLMMs
  within <- if (lme4::isGLMM(fit)) pi^2 / 3 else lme4::sigma(fit)^2
  between / (between + within)
}

#' Fixed-effect estimates with Wald confidence intervals
#'
#' @param fit A fitted `merMod` object.
#' @param conf_level Confidence level for the intervals.
#' @param exponentiate If `TRUE`, report estimates and limits on the
#'   exponentiated scale (odds or rate ratios).
#' @return A data frame with one row per fixed effect.
#' @export
fixed_effects_table <- function(fit, conf_level = 0.95, exponentiate = FALSE) {
  est <- lme4::fixef(fit)
  se <- sqrt(diag(as.matrix(stats::vcov(fit))))
  z <- stats::qnorm(1 - (1 - conf_level) / 2)

  out <- tibble::tibble(
    term = names(est),
    estimate = unname(est),
    std_error = unname(se),
    conf_low = estimate - z * std_error,
    conf_high = estimate + z * std_error
  )
  # Limits are built on the link scale first, then transformed, so the
  # interval stays asymmetric around the ratio as it should
  if (exponentiate) {
    out <- out %>%
      dplyr::mutate(dplyr::across(c("estimate", "conf_low", "conf_high"), exp))
  }
  out
}

#' Conditional modes of the random effects
#'
#' @param fit A fitted `merMod` object.
#' @param group Name of the grouping factor whose random effects to return.
#' @return A data frame with one row per level of `group` and one column
#'   per random-effect term.
#' @export
random_effects_table <- function(fit, group) {
  re <- lme4::ranef(fit)
  if (!group %in% names(re)) {
    stop("`", group, "` is not a grouping factor of the model.", call. = FALSE)
  }
  re[[group]] %>%
    tibble::rownames_to_column(var = group) %>%
    tibble::as_tibble()
}

#' Summarise convergence problems of a mixed model
#'
#' Collects the singular-fit flag and any optimiser warnings so that a
#' batch of fits can be screened before reporting.
#'
#' @param fit A fitted `merMod` object.
#' @return A one-row data frame with columns `singular`, `n_messages` and
#'   `messages`.
#' @export
fit_diagnostics <- function(fit) {
  msgs <- fit@optinfo$conv$lme4$messages
  tibble::tibble(
    singular = lme4::isSingular(fit),
    n_messages = length(msgs),
    messages = if (length(msgs) == 0) NA_character_ else paste(msgs, collapse = "; ")
  )
}
