#' Turn a column name into a one-sided formula
#'
#' Helper used to build the formula arguments expected by the survey package.
#' `NULL` stays `NULL` and `"1"` gives the intercept-only formula.
#'
#' @param vars Character vector of column names, or `NULL`.
#' @return A one-sided formula, or `NULL`.
#' @noRd
as_rhs_formula = function(vars) {
  if (is.null(vars)) {
    return(NULL)
  }
  stats::as.formula(paste("~", paste(vars, collapse = " + ")))
}

#' Build a survey design from raked weights
#'
#' Wraps [survey::svydesign()] with column-name arguments instead of formulas
#' and stores the weights in a column called `.weight`.
#'
#' @param data A data frame of survey respondents, one row per respondent.
#' @param weights A numeric vector of respondent weights.
#' @param ids Character vector of cluster id columns, or `NULL` for none.
#' @param strata Character vector of stratum columns, or `NULL`.
#' @param fpc Name of the finite population correction column, or `NULL`.
#' @return A `survey.design2` object.
#' @export
design_from_weights = function(data, weights, ids = NULL, strata = NULL, fpc = NULL) {
  if (length(weights) != nrow(data)) {
    stop("`weights` must have one value per row of `data`.", call. = FALSE)
  }
  if (anyNA(weights) || any(weights <= 0)) {
    stop("All weights must be non-missing and strictly positive.", call. = FALSE)
  }

  # svydesign() wants formulas; keep the weights in a column of their own so
  # they travel with the data through subsetting.
  data$.weight = weights
  id_formula = if (is.null(ids)) ~1 else as_rhs_formula(ids)

  survey::svydesign(
    ids = id_formula,
    strata = as_rhs_formula(strata),
    fpc = as_rhs_formula(fpc),
    weights = ~.weight,
    data = data,
    nest = !is.null(strata)
  )
}

#' Build a replicate-weight design from raked weights
#'
#' Converts the linearisation design from [design_from_weights()] into a
#' replicate design so that variance estimates reflect the raking step.
#'
#' @inheritParams design_from_weights
#' @param type Replication method passed to [survey::as.svrepdesign()].
#' @param replicates Number of bootstrap replicates when `type = "bootstrap"`.
#' @return A `svyrep.design` object.
#' @export
replicate_design = function(data, weights, ids = NULL, strata = NULL, fpc = NULL,
                            type = c("JK1", "JKn", "bootstrap"), replicates = 200L) {
  type = match.arg(type)
  base = design_from_weights(data, weights, ids = ids, strata = strata, fpc = fpc)

  switch(type,
    JK1 = survey::as.svrepdesign(base, type = "JK1"),
    JKn = {
      if (is.null(strata)) {
        stop("`type = \"JKn\"` requires `strata`.", call. = FALSE)
      }
      survey::as.svrepdesign(base, type = "JKn")
    },
    bootstrap = survey::as.svrepdesign(base, type = "bootstrap", replicates = replicates)
  )
}

#' Tidy a survey statistic into a data frame
#'
#' Pulls estimates and standard errors out of an `svystat` or `svyby` object
#' and adds a normal-approximation confidence interval.
#'
#' @param stat An object returned by [survey::svymean()], [survey::svytotal()]
#'   or [survey::svyby()].
#' @param level Confidence level of the interval.
#' @return A data frame with columns `term`, `estimate`, `std_error`,
#'   `conf_low` and `conf_high`.
#' @keywords internal
tidy_svystat = function(stat, level = 0.95) {
  estimate = stats::coef(stat)
  se = survey::SE(stat)
  z = stats::qnorm(1 - (1 - level) / 2)

  out = data.frame(
    term = names(estimate),
    estimate = unname(estimate),
    std_error = unname(as.numeric(se)),
    stringsAsFactors = FALSE
  )

  # Normal approximation; svyciprop() intervals are not reused here so all
  # three wrappers report the same kind of interval.
  out$conf_low = out$estimate - z * out$std_error
  out$conf_high = out$estimate + z * out$std_error
  rownames(out) = NULL
  out
}

#' Weighted means, totals and proportions by group
#'
#' Thin wrappers around [survey::svyby()] with [survey::svymean()],
#' [survey::svytotal()] or [survey::svyciprop()] that take column names and
#' return a tidy data frame.
#'
#' @param design A survey design object.
#' @param variable Name of the outcome column.
#' @param by Character vector of grouping columns, or `NULL` for no grouping.
#' @param level Confidence level of the interval.
#' @return A data frame as returned by `tidy_svystat()`.
#' @export
weighted_mean_by = function(design, variable, by = NULL, level = 0.95) {
  outcome = as_rhs_formula(variable)
  if (is.null(by)) {
    stat = survey::svymean(outcome, design, na.rm = TRUE)
  } else {
    stat = survey::svyby(outcome, as_rhs_formula(by), design, survey::svymean,
      na.rm = TRUE, keep.var = TRUE
    )
  }

  tidy_svystat(stat, level = level)
}

#' @rdname weighted_mean_by
#' @export
weighted_total_by = function(design, variable, by = NULL, level = 0.95) {
  outcome = as_rhs_formula(variable)
  if (is.null(by)) {
    stat = survey::svytotal(outcome, design, na.rm = TRUE)
  } else {
    stat = survey::svyby(outcome, as_rhs_formula(by), design, survey::svytotal,
      na.rm = TRUE, keep.var = TRUE
    )
  }

  tidy_svystat(stat, level = level)
}

#' @rdname weighted_mean_by
#' @details `weighted_prop_by()` expects a logical or 0/1 outcome column and
#'   uses [survey::svyciprop()] with the logit method.
#' @export
weighted_prop_by = function(design, variable, by = NULL, level = 0.95) {
  # svyciprop() needs a 0/1 numeric, so coerce logicals inside the formula.
  outcome = stats::as.formula(paste0("~I(as.numeric(", variable, "))"))
  if (is.null(by)) {
    stat = survey::svyciprop(outcome, design, method = "logit", level = level)
  } else {
    stat = survey::svyby(outcome, as_rhs_formula(by), design, survey::svyciprop,
      method = "logit", level = level, vartype = "se"
    )
  }

  tidy_svystat(stat, level = level)
}
