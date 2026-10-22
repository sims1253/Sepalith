#' Random effect terms
#'
#' Build lme4-style terms for a grouping factor. `randomInterceptTerm()`
#' gives a random intercept per level; `randomSlopeTerm()` adds random
#' slopes of one or more predictors, with or without correlated effects.
#'
#' @export
#' @param group Name of the grouping factor.
#' @return A single string such as `"(1 | site)"` or
#'   `"(1 + dose | site)"`.
randomInterceptTerm <- function(group) {
  if (!is.character(group) || length(group) != 1L || !nzchar(group)) {
    stop("`group` must be a single non-empty string.", call. = FALSE)
  }
  return(sprintf("(1 | %s)", group))
}

#' @rdname randomInterceptTerm
#' @export
#' @param vars Character vector of predictor names with varying slopes.
#' @param correlated Whether slopes and intercept share a correlation matrix.
randomSlopeTerm <- function(vars, group, correlated = TRUE) {
  if (!is.character(group) || length(group) != 1L || !nzchar(group)) {
    stop("`group` must be a single non-empty string.", call. = FALSE)
  }
  if (!is.character(vars) || length(vars) == 0L) {
    stop("`vars` must be a non-empty character vector.", call. = FALSE)
  }

  # The double bar drops the correlations between the varying effects.
  bar <- if (correlated) "|" else "||"
  return(sprintf("(1 + %s %s %s)", paste(vars, collapse = " + "), bar, group))
}

#' Validate a grouping specification
#'
#' Checks a named list mapping grouping factors to the predictors whose slopes
#' vary by that factor. An empty character vector means intercept only.
#'
#' @param groups A named list of character vectors, or `NULL`.
#' @noRd
validateGroupSpec <- function(groups) {
  if (is.null(groups)) {
    return(invisible(list()))
  }
  if (!is.list(groups) || is.null(names(groups)) || any(names(groups) == "")) {
    stop("`groups` must be a fully named list.", call. = FALSE)
  }

  for (g in names(groups)) {
    slopes <- groups[[g]]
    if (!is.character(slopes)) {
      stop(sprintf("Slopes for group `%s` must be character.", g), call. = FALSE)
    }
    if (anyDuplicated(slopes) > 0L) {
      stop(sprintf("Duplicated slopes for group `%s`.", g), call. = FALSE)
    }
  }
  return(invisible(groups))
}

#' Start a model formula specification
#'
#' Collects the response, fixed effects and grouping structure into a
#' backend-neutral specification that can later be turned into a brms or
#' rstanarm formula.
#'
#' @export
#' @param response Name of the response variable, or a response expression.
#' @param fixed Character vector of fixed-effect terms.
#' @param groups A named list of character vectors, or `NULL`.
#' @param intercept Whether to keep the population-level intercept.
#' @return An object of class `pcFormulaSpec`.
formulaSpec <- function(response, fixed = character(), groups = NULL,
                        intercept = TRUE) {
  if (!is.character(response) || length(response) != 1L) {
    stop("`response` must be a single string.", call. = FALSE)
  }
  if (!is.character(fixed)) {
    stop("`fixed` must be a character vector.", call. = FALSE)
  }
  if (response %in% fixed) {
    stop("The response cannot also be a fixed effect.", call. = FALSE)
  }

  groups <- validateGroupSpec(groups)
  random <- character(length(groups))
  for (i in seq_along(groups)) {
    slopes <- groups[[i]]
    # a varying slope without its population-level slope is rarely intended
    missingFixed <- setdiff(slopes, fixed)
    if (length(missingFixed) > 0L) {
      warning(
        sprintf(
          "Varying slopes without fixed effects: %s.",
          paste(missingFixed, collapse = ", ")
        ),
        call. = FALSE
      )
    }

    random[i] <- if (length(slopes) == 0L) {
      randomInterceptTerm(names(groups)[i])
    } else {
      randomSlopeTerm(slopes, names(groups)[i])
    }
  }

  spec <- list(
    response = response,
    fixed = unique(fixed),
    random = random,
    intercept = intercept,
    dpars = list()
  )
  class(spec) <- "pcFormulaSpec"
  return(spec)
}

#' Add a distributional parameter formula
#'
#' Attaches a predictor formula for an auxiliary parameter such as `sigma` or
#' `phi`. Only the brms backend supports these.
#'
#' @export
#' @param spec An object of class `pcFormulaSpec`.
#' @param dpar Name of the distributional parameter.
#' @param terms Character vector of terms predicting `dpar`.
#' @return The updated `spec`.
addDistParam <- function(spec, dpar, terms) {
  stopifnot(inherits(spec, "pcFormulaSpec"))
  if (dpar %in% c("mu", spec$response)) {
    stop("`dpar` must name an auxiliary parameter, not the mean.", call. = FALSE)
  }

  # No terms means a constant (intercept-only) auxiliary parameter.
  if (length(terms) == 0L) {
    terms <- "1"
  }
  spec$dpars[[dpar]] <- unique(terms)
  return(spec)
}

#' Right-hand side of a specification
#'
#' Pastes fixed and random terms together, adding `0 +` when the intercept is
#' suppressed. A specification without terms gives `"1"`, or `"0"` for an
#' empty model without intercept.
#'
#' @param spec An object of class `pcFormulaSpec`.
#' @return A single string.
#' @keywords internal
specRhs <- function(spec) {
  rhsTerms <- c(spec$fixed, spec$random)
  if (length(rhsTerms) == 0L) {
    return(if (spec$intercept) "1" else "0")
  }

  rhs <- paste(rhsTerms, collapse = " + ")
  if (!spec$intercept) {
    rhs <- paste("0 +", rhs)
  }
  return(rhs)
}

#' Convert a specification to a brms formula
#'
#' Builds the main formula and one formula per distributional parameter and
#' combines them with [brms::bf()].
#'
#' @export
#' @param spec An object of class `pcFormulaSpec`.
#' @param env Environment for the created formulas.
#' @return A `brmsformula` object.
asBrmsFormula <- function(spec, env = parent.frame()) {
  stopifnot(inherits(spec, "pcFormulaSpec"))
  main <- stats::as.formula(paste(spec$response, "~", specRhs(spec)), env = env)

  auxiliary <- vector("list", length(spec$dpars))
  for (i in seq_along(spec$dpars)) {
    dpar <- names(spec$dpars)[i]
    rhs <- paste(spec$dpars[[i]], collapse = " + ")
    auxiliary[[i]] <- stats::as.formula(paste(dpar, "~", rhs), env = env)
  }
  return(do.call(brms::bf, c(list(main), auxiliary)))
}

#' Convert a specification to an rstanarm formula
#'
#' rstanarm takes plain lme4-style formulas, so distributional parameters are
#' rejected rather than silently dropped.
#'
#' @export
#' @inheritParams asBrmsFormula
#' @return A formula.
asStanarmFormula <- function(spec, env = parent.frame()) {
  stopifnot(inherits(spec, "pcFormulaSpec"))
  if (length(spec$dpars) > 0L) {
    stop(
      sprintf(
        "rstanarm does not support distributional formulas (found: %s).",
        paste(names(spec$dpars), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  return(stats::as.formula(paste(spec$response, "~", specRhs(spec)), env = env))
}

#' Print a formula specification
#'
#' Shows the main formula and any distributional parameter formulas, one per
#' line.
#'
#' @export
#' @param x An object of class `pcFormulaSpec`.
#' @param ... Unused.
#' @return `x`, invisibly.
print.pcFormulaSpec <- function(x, ...) {
  cat("<formula spec>\n")
  cat("  ", x$response, " ~ ", specRhs(x), "\n", sep = "")
  for (dpar in names(x$dpars)) {
    cat("  ", dpar, " ~ ", paste(x$dpars[[dpar]], collapse = " + "), "\n", sep = "")
  }
  return(invisible(x))
}
