#' Check that an object is a formula
#'
#' Signals a classed error when `x` is not a formula, so that every editor in
#' this file fails with the same message.
#'
#' @param x The object to check.
#' @param arg The argument name used in the error message.
#' @return `x`, invisibly.
#' @keywords internal
check_formula <- function(x, arg = "formula") {
  if (!inherits(x, "formula")) {
    msg <- sprintf("`%s` must be a formula, not %s.", arg, class(x)[1L])
    stop(structure(
      list(message = msg, call = NULL),
      class = c("termsmith_bad_formula", "error", "condition")
    ))
  }
  invisible(x)
}

#' Add terms to a model formula
#'
#' Appends one or more terms to the right-hand side of a formula while keeping
#' the response, the intercept and the formula environment intact.
#'
#' @param formula A model formula, one- or two-sided.
#' @param terms A character vector of term labels.
#' @return A formula with the same environment as `formula`.
#' @family formula editors
#' @export
add_terms <- function(formula, terms) {
  check_formula(formula)
  if (!is.character(terms) || length(terms) == 0L) {
    stop("`terms` must be a non-empty character vector.", call. = FALSE)
  }
  tt <- stats::terms(formula)
  current <- attr(tt, "term.labels")
  new_terms <- setdiff(terms, current)
  if (length(new_terms) == 0L) {
    return(formula)
  }
  response <- if (attr(tt, "response") == 1L) formula[[2L]] else NULL
  out <- stats::reformulate(
    c(current, new_terms),
    response = response,
    intercept = attr(tt, "intercept") == 1L
  )
  environment(out) <- environment(formula)
  out
}

#' Drop terms from a model formula
#'
#' Removes one or more terms from the right-hand side of a formula while
#' keeping the response, the intercept and the formula environment intact.
#'
#' @param formula A model formula, one- or two-sided.
#' @param terms A character vector of term labels.
#' @return A formula with the same environment as `formula`.
#' @family formula editors
#' @export
drop_terms <- function(formula, terms) {
  check_formula(formula)
  if (!is.character(terms) || length(terms) == 0L) {
    stop("`terms` must be a non-empty character vector.", call. = FALSE)
  }
  tt <- stats::terms(formula)
  current <- attr(tt, "term.labels")
  keep <- setdiff(current, terms)
  if (length(keep) == length(current)) {
    return(formula)
  }
  response <- if (attr(tt, "response") == 1L) formula[[2L]] else NULL
  # reformulate() needs at least one label; fall back to an intercept-only rhs
  if (length(keep) == 0L) {
    keep <- "1"
  }
  out <- stats::reformulate(
    keep,
    response = response,
    intercept = attr(tt, "intercept") == 1L
  )
  environment(out) <- environment(formula)
  out
}

#' Replace the response of a model formula
#'
#' Builds a new two-sided formula with `response` on the left and the original
#' right-hand side unchanged.
#'
#' @param formula A model formula, one- or two-sided.
#' @param response The new response, as a string or a language object.
#' @return A formula with the same environment as `formula`.
#' @family formula editors
#' @export
swap_response <- function(formula, response) {
  check_formula(formula)
  if (is.character(response)) {
    if (length(response) != 1L) {
      stop("`response` must be a single string.", call. = FALSE)
    }
    response <- str2lang(response)
  }
  rhs <- formula[[length(formula)]]
  out <- call("~", response, rhs)
  stats::as.formula(out, env = environment(formula))
}

#' Variables on the left-hand side
#'
#' Lists the variable names used in the response of a formula.
#'
#' @param formula A model formula, one- or two-sided.
#' @return A character vector, empty for one-sided formulas.
#' @family formula inspectors
#' @export
lhs_vars <- function(formula) {
  check_formula(formula)
  if (length(formula) < 3L) {
    return(character())
  }
  vars <- all.vars(formula[[2L]])
  unique(vars)
}

#' Variables on the right-hand side
#'
#' Lists the variable names used in the predictors of a formula, ignoring the
#' `.` placeholder.
#'
#' @param formula A model formula, one- or two-sided.
#' @return A character vector, possibly empty.
#' @family formula inspectors
#' @export
rhs_vars <- function(formula) {
  check_formula(formula)
  if (length(formula) < 2L) {
    return(character())
  }
  vars <- all.vars(formula[[length(formula)]])
  unique(vars[vars != "."])
}

#' Interact terms with a moderator
#'
#' Adds the moderator as a main effect and appends its two-way interaction with
#' each of `terms`. By default every first-order term is moderated.
#'
#' @param formula A model formula, one- or two-sided.
#' @param moderator A single term label used as moderator.
#' @param terms A character vector of term labels, or `NULL` for all main
#'   effects.
#' @return A formula with the same environment as `formula`.
#' @family formula editors
#' @export
interact_with <- function(formula, moderator, terms = NULL) {
  check_formula(formula)
  if (!is.character(moderator) || length(moderator) != 1L) {
    stop("`moderator` must be a single term label.", call. = FALSE)
  }
  tt <- stats::terms(formula)
  current <- attr(tt, "term.labels")
  if (is.null(terms)) {
    terms <- current[attr(tt, "order") == 1L]
  }
  unknown <- setdiff(terms, current)
  if (length(unknown) > 0L) {
    stop(
      sprintf("Unknown term(s): %s.", paste(unknown, collapse = ", ")),
      call. = FALSE
    )
  }
  terms <- setdiff(terms, moderator)
  if (length(terms) == 0L) {
    return(add_terms(formula, moderator))
  }
  products <- paste(terms, moderator, sep = ":")
  add_terms(add_terms(formula, moderator), products)
}

#' Canonical signature of a formula
#'
#' Produces a string that is identical for formulas that differ only in term
#' order or in the order of variables inside interactions.
#'
#' @param formula A model formula, one- or two-sided.
#' @return A single string.
#' @family formula inspectors
#' @export
formula_signature <- function(formula) {
  check_formula(formula)
  tt <- stats::terms(formula)
  labels <- attr(tt, "term.labels")
  labels <- vapply(
    strsplit(labels, ":", fixed = TRUE),
    function(parts) paste(sort(parts), collapse = ":"),
    character(1)
  )
  lhs <- if (length(formula) == 3L) deparse1(formula[[2L]]) else ""
  rhs <- if (length(labels) > 0L) paste(sort(labels), collapse = " + ") else "1"
  if (attr(tt, "intercept") == 0L) {
    rhs <- paste(rhs, "- 1")
  }
  trimws(paste(lhs, "~", rhs))
}

#' Expand the dot in a formula
#'
#' Replaces `.` on the right-hand side with the columns of `data` that are not
#' already used as the response.
#'
#' @param formula A model formula, one- or two-sided.
#' @param data A data frame supplying the columns that `.` stands for.
#' @return A formula with the same environment as `formula`.
#' @family formula editors
#' @export
expand_dot <- function(formula, data) {
  check_formula(formula)
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  if (!"." %in% all.vars(formula[[length(formula)]])) {
    return(formula)
  }
  tt <- stats::terms(formula, data = data)
  response <- if (attr(tt, "response") == 1L) formula[[2L]] else NULL
  out <- stats::reformulate(
    attr(tt, "term.labels"),
    response = response,
    intercept = attr(tt, "intercept") == 1L
  )
  environment(out) <- environment(formula)
  out
}
