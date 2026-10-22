#' Univariate smooth specification
#'
#' Describes an `s()` term for [mgcv::gam()] without building it, so that
#' basis sizes can be checked against the data first.
#'
#' @param var Name of the covariate to smooth.
#' @param bs Basis type.
#' @param k Basis dimension; `-1` lets mgcv choose.
#' @param by Optional name of a factor or numeric `by` variable.
#' @return An object of class `pcSmooth`.
#' @family smooth specs
#' @export
smoothTerm <- function(var, bs = c("tp", "cr", "cc", "ps", "re"), k = -1, by = NULL) {
  bs <- match.arg(bs)
  if (!is.character(var) || length(var) != 1L) {
    stop("`var` must be a single covariate name.", call. = FALSE)
  }
  if (bs == "re" && k != -1) {
    warning("`k` is ignored for random-effect smooths.", call. = FALSE)
    k <- -1
  }
  spec <- list(type = "s", vars = var, bs = bs, k = k, by = by)
  class(spec) <- "pcSmooth"
  return(spec)
}

#' Tensor product smooth specification
#'
#' Describes a `te()` term, or a `ti()` term when only the interaction is
#' wanted, for two or more covariates.
#'
#' @param vars Character vector of covariate names.
#' @param bs Basis type for every margin.
#' @param k Basis dimension per margin; `NA` lets mgcv choose.
#' @param interactionOnly Whether to build a `ti()` term without main effects.
#' @return An object of class `pcSmooth`.
#' @family smooth specs
#' @export
tensorTerm <- function(vars, bs = c("cr", "tp", "ps"), k = NA, interactionOnly = FALSE) {
  bs <- match.arg(bs)
  if (!is.character(vars) || length(vars) < 2L) {
    stop("`vars` must name at least two covariates.", call. = FALSE)
  }
  k <- rep_len(k, length(vars))
  type <- if (interactionOnly) "ti" else "te"
  spec <- list(type = type, vars = vars, bs = bs, k = k, by = NULL)
  class(spec) <- "pcSmooth"
  return(spec)
}

#' Render a smooth specification
#'
#' Turns a smooth specification into the term string mgcv expects.
#'
#' @param x An object of class `pcSmooth`.
#' @param ... Unused.
#' @return A single string such as `"s(age, bs = \"cr\", k = 10)"`.
#' @export
format.pcSmooth <- function(x, ...) {
  args <- c(x$vars, sprintf("bs = \"%s\"", x$bs))
  if (x$type == "s") {
    if (x$k != -1) {
      args <- c(args, sprintf("k = %d", as.integer(x$k)))
    }
  } else if (!all(is.na(x$k))) {
    args <- c(args, sprintf("k = c(%s)", paste(as.integer(x$k), collapse = ", ")))
  }
  if (!is.null(x$by)) {
    args <- c(args, sprintf("by = %s", x$by))
  }
  return(sprintf("%s(%s)", x$type, paste(args, collapse = ", ")))
}

#' Build a GAM formula
#'
#' Combines a response, smooth specifications and parametric terms into a
#' formula for [mgcv::gam()].
#'
#' @param response Name of the response variable.
#' @param smooths A list of `pcSmooth` objects.
#' @param parametric Character vector of parametric terms.
#' @param env Environment for the created formula.
#' @return A formula.
#' @export
gamFormula <- function(response, smooths, parametric = character(), env = parent.frame()) {
  if (inherits(smooths, "pcSmooth")) {
    smooths <- list(smooths)
  }
  smoothTerms <- vapply(smooths, format, character(1))
  byVars <- unlist(lapply(smooths, `[[`, "by"))
  # factor-by smooths need the factor's main effect to stay identifiable
  parametric <- union(parametric, byVars)
  rhs <- paste(c(parametric, smoothTerms), collapse = " + ")
  if (!nzchar(rhs)) {
    rhs <- "1"
  }
  return(stats::as.formula(paste(response, "~", rhs), env = env))
}

#' Check basis sizes against the data
#'
#' Compares each smooth's basis dimension with the number of distinct values
#' of its covariate, since mgcv refuses a basis larger than that.
#'
#' @param smooths A list of `pcSmooth` objects.
#' @param data A data frame containing the covariates.
#' @return A data frame with one row per covariate and a logical `ok` column.
#' @export
checkBasisSize <- function(smooths, data) {
  if (inherits(smooths, "pcSmooth")) {
    smooths <- list(smooths)
  }
  rows <- list()
  for (sm in smooths) {
    for (j in seq_along(sm$vars)) {
      v <- sm$vars[j]
      if (!v %in% names(data)) {
        stop(sprintf("Covariate `%s` not found in `data`.", v), call. = FALSE)
      }
      nUnique <- length(unique(stats::na.omit(data[[v]])))
      k <- sm$k[j]
      if (is.na(k) || k == -1) {
        k <- if (sm$type == "s") 10L else 5L
      }
      rows[[length(rows) + 1L]] <- data.frame(
        term = format(sm),
        var = v,
        k = k,
        unique = nUnique,
        ok = k <= nUnique,
        stringsAsFactors = FALSE
      )
    }
  }
  return(do.call(rbind, rows))
}

#' Fit a GAM from specifications
#'
#' Checks basis sizes, shrinks any that are too large, builds the formula and
#' fits it with [mgcv::gam()].
#'
#' @param response Name of the response variable.
#' @param smooths A list of `pcSmooth` objects.
#' @param data A data frame containing the covariates.
#' @param family A family object passed to [mgcv::gam()].
#' @param method Smoothing parameter estimation method.
#' @param ... Passed to [mgcv::gam()].
#' @return A fitted `gam` object.
#' @export
fitGamSpec <- function(response, smooths, data, family = stats::gaussian(),
                       method = c("REML", "ML", "GCV.Cp"), ...) {
  method <- match.arg(method)
  if (inherits(smooths, "pcSmooth")) {
    smooths <- list(smooths)
  }
  sizes <- checkBasisSize(smooths, data)
  if (!all(sizes$ok)) {
    tooBig <- sizes[!sizes$ok, , drop = FALSE]
    for (i in seq_along(smooths)) {
      hit <- tooBig$term == format(smooths[[i]])
      if (any(hit)) {
        smooths[[i]]$k <- pmax(3L, min(tooBig$unique[hit]) - 1L)
      }
    }
    warning("Reduced basis size for: ", paste(unique(tooBig$term), collapse = ", "))
  }
  f <- gamFormula(response, smooths)
  fit <- tryCatch(
    mgcv::gam(f, data = data, family = family, method = method, ...),
    error = function(e) {
      stop(
        sprintf("GAM fit failed for `%s`: %s", deparse1(f), conditionMessage(e)),
        call. = FALSE
      )
    },
    warning = function(w) {
      warning(
        sprintf("GAM fit for `%s` warned: %s", deparse1(f), conditionMessage(w)),
        call. = FALSE
      )
      suppressWarnings(mgcv::gam(f, data = data, family = family, method = method, ...))
    }
  )
  return(fit)
}

#' Summarise smooth terms of a GAM
#'
#' Extracts the smooth-term table from a fitted GAM as a data frame with the
#' effective degrees of freedom and approximate significance.
#'
#' @param fit A fitted `gam` object.
#' @return A data frame with one row per smooth term.
#' @export
smoothSummary <- function(fit) {
  stopifnot(inherits(fit, "gam"))
  sTable <- summary(fit)$s.table
  if (is.null(sTable) || nrow(sTable) == 0L) {
    return(data.frame(term = character(), edf = numeric(), p.value = numeric()))
  }
  out <- data.frame(
    term = rownames(sTable),
    edf = sTable[, "edf"],
    ref.df = sTable[, "Ref.df"],
    statistic = sTable[, 3L],
    p.value = sTable[, "p-value"],
    stringsAsFactors = FALSE
  )
  out$wiggly <- out$edf > 1.5
  rownames(out) <- NULL
  return(out)
}
