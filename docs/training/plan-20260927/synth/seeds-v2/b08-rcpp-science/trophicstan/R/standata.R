#' Check a vector of non-negative counts
#'
#' @param x Vector to check.
#' @param arg Name of the argument, used in error messages.
#' @return `x` as an integer vector, if all checks pass.
#' @noRd
check_counts <- function(x, arg) {
  if (!is.numeric(x) || anyNA(x)) {
    stop(sprintf("`%s` must be a numeric vector without missing values.", arg),
      call. = FALSE)
  }
  if (any(x < 0) || any(x != round(x))) {
    stop(sprintf("`%s` must contain non-negative whole numbers.", arg), call. = FALSE)
  }

  return(as.integer(x))
}

#' Check a vector of positive exposures
#'
#' @param x Vector to check.
#' @param arg Name of the argument, used in error messages.
#' @noRd
check_exposure <- function(x, arg) {
  if (!is.numeric(x) || anyNA(x)) {
    stop(sprintf("`%s` must be a numeric vector without missing values.", arg),
      call. = FALSE)
  }
  if (any(x <= 0)) {
    stop(sprintf("`%s` must be strictly positive.", arg), call. = FALSE)
  }
  return(as.double(x))
}

#' Encode a grouping variable as consecutive integers
#'
#' Stan needs group indices in `1:J`; this keeps the level labels so that
#' posterior summaries can be relabelled afterwards.
#'
#' @param g A vector of group labels.
#' @return A list with integer `index`, character `levels` and count `J`.
#' @keywords internal
encode_groups <- function(g) {
  if (anyNA(g)) {
    stop("Grouping variable must not contain missing values.", call. = FALSE)
  }

  # droplevels() matters when `g` is a factor with unused levels
  f <- droplevels(factor(g))
  return(list(index = as.integer(f), levels = levels(f), J = nlevels(f)))
}

#' Build Stan data for a hierarchical Poisson count model
#'
#' Prepares the data list for a model with log-linear predictors, an exposure
#' offset and varying intercepts by site.
#'
#' @param data A data frame with one row per survey visit.
#' @param count Name of the column holding observed counts.
#' @param exposure Name of the column holding survey effort, or `NULL` for unit
#'   effort.
#' @param site Name of the column identifying the survey site.
#' @param covariates Character vector of numeric covariate column names.
#' @param scale Logical; centre and scale covariates before modelling?
#' @return A named list suitable for `rstan::sampling()` or
#'   `cmdstanr::sample()`, with attribute `site_levels`.
#' @export
standata_poisson <- function(data, count, exposure = NULL, site = "site",
                             covariates = character(), scale = TRUE) {
  missing_cols <- setdiff(c(count, exposure, site, covariates), names(data))
  if (length(missing_cols) > 0L) {
    stop("Column(s) not found in `data`: ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }

  y <- check_counts(data[[count]], count)
  # Unit effort gives a zero offset on the log scale
  effort <- rep(1, nrow(data))
  if (!is.null(exposure)) {
    effort <- check_exposure(data[[exposure]], exposure)
  }
  groups <- encode_groups(data[[site]])

  is_num <- vapply(data[covariates], is.numeric, logical(1))
  if (!all(is_num)) {
    stop("Covariate(s) must be numeric: ", paste(covariates[!is_num], collapse = ", "),
      call. = FALSE)
  }
  X <- as.matrix(data[, covariates, drop = FALSE])
  if (scale && ncol(X) > 0L) {
    # A constant column would turn into NaN and break the sampler
    sds <- apply(X, 2, stats::sd)
    if (any(sds == 0)) {
      stop("Cannot scale constant covariate(s): ",
        paste(covariates[sds == 0], collapse = ", "), call. = FALSE)
    }
    X <- base::scale(X)
  }

  out <- list(
    N = length(y),
    J = groups$J,
    K = ncol(X),
    y = y,
    log_offset = log(effort),
    site = groups$index,
    X = unname(X)
  )
  attr(out, "site_levels") <- groups$levels
  return(out)
}

#' Build Stan data for a single-season occupancy model
#'
#' Converts long detection records into the site-by-visit matrix layout that
#' occupancy models expect, padding unequal visit counts with zeros.
#'
#' @param data A data frame of detection records.
#' @param site Name of the column identifying the survey site.
#' @param visit Name of the column numbering repeat visits within a site.
#' @param detected Name of the 0/1 detection column.
#' @return A named list with `N`, `T`, `n_visits`, `y` and `any_detected`, with
#'   attribute `site_levels`.
#' @export
standata_occupancy <- function(data, site = "site", visit = "visit", detected = "detected") {
  missing_cols <- setdiff(c(site, visit, detected), names(data))
  if (length(missing_cols) > 0L) {
    stop("Column(s) not found in `data`: ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }

  det <- check_counts(data[[detected]], detected)
  if (any(det > 1L)) {
    stop(sprintf("`%s` must be coded 0/1.", detected), call. = FALSE)
  }
  groups <- encode_groups(data[[site]])
  v <- check_counts(data[[visit]], visit)
  if (any(v < 1L)) {
    stop(sprintf("`%s` must number visits from 1.", visit), call. = FALSE)
  }

  # Sites with fewer visits keep zeros in the unused columns; the model
  # uses `n_visits` to ignore them
  n_visits <- tabulate(groups$index, nbins = groups$J)
  y <- matrix(0L, nrow = groups$J, ncol = max(v))
  y[cbind(groups$index, v)] <- det

  out <- list(
    N = groups$J,
    T = ncol(y),
    n_visits = n_visits,
    y = y,
    any_detected = as.integer(rowSums(y) > 0L)
  )
  attr(out, "site_levels") <- groups$levels
  return(out)
}

#' Build Stan data for a binomial prevalence model
#'
#' Aggregates individual test results to positives and totals per stratum.
#'
#' @param data A data frame with one row per tested individual.
#' @param result Name of the logical or 0/1 test result column.
#' @param stratum Name of the column defining strata, e.g. region or age band.
#' @return A named list with `J`, `positives` and `tested`, with attribute
#'   `stratum_levels`.
#' @export
standata_prevalence <- function(data, result, stratum) {
  missing_cols <- setdiff(c(result, stratum), names(data))
  if (length(missing_cols) > 0L) {
    stop("Column(s) not found in `data`: ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }

  res <- check_counts(as.integer(data[[result]]), result)
  if (any(res > 1L)) {
    stop(sprintf("`%s` must be logical or coded 0/1.", result), call. = FALSE)
  }

  groups <- encode_groups(data[[stratum]])
  positives <- tabulate(groups$index[res == 1L], nbins = groups$J)
  tested <- tabulate(groups$index, nbins = groups$J)

  out <- list(J = groups$J, positives = positives, tested = tested)
  attr(out, "stratum_levels") <- groups$levels
  return(out)
}

#' Add prior hyperparameters to a Stan data list
#'
#' @param standata A named list as returned by one of the `standata_*()`
#'   builders.
#' @param ... Named numeric hyperparameters, e.g. `prior_sigma_scale = 1`.
#' @return `standata` with the hyperparameters appended.
#' @export
with_priors <- function(standata, ...) {
  priors <- list(...)
  if (length(priors) == 0L) {
    return(standata)
  }
  if (is.null(names(priors)) || any(!nzchar(names(priors)))) {
    stop("All priors must be named.", call. = FALSE)
  }

  # Silently overwriting a data element would be very hard to debug
  clash <- intersect(names(priors), names(standata))
  if (length(clash) > 0L) {
    stop("Prior name(s) clash with data: ", paste(clash, collapse = ", "), call. = FALSE)
  }
  ok <- vapply(priors, function(p) is.numeric(p) && all(is.finite(p)), logical(1))
  if (!all(ok)) {
    stop("Priors must be finite numbers: ", paste(names(priors)[!ok], collapse = ", "),
      call. = FALSE)
  }

  standata[names(priors)] <- priors
  return(standata)
}
