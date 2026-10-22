#' Check that panel keys identify rows uniquely
#'
#' @description Stops with an informative message if `id` and `time` columns
#'   are missing or if any id-time pair appears more than once.
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @keywords internal
checkPanelKeys = function(dt, id, time) {
  if (!data.table::is.data.table(dt)) {
    stop("`dt` must be a data.table.", call. = FALSE)
  }
  missingCols = setdiff(c(id, time), names(dt))
  if (length(missingCols) > 0) {
    stop("Missing key columns: ", paste(missingCols, collapse = ", "), call. = FALSE)
  }

  dupes = dt[, .N, by = c(id, time)][N > 1]
  if (nrow(dupes) > 0) {
    stop(sprintf("%d id-time pairs occur more than once.", nrow(dupes)), call. = FALSE)
  }
  invisible(dt)
}

#' Within (fixed effects) transformation
#'
#' @description Subtracts the unit mean from each variable, optionally
#'   adding back the grand mean so that levels stay interpretable.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of numeric columns to transform.
#' @param id Name of the column identifying panel units.
#' @param addGrandMean Logical; add the overall mean back after demeaning?
#' @param suffix Suffix appended to the names of the new columns.
#' @return `dt`, modified by reference, invisibly.
#' @importFrom data.table := .SD
#' @export
withinTransform = function(dt, vars, id, addGrandMean = FALSE, suffix = "_within") {
  outCols = paste0(vars, suffix)
  dt[, (outCols) := lapply(.SD, function(v) v - mean(v, na.rm = TRUE)),
    by = id, .SDcols = vars]

  if (addGrandMean) {
    grand = dt[, lapply(.SD, mean, na.rm = TRUE), .SDcols = vars]
    for (j in seq_along(vars)) {
      data.table::set(dt, j = outCols[j], value = dt[[outCols[j]]] + grand[[j]])
    }
  }
  invisible(dt)
}

#' Between transformation
#'
#' @description Replaces each variable by its unit mean, repeated on every
#'   row of the unit.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of numeric columns to transform.
#' @param id Name of the column identifying panel units.
#' @param suffix Suffix appended to the names of the new columns.
#' @return `dt`, modified by reference, invisibly.
#' @export
betweenTransform = function(dt, vars, id, suffix = "_between") {
  outCols = paste0(vars, suffix)
  dt[, (outCols) := lapply(.SD, function(v) rep(mean(v, na.rm = TRUE), .N)),
    by = id, .SDcols = vars]
  invisible(dt)
}

#' Standardise variables within each unit
#'
#' @description Centres and scales each variable by the unit mean and
#'   standard deviation. Units with a single observation receive `NA`.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of numeric columns to transform.
#' @param id Name of the column identifying panel units.
#' @param suffix Suffix appended to the names of the new columns.
#' @return `dt`, modified by reference, invisibly.
#' @export
standardiseWithin = function(dt, vars, id, suffix = "_z") {
  outCols = paste0(vars, suffix)
  dt[, (outCols) := lapply(.SD, function(v) {
    s = stats::sd(v, na.rm = TRUE)
    # constant or single-row units have no scale to divide by
    if (is.na(s) || s == 0) rep(NA_real_, .N) else (v - mean(v, na.rm = TRUE)) / s
  }), by = id, .SDcols = vars]
  invisible(dt)
}

#' Quasi-demeaning for random effects
#'
#' @description Subtracts `theta` times the unit mean, where `theta` depends
#'   on the supplied variance components and on the number of rows in each
#'   unit.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of numeric columns to transform.
#' @param id Name of the column identifying panel units.
#' @param sigmaE,sigmaU Idiosyncratic and unit-level error standard
#'   deviations.
#' @param suffix Suffix appended to the names of the new columns.
#' @return `dt`, modified by reference, invisibly, with a `theta` attribute.
#' @importFrom data.table := .SD .N
#' @export
quasiDemean = function(dt, vars, id, sigmaE, sigmaU, suffix = "_re") {
  if (sigmaE <= 0 || sigmaU < 0) {
    stop("`sigmaE` must be positive and `sigmaU` non-negative.", call. = FALSE)
  }
  outCols = paste0(vars, suffix)

  # theta varies by unit in unbalanced panels, so keep it as a temp column
  dt[, .theta := 1 - sigmaE / sqrt(sigmaE^2 + .N * sigmaU^2), by = id]
  dt[, (outCols) := lapply(.SD, function(v) v - .theta * mean(v, na.rm = TRUE)),
    by = id, .SDcols = vars]

  theta = unique(dt[, c(id, ".theta"), with = FALSE])
  dt[, .theta := NULL]
  data.table::setattr(dt, "theta", theta)
  invisible(dt)
}

#' Decompose variance into within and between components
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of numeric columns to transform.
#' @param id Name of the column identifying panel units.
#' @return A data.table with one row per variable and columns `overall`,
#'   `between`, `within` and `withinShare`.
#' @export
varianceDecomposition = function(dt, vars, id) {
  out = data.table::data.table(
    variable = vars,
    overall = NA_real_,
    between = NA_real_,
    within = NA_real_
  )

  for (i in seq_along(vars)) {
    v = vars[i]
    unitMeans = dt[, .(m = mean(get(v), na.rm = TRUE)), by = id]
    deviations = dt[, get(v) - mean(get(v), na.rm = TRUE), by = id]$V1
    data.table::set(out, i, "overall", stats::sd(dt[[v]], na.rm = TRUE))
    data.table::set(out, i, "between", stats::sd(unitMeans$m, na.rm = TRUE))
    data.table::set(out, i, "within", stats::sd(deviations, na.rm = TRUE))
  }

  out[, withinShare := within^2 / (within^2 + between^2)]
  out[]
}

#' Fixed effects regression by the within estimator
#'
#' @description Demeans the response and regressors within units, fits
#'   least squares on the transformed data and corrects the residual degrees
#'   of freedom for the absorbed unit means.
#'
#' @param dt A data.table in long panel format.
#' @param y Name of the response column.
#' @param x Character vector of regressor columns.
#' @param id Name of the column identifying panel units.
#' @param cluster Logical; report standard errors clustered by `id`?
#' @return A data.table of coefficients with `estimate`, `stdError` and
#'   `tValue` columns, with `nUnits` and `df` attributes.
#' @export
withinRegression = function(dt, y, x, id, cluster = TRUE) {
  vars = c(y, x)
  work = stats::na.omit(dt[, c(id, vars), with = FALSE])
  if (nrow(work) == 0) {
    stop("No complete rows for the requested variables.", call. = FALSE)
  }

  withinTransform(work, vars, id, suffix = "_w")
  X = as.matrix(work[, paste0(x, "_w"), with = FALSE])
  Y = work[[paste0(y, "_w")]]
  XtXinv = solve(crossprod(X))
  beta = drop(XtXinv %*% crossprod(X, Y))
  resid = Y - drop(X %*% beta)

  # one degree of freedom is lost per absorbed unit mean
  nUnits = data.table::uniqueN(work[[id]])
  df = nrow(work) - nUnits - length(x)
  if (df <= 0) {
    stop("Not enough within-unit variation to estimate the model.", call. = FALSE)
  }

  if (cluster) {
    if (nUnits < 2) {
      stop("Clustered standard errors need at least two units.", call. = FALSE)
    }
    scores = rowsum(X * resid, work[[id]])
    meat = crossprod(scores)
    adj = nUnits / (nUnits - 1) * (nrow(work) - 1) / df
    vcov = adj * XtXinv %*% meat %*% XtXinv
  } else {
    sigma2 = sum(resid^2) / df
    vcov = sigma2 * XtXinv
  }

  se = sqrt(diag(vcov))
  out = data.table::data.table(
    term = x,
    estimate = beta,
    stdError = se,
    tValue = beta / se
  )
  data.table::setattr(out, "nUnits", nUnits)
  data.table::setattr(out, "df", df)
  out[]
}
