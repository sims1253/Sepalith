#' Number of units and periods in a panel
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @return A named integer vector with `units`, `periods` and `rows`.
#' @family panel shape helpers
#' @export
panelDims = function(dt, id, time) {
  c(
    units = data.table::uniqueN(dt[[id]]),
    periods = data.table::uniqueN(dt[[time]]),
    rows = nrow(dt)
  )
}

#' Is a panel balanced?
#'
#' A panel is balanced when every unit is observed in every period.
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @return A single logical.
#' @family panel shape helpers
#' @export
isBalanced = function(dt, id, time) {
  checkPanelKeys(dt, id, time)
  dims = panelDims(dt, id, time)
  dims[["rows"]] == dims[["units"]] * dims[["periods"]]
}

#' Observations per unit
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @return A data.table with `id`, `nObs`, `firstPeriod` and `lastPeriod`.
#' @family panel shape helpers
#' @export
unitCoverage = function(dt, id, time) {
  checkPanelKeys(dt, id, time)
  out = dt[, .(
    nObs = .N,
    firstPeriod = min(get(time)),
    lastPeriod = max(get(time))
  ), by = id]
  data.table::setorderv(out, id)
  out[]
}

#' Balance a panel
#'
#' Either drops units that are not observed in every period, or expands the
#' panel to the full id-time grid with `NA` for unobserved cells.
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param how Either `"drop"` or `"expand"`.
#' @return A new data.table keyed by `id` and `time`.
#' @export
balancePanel = function(dt, id, time, how = c("drop", "expand")) {
  how = match.arg(how)
  checkPanelKeys(dt, id, time)
  nPeriods = data.table::uniqueN(dt[[time]])
  out = switch(how,
    drop = {
      keep = dt[, .N, by = id][N == nPeriods][[id]]
      dt[get(id) %in% keep]
    },
    expand = {
      grid = data.table::CJ(unique(dt[[id]]), unique(dt[[time]]))
      data.table::setnames(grid, c(id, time))
      dt[grid, on = c(id, time)]
    }
  )
  data.table::setkeyv(out, c(id, time))
  out[]
}

#' Attrition table
#'
#' For each period, counts the units present, the units entering for the
#' first time and the units seen for the last time.
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @return A data.table with one row per period.
#' @export
attritionTable = function(dt, id, time) {
  cover = unitCoverage(dt, id, time)
  periods = sort(unique(dt[[time]]))
  out = data.table::data.table(period = periods)
  out[, present := vapply(periods, function(p) {
    data.table::uniqueN(dt[get(time) == p][[id]])
  }, integer(1))]
  out[, entering := vapply(periods, function(p) sum(cover$firstPeriod == p), integer(1))]
  out[, leaving := vapply(periods, function(p) sum(cover$lastPeriod == p), integer(1))]
  out[, retention := present / data.table::shift(present)]
  data.table::setnames(out, "period", time)
  out[]
}

#' Describe the shape of a panel
#'
#' Prints a compact summary and returns the components invisibly.
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @return A list with `dims`, `balanced` and `coverage`, invisibly.
#' @export
describePanel = function(dt, id, time) {
  dims = panelDims(dt, id, time)
  balanced = isBalanced(dt, id, time)
  coverage = unitCoverage(dt, id, time)
  cat(sprintf("Panel: %d units x %d periods (%d rows)\n",
    dims[["units"]], dims[["periods"]], dims[["rows"]]))
  cat("Balanced:", if (balanced) "yes" else "no", "\n")
  if (!balanced) {
    q = stats::quantile(coverage$nObs, c(0, 0.5, 1))
    cat(sprintf("Obs per unit: min %d, median %g, max %d\n",
      as.integer(q[1]), q[2], as.integer(q[3])))
  }
  invisible(list(dims = dims, balanced = balanced, coverage = coverage))
}
