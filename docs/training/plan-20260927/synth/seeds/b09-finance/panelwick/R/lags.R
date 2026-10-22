#' Lag variables within each panel unit
#'
#' Rows are ordered by `time` within `id` before shifting, so the input
#' order does not matter.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @family group shifts
#' @importFrom data.table := .SD
#' @export
lagBy = function(dt, vars, id, time, n = 1L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  outCols = paste0(vars, "_lag", n)
  dt[, (outCols) := data.table::shift(.SD, n, type = "lag"), by = id, .SDcols = vars]
  invisible(dt)
}

#' Lead variables within each panel unit
#'
#' Rows are ordered by `time` within `id` before shifting, so the input
#' order does not matter.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @family group shifts
#' @importFrom data.table := .SD
#' @export
leadBy = function(dt, vars, id, time, n = 1L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  outCols = paste0(vars, "_lead", n)
  dt[, (outCols) := data.table::shift(.SD, n, type = "lead"), by = id, .SDcols = vars]
  invisible(dt)
}

#' Difference variables within each panel unit
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @family group shifts
#' @importFrom data.table := .SD
#' @export
diffBy = function(dt, vars, id, time, n = 1L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  outCols = paste0(vars, "_diff", n)
  dt[, (outCols) := lapply(.SD, function(v) v - data.table::shift(v, n)),
    by = id, .SDcols = vars]
  invisible(dt)
}

#' Growth rates within each panel unit
#'
#' Computes simple or log growth relative to `n` periods earlier.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param n Number of periods to shift by.
#' @param method Either `"simple"` or `"log"`.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @family group shifts
#' @importFrom data.table := .SD
#' @export
growthBy = function(dt, vars, id, time, n = 1L, method = c("simple", "log")) {
  method = match.arg(method)
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  growthFun = switch(method,
    simple = function(v) v / data.table::shift(v, n) - 1,
    log = function(v) log(v) - log(data.table::shift(v, n))
  )
  outCols = paste0(vars, "_growth", n)
  dt[, (outCols) := lapply(.SD, growthFun), by = id, .SDcols = vars]
  invisible(dt)
}

#' Rolling means within each panel unit
#'
#' Right-aligned rolling mean over `window` periods. Units shorter than the
#' window receive `NA` throughout.
#'
#' @param dt A data.table in long panel format.
#' @param vars Character vector of columns to shift.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param window Number of periods in each window.
#' @return `dt`, keyed by `id` and `time` and modified by reference, invisibly.
#' @importFrom data.table := .SD
#' @export
rollMeanBy = function(dt, vars, id, time, window = 3L) {
  checkPanelKeys(dt, id, time)
  data.table::setkeyv(dt, c(id, time))
  outCols = paste0(vars, "_roll", window)
  dt[, (outCols) := data.table::frollmean(.SD, window, align = "right"),
    by = id, .SDcols = vars]
  invisible(dt)
}

#' Fill gaps in the time index of each unit
#'
#' Inserts the missing id-time combinations between each unit's first and
#' last period so that lags refer to calendar neighbours, not row
#' neighbours. Inserted rows carry `NA` in all other columns.
#'
#' @param dt A data.table in long panel format.
#' @param id Name of the column identifying panel units.
#' @param time Name of the column identifying time periods.
#' @param step Spacing of the integer time index.
#' @return A new data.table keyed by `id` and `time`.
#' @export
fillTimeGaps = function(dt, id, time, step = 1L) {
  checkPanelKeys(dt, id, time)
  if (!is.numeric(dt[[time]])) {
    stop("`time` must refer to an integer period index.", call. = FALSE)
  }
  spans = dt[, .(start = min(get(time)), end = max(get(time))), by = id]
  grid = spans[, .(t = seq(start, end, by = step)), by = id]
  data.table::setnames(grid, "t", time)
  out = dt[grid, on = c(id, time)]
  out[, .inserted := FALSE]
  out[!dt, on = c(id, time), .inserted := TRUE]
  data.table::setkeyv(out, c(id, time))
  out[]
}
