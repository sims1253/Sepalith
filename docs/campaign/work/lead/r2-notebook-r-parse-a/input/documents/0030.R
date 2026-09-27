panel2cs2 <- function(data, yname, idname, tname, balance_panel = TRUE) {
  # check that only 2 periods of data
  if (length(unique(data[[tname]])) != 2) {
    stop("panel2cs only for 2 periods of panel data")
  }

  # balance the data, just in case
  if (balance_panel) {
    data <- make_balanced_panel(data, idname, tname)
  }

  # data.table sorting (fast and memory efficient)
  data.table::setDT(data)
  data.table::setorderv(data, cols = c(idname, tname))

  # Fast global shift is valid for balanced two-period panels.  In unbalanced
  # panels, shift within id to avoid borrowing outcomes from the next unit.
  if (balance_panel) {
    data$.y1 <- data.table::shift(data[[yname]], -1)
  } else {
    data[, c(".y1") := data.table::shift(get(yname), -1), by = idname]
  }
  data$.y0 <- data[[yname]]
  data$.dy <- data$.y1 - data$.y0

  # Subset to first period after computing the logical index explicitly.
  first_period <- min(data[[tname]])
  first_period_idx <- data[[tname]] == first_period
  data <- data[first_period_idx, ]

  data
}