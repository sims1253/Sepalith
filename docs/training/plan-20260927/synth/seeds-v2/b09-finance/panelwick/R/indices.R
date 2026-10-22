#' Validate a long price-quantity table
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @keywords internal
checkPriceTable = function(prices, cols) {
  needed = c("item", "period", "price", "qty")
  if (!all(needed %in% names(cols))) {
    stop("`cols` must name: ", paste(needed, collapse = ", "), call. = FALSE)
  }

  absent = setdiff(unname(cols), names(prices))
  if (length(absent) > 0) {
    stop("Columns not found: ", paste(absent, collapse = ", "), call. = FALSE)
  }
  # zero prices would make every price relative infinite
  if (any(prices[[cols[["price"]]]] <= 0, na.rm = TRUE)) {
    stop("Prices must be strictly positive.", call. = FALSE)
  }
  invisible(prices)
}

#' Join each period's prices to the base period
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param base Label of the base period.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @return A data.table with columns `period`, `p1`, `q1`, `p0`, `q0`.
#' @keywords internal
alignToBase = function(prices, base, cols) {
  checkPriceTable(prices, cols)
  dt = prices[, c(cols[["item"]], cols[["period"]], cols[["price"]], cols[["qty"]]),
    with = FALSE]
  data.table::setnames(dt, c("item", "period", "p1", "q1"))

  baseRows = dt[period == base, .(item, p0 = p1, q0 = q1)]
  if (nrow(baseRows) == 0) {
    stop("No rows found for base period ", base, ".", call. = FALSE)
  }
  # items missing from the base period cannot be priced and are dropped
  baseRows[dt, on = "item", nomatch = NULL]
}

#' Laspeyres and Paasche price indices
#'
#' @description The Laspeyres index weights price relatives by base-period
#'   quantities; the Paasche index uses current-period quantities.
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param base Label of the base period.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @return A data.table with columns `period` and `index` (base = 100).
#' @rdname fixedBaseIndex
#' @export
laspeyresIndex = function(prices, base,
                          cols = c(item = "item", period = "period",
                                   price = "price", qty = "qty")) {
  dt = alignToBase(prices, base, cols)
  out = dt[, .(index = 100 * sum(p1 * q0) / sum(p0 * q0)), by = period]
  out[order(period)]
}

#' @rdname fixedBaseIndex
#' @export
paascheIndex = function(prices, base,
                        cols = c(item = "item", period = "period",
                                 price = "price", qty = "qty")) {
  dt = alignToBase(prices, base, cols)
  out = dt[, .(index = 100 * sum(p1 * q1) / sum(p0 * q1)), by = period]
  out[order(period)]
}

#' Fisher ideal price index
#'
#' @description The geometric mean of the Laspeyres and Paasche indices.
#'
#' @inheritParams laspeyresIndex
#' @return A data.table with columns `period` and `index` (base = 100).
#' @export
fisherIndex = function(prices, base,
                       cols = c(item = "item", period = "period",
                                price = "price", qty = "qty")) {
  las = laspeyresIndex(prices, base, cols)
  paa = paascheIndex(prices, base, cols)
  out = las[paa, on = "period", .(period, index = sqrt(index * i.index))]
  out[order(period)]
}

#' Chain-linked price index
#'
#' @description Computes a one-period index between each pair of adjacent
#'   periods and multiplies the links together, so the item basket can
#'   change over time.
#'
#' @param prices A data.table with one row per item and period holding
#'   prices and quantities.
#' @param cols Named character vector giving the `item`, `period`, `price`
#'   and `qty` column names.
#' @param formula Index formula used for each link.
#' @return A data.table with columns `period`, `link` and `index`.
#' @export
chainIndex = function(prices,
                      cols = c(item = "item", period = "period",
                               price = "price", qty = "qty"),
                      formula = c("fisher", "laspeyres", "paasche")) {
  formula = match.arg(formula)
  linkFun = switch(formula,
    fisher = fisherIndex,
    laspeyres = laspeyresIndex,
    paasche = paascheIndex
  )

  periods = sort(unique(prices[[cols[["period"]]]]))
  links = rep(NA_real_, length(periods))
  links[1] = 1

  for (i in seq_along(periods)[-1]) {
    # each link only sees the two adjacent periods
    pair = prices[get(cols[["period"]]) %in% periods[c(i - 1, i)]]
    step = linkFun(pair, periods[i - 1], cols)
    links[i] = step[period == periods[i], index] / 100
  }

  data.table::data.table(period = periods, link = links, index = 100 * cumprod(links))
}

#' Rebase an index table to a new reference period
#'
#' @param index A data.table with `period` and `index` columns.
#' @param base Label of the base period.
#' @return A copy of `index` with the `index` column rescaled.
#' @export
rebaseIndexTable = function(index, base) {
  ref = index[period == base, index]
  if (length(ref) != 1 || is.na(ref)) {
    stop("Base period must match exactly one non-missing row.", call. = FALSE)
  }

  out = data.table::copy(index)
  out[, index := 100 * index / ref]
  out[]
}
