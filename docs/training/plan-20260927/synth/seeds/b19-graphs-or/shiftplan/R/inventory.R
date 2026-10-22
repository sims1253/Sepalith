#' Economic order quantity
#'
#' Classic square-root formula balancing ordering and holding costs.
#'
#' @param demandRate Demand per period.
#' @param orderCost Fixed cost per order.
#' @param holdingCost Holding cost per unit per period.
#' @return A list with `quantity`, `ordersPerPeriod` and `totalCost`.
#' @export
#' @family inventoryFormulas
eoq = function(demandRate, orderCost, holdingCost) {
  stopifnot(demandRate > 0, orderCost > 0, holdingCost > 0)
  q = sqrt(2 * demandRate * orderCost / holdingCost)
  ordersPerPeriod = demandRate / q
  totalCost = ordersPerPeriod * orderCost + q / 2 * holdingCost
  list(quantity = q, ordersPerPeriod = ordersPerPeriod, totalCost = totalCost)
}

#' Reorder point under normally distributed lead-time demand
#'
#' @param demandMean Mean demand per period.
#' @param demandSd Standard deviation of demand per period.
#' @param leadTime Lead time in periods.
#' @param serviceLevel Target probability of no stockout during lead time.
#' @return A list with `reorderPoint`, `safetyStock` and `z`.
#' @export
#' @family inventoryFormulas
reorderPoint = function(demandMean, demandSd, leadTime, serviceLevel = 0.95) {
  stopifnot(serviceLevel > 0, serviceLevel < 1, leadTime >= 0)
  z = stats::qnorm(serviceLevel)
  safetyStock = z * demandSd * sqrt(leadTime)
  list(
    reorderPoint = demandMean * leadTime + safetyStock,
    safetyStock = safetyStock,
    z = z
  )
}

#' Simulate a continuous-review (s, Q) policy
#'
#' Each period: receive due orders, meet demand (lost sales), then place an
#' order of size `Q` if inventory position is at or below `s`.
#'
#' @param demand Numeric vector of demand per period.
#' @param s Reorder point.
#' @param Q Order quantity.
#' @param leadTime Lead time in whole periods.
#' @param initial Starting on-hand inventory.
#' @return A data frame with one row per period: `period`, `demand`,
#'   `onHand`, `sold`, `lost`, `ordered` and `received`.
#' @export
#' @family inventorySim
simulateSQ = function(demand, s, Q, leadTime = 2L, initial = s + Q) {
  n = length(demand)
  onHand = initial
  pipeline = numeric(n + leadTime + 1L)
  out = data.frame(
    period = seq_len(n), demand = demand, onHand = 0, sold = 0,
    lost = 0, ordered = 0, received = 0
  )
  for (t in seq_len(n)) {
    received = pipeline[t]
    onHand = onHand + received
    sold = min(onHand, demand[t])
    onHand = onHand - sold
    position = onHand + sum(pipeline[(t + 1L):(t + leadTime)])
    ordered = 0
    if (position <= s) {
      ordered = Q
      pipeline[t + leadTime] = pipeline[t + leadTime] + Q
    }
    out$onHand[t] = onHand
    out$sold[t] = sold
    out$lost[t] = demand[t] - sold
    out$ordered[t] = ordered
    out$received[t] = received
  }
  out
}

#' Simulate a periodic-review (R, S) policy
#'
#' Every `R` periods: order up to level `S` based on inventory position.
#' Demand is met from stock with lost sales.
#'
#' @param demand Numeric vector of demand per period.
#' @param R Review interval in periods.
#' @param S Order-up-to level.
#' @param leadTime Lead time in whole periods.
#' @param initial Starting on-hand inventory.
#' @return A data frame with one row per period: `period`, `demand`,
#'   `onHand`, `sold`, `lost`, `ordered` and `received`.
#' @export
#' @family inventorySim
simulateRS = function(demand, R, S, leadTime = 2L, initial = S) {
  n = length(demand)
  onHand = initial
  pipeline = numeric(n + leadTime + 1L)
  out = data.frame(
    period = seq_len(n), demand = demand, onHand = 0, sold = 0,
    lost = 0, ordered = 0, received = 0
  )
  for (t in seq_len(n)) {
    received = pipeline[t]
    onHand = onHand + received
    sold = min(onHand, demand[t])
    onHand = onHand - sold
    position = onHand + sum(pipeline[(t + 1L):(t + leadTime)])
    ordered = 0
    if (t %% R == 0 && position < S) {
      ordered = S - position
      pipeline[t + leadTime] = pipeline[t + leadTime] + ordered
    }
    out$onHand[t] = onHand
    out$sold[t] = sold
    out$lost[t] = demand[t] - sold
    out$ordered[t] = ordered
    out$received[t] = received
  }
  out
}

#' Cost and service metrics of a simulated policy
#'
#' @param sim A data frame as returned by [simulateSQ()] or [simulateRS()].
#' @param holdingCost Holding cost per unit per period.
#' @param orderCost Fixed cost per order.
#' @param lostSaleCost Penalty per unit of lost demand.
#' @return A one-row data frame with fill rate, cycle service level, order
#'   count and cost components.
#' @export
#' @family inventorySim
policyMetrics = function(sim, holdingCost, orderCost, lostSaleCost) {
  nOrders = sum(sim$ordered > 0)
  holding = sum(sim$onHand) * holdingCost
  ordering = nOrders * orderCost
  shortage = sum(sim$lost) * lostSaleCost
  data.frame(
    fillRate = sum(sim$sold) / sum(sim$demand),
    cycleService = mean(sim$lost == 0),
    nOrders = nOrders,
    avgOnHand = mean(sim$onHand),
    holding = holding,
    ordering = ordering,
    shortage = shortage,
    total = holding + ordering + shortage
  )
}

#' Grid-search an (s, Q) policy by simulation
#'
#' Evaluates every combination of `s` and `Q` on the same demand path and
#' returns the metrics sorted by total cost.
#'
#' @param demand Numeric vector of demand per period.
#' @param sGrid Numeric vector of candidate reorder points.
#' @param qGrid Numeric vector of candidate order quantities.
#' @param leadTime Lead time in whole periods.
#' @param holdingCost Holding cost per unit per period.
#' @param orderCost Fixed cost per order.
#' @param lostSaleCost Penalty per unit of lost demand.
#' @return A data frame with `s`, `Q` and the columns of [policyMetrics()],
#'   sorted by increasing `total`.
#' @export
#' @family inventorySim
tuneSQ = function(demand, sGrid, qGrid, leadTime = 2L,
                  holdingCost = 1, orderCost = 50, lostSaleCost = 10) {
  grid = expand.grid(s = sGrid, Q = qGrid)
  rows = vector("list", nrow(grid))
  for (k in seq_len(nrow(grid))) {
    sim = simulateSQ(demand, grid$s[k], grid$Q[k], leadTime = leadTime)
    met = policyMetrics(sim, holdingCost, orderCost, lostSaleCost)
    rows[[k]] = cbind(grid[k, , drop = FALSE], met)
  }
  res = do.call(rbind, rows)
  res = res[order(res$total), , drop = FALSE]
  rownames(res) = NULL
  res
}
