#' Order quantity under an inventory policy
#'
#' @param position Inventory position: on hand plus on order minus backorders.
#' @param policy Policy type, one of `"sq"`, `"ss"`, `"rs"`, `"rss"` or
#'   `"base.stock"`.
#' @param params A list with the policy parameters `s`, `S`, `Q` and `R` as
#'   needed by `policy`.
#' @param period Current period number, used by the periodic-review policies.
#' @return The quantity to order now, zero when no order is due.
#' @export
replenish.qty <- function(position, policy = c("sq", "ss", "rs", "rss", "base.stock"),
                          params, period = 1) {
  policy <- match.arg(policy)

  qty <- switch(policy,
    sq = {
      # Continuous review: order whole lots of Q until above s
      if (position > params$s) return(0)
      lots <- floor((params$s - position) / params$Q) + 1
      lots * params$Q
    },
    ss = {
      # Continuous review: order up to S once at or below s
      if (position > params$s) return(0)
      params$S - position
    },
    rs = {
      # Periodic review: order up to S every R periods
      if (period %% params$R != 0) return(0)
      params$S - position
    },
    rss = {
      # Periodic review: order up to S every R periods if at or below s
      if (period %% params$R != 0) return(0)
      if (position > params$s) return(0)
      params$S - position
    },
    base.stock = {
      # Replace every unit of demand as soon as it happens
      if (position >= params$S) return(0)
      params$S - position
    }
  )
  return(max(qty, 0))
}

#' Parameters of an inventory policy from demand statistics
#'
#' @param demand.mean Mean demand per period.
#' @param demand.sd Standard deviation of demand per period.
#' @param lead.time Replenishment lead time in periods.
#' @param policy Policy type, one of `"sq"`, `"ss"`, `"rs"`, `"rss"` or
#'   `"base.stock"`.
#' @param review Review interval in periods for the periodic policies.
#' @param service Cycle service level used for the safety factor.
#' @param ordering.cost Fixed cost of placing one order.
#' @param holding.cost Cost of holding one unit for one period.
#' @return A list with elements `s`, `S`, `Q`, `R` and `safety`; parameters
#'   the policy does not use are `NA`.
#' @export
policy.parameters <- function(demand.mean, demand.sd, lead.time,
                              policy = c("sq", "ss", "rs", "rss", "base.stock"),
                              review = 1, service = 0.95, ordering.cost = 50,
                              holding.cost = 1) {
  policy <- match.arg(policy)
  z <- stats::qnorm(service)
  eoq <- sqrt(2 * demand.mean * ordering.cost / holding.cost)
  out <- list(s = NA_real_, S = NA_real_, Q = NA_real_, R = NA_real_, safety = NA_real_)

  out <- switch(policy,
    sq = {
      # Cover the lead time; lot size from the EOQ
      out$safety <- z * demand.sd * sqrt(lead.time)
      out$s <- demand.mean * lead.time + out$safety
      out$Q <- eoq
      out
    },
    ss = {
      # Cover the lead time; order up to s plus one EOQ
      out$safety <- z * demand.sd * sqrt(lead.time)
      out$s <- demand.mean * lead.time + out$safety
      out$S <- out$s + eoq
      out
    },
    rs = {
      # Cover the review interval plus the lead time
      out$safety <- z * demand.sd * sqrt(review + lead.time)
      out$S <- demand.mean * (review + lead.time) + out$safety
      out$R <- review
      out
    },
    rss = {
      # Cover the review interval plus the lead time; reorder point as for ss
      out$safety <- z * demand.sd * sqrt(review + lead.time)
      out$s <- demand.mean * lead.time + z * demand.sd * sqrt(lead.time)
      out$S <- demand.mean * (review + lead.time) + out$safety
      out$R <- review
      out
    },
    base.stock = {
      # Cover the lead time plus the one-period replenishment delay
      out$safety <- z * demand.sd * sqrt(lead.time + 1)
      out$S <- demand.mean * (lead.time + 1) + out$safety
      out
    }
  )
  return(out)
}

#' Validate the parameters of an inventory policy
#'
#' @param params A list with the policy parameters `s`, `S`, `Q` and `R`.
#' @param policy Policy type, one of `"sq"`, `"ss"`, `"rs"`, `"rss"` or
#'   `"base.stock"`.
#' @return `params`, invisibly.
#' @keywords internal
validate.policy.params <- function(params, policy = c("sq", "ss", "rs", "rss", "base.stock")) {
  policy <- match.arg(policy)
  needs <- list(sq = c("s", "Q"), ss = c("s", "S"), rs = c("R", "S"),
    rss = c("R", "s", "S"), base.stock = "S")[[policy]]
  missing.params <- needs[!needs %in% names(params) | vapply(needs, function(p) {
    is.null(params[[p]]) || anyNA(params[[p]])
  }, logical(1))]
  if (length(missing.params) > 0) {
    stop(sprintf("Policy '%s' needs parameter(s): %s.", policy,
      paste(missing.params, collapse = ", ")), call. = FALSE)
  }

  # Reorder point s
  if ("s" %in% needs && (!is.numeric(params$s) || length(params$s) != 1)) {
    stop(sprintf("Parameter `%s` must be a single number.", "s"), call. = FALSE)
  }
  # Order-up-to level S
  if ("S" %in% needs && (!is.numeric(params$S) || length(params$S) != 1)) {
    stop(sprintf("Parameter `%s` must be a single number.", "S"), call. = FALSE)
  }
  if ("S" %in% needs && "s" %in% needs && params$S <= params$s) {
    stop(sprintf("Parameter `%s` must exceed `%s`.", "S", "s"), call. = FALSE)
  }
  # Lot size Q
  if ("Q" %in% needs && (!is.numeric(params$Q) || length(params$Q) != 1)) {
    stop(sprintf("Parameter `%s` must be a single number.", "Q"), call. = FALSE)
  }
  if ("Q" %in% needs && params$Q <= 0) {
    stop(sprintf("Parameter `%s` must be positive.", "Q"), call. = FALSE)
  }
  # Review interval R
  if ("R" %in% needs && (!is.numeric(params$R) || length(params$R) != 1)) {
    stop(sprintf("Parameter `%s` must be a single number.", "R"), call. = FALSE)
  }
  if ("R" %in% needs && (params$R < 1 || params$R != round(params$R))) {
    stop(sprintf("Parameter `%s` must be a whole number of periods.", "R"), call. = FALSE)
  }
  return(invisible(params))
}

#' Simulate an inventory policy against a demand series
#'
#' Unmet demand is backordered. Orders placed in period `t` arrive at the
#' start of period `t + lead.time + 1`.
#'
#' @param demand Numeric demand per period.
#' @param policy Policy type, one of `"sq"`, `"ss"`, `"rs"`, `"rss"` or
#'   `"base.stock"`.
#' @param params A list with the policy parameters `s`, `S`, `Q` and `R`.
#' @param lead.time Replenishment lead time in whole periods.
#' @param initial Stock on hand at the start.
#' @return A data frame with columns `period`, `demand`, `receipt`,
#'   `on.hand`, `backorder`, `position` and `order`.
#' @export
policy.simulation <- function(demand, policy = c("sq", "ss", "rs", "rss", "base.stock"),
                            params, lead.time = 1, initial = 0) {
  policy <- match.arg(policy)
  validate.policy.params(params, policy)
  n <- length(demand)
  pipeline <- numeric(n + lead.time + 1)
  receipt <- on.hand <- backorder <- position <- order <- numeric(n)
  stock <- initial
  owed <- 0

  for (t in seq_len(n)) {
    # Receive what arrives this period and serve backorders first
    receipt[t] <- pipeline[t]
    stock <- stock + receipt[t]
    served <- min(stock, owed)
    stock <- stock - served
    owed <- owed - served
    # Serve today's demand, backordering the rest
    filled <- min(stock, demand[t])
    stock <- stock - filled
    owed <- owed + demand[t] - filled
    # Review the position and order
    on.order <- sum(pipeline[(t + 1):length(pipeline)])
    position[t] <- stock + on.order - owed
    order[t] <- replenish.qty(position[t], policy, params, period = t)
    pipeline[t + lead.time + 1] <- pipeline[t + lead.time + 1] + order[t]
    on.hand[t] <- stock
    backorder[t] <- owed
  }

  return(data.frame(period = seq_len(n), demand = demand, receipt = receipt,
    on.hand = on.hand, backorder = backorder, position = position, order = order))
}

#' Cost of a simulated inventory policy
#'
#' @param sim A data frame as returned by [policy.simulation()].
#' @param holding Cost of holding one unit for one period.
#' @param shortage Cost of one unit backordered for one period.
#' @param ordering Fixed cost of placing one order.
#' @param unit.cost Purchase price of one unit.
#' @return A list with elements `holding`, `shortage`, `ordering`,
#'   `purchase` and `total`.
#' @export
policy.cost <- function(sim, holding = 1, shortage = 10, ordering = 50, unit.cost = 0) {
  # Holding: on-hand stock at the end of every period
  holding.units <- sum(sim$on.hand)
  holding.cost <- holding.units * holding
  # Shortage: backordered units at the end of every period
  shortage.units <- sum(sim$backorder)
  shortage.cost <- shortage.units * shortage
  # Ordering: one fixed charge per order placed
  ordering.count <- sum(sim$order > 0)
  ordering.cost <- ordering.count * ordering
  # Purchase: every unit ordered at the unit price
  purchase.units <- sum(sim$order)
  purchase.cost <- purchase.units * unit.cost

  return(list(
    holding = holding.cost,
    shortage = shortage.cost,
    ordering = ordering.cost,
    purchase = purchase.cost,
    total = holding.cost + shortage.cost + ordering.cost + purchase.cost
  ))
}
