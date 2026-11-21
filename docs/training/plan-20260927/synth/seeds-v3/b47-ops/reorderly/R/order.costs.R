#' Economic order quantity under several cost models
#'
#' @param demand Demand per year.
#' @param ordering Fixed cost of placing one order.
#' @param holding Cost of holding one unit for one year.
#' @param model Cost model, one of `"classic"`, `"backorder"`, `"production"`
#'   or `"discount"`.
#' @param shortage Cost of one unit backordered for one year (`"backorder"`).
#' @param production.rate Units produced per year (`"production"`).
#' @param breaks A data frame with columns `min.qty` and `price` (`"discount"`);
#'   `holding` is then read as a share of the unit price.
#' @return A list with elements `q`, `orders`, `cycle` and `cost`.
#' @export
eoq.model <- function(demand, ordering, holding,
                      model = c("classic", "backorder", "production", "discount"),
                      shortage = NULL, production.rate = NULL, breaks = NULL) {
  model <- match.arg(model)

  res <- switch(model,
    classic = {
      # Balance ordering against holding cost
      q <- sqrt(2 * demand * ordering / holding)
      cost <- demand / q * ordering + q / 2 * holding
      list(q = q, cost = cost)
    },
    backorder = {
      # Planned backorders lower the average stock held
      ratio <- (holding + shortage) / shortage
      q <- sqrt(2 * demand * ordering / holding * ratio)
      cost <- sqrt(2 * demand * ordering * holding / ratio)
      list(q = q, cost = cost)
    },
    production = {
      # Stock builds up only at the net production rate
      ratio <- 1 - demand / production.rate
      q <- sqrt(2 * demand * ordering / (holding * ratio))
      cost <- demand / q * ordering + q / 2 * ratio * holding
      list(q = q, cost = cost)
    },
    discount = {
      # Best EOQ per price break, raised to the break minimum, plus purchase cost
      q.all <- pmax(sqrt(2 * demand * ordering / (holding * breaks$price)), breaks$min.qty)
      cost.all <- demand / q.all * ordering + q.all / 2 * holding * breaks$price +
        demand * breaks$price
      best <- which.min(cost.all)
      list(q = q.all[best], cost = cost.all[best])
    }
  )

  res$orders <- demand / res$q
  res$cycle <- 1 / res$orders
  return(res[c("q", "orders", "cycle", "cost")])
}

#' Annual cost components over a range of order quantities
#'
#' @param q Order quantities to evaluate.
#' @param demand Demand per year.
#' @param ordering Fixed cost of placing one order.
#' @param holding Cost of holding one unit for one year.
#' @param unit.cost Purchase price of one unit.
#' @return A data frame with columns `q`, `ordering`, `holding`, `purchase`
#'   and `total`, and the cheapest quantity in the `best` attribute.
#' @export
total.cost.curve <- function(q, demand, ordering, holding, unit.cost = 0) {
  if (any(q <= 0)) {
    stop("`q` must be positive.", call. = FALSE)
  }

  # Ordering cost: number of orders times the fixed charge
  orders.per.year <- demand / q
  ordering.cost <- orders.per.year * ordering
  # Holding cost: half a lot on average
  average.stock <- q / 2
  holding.cost <- average.stock * holding
  # Purchase cost: does not depend on the lot size
  units.bought <- rep(demand, length(q))
  purchase.cost <- units.bought * unit.cost

  out <- data.frame(q = q, ordering = ordering.cost, holding = holding.cost,
    purchase = purchase.cost)
  out$total <- out$ordering + out$holding + out$purchase
  attr(out, "best") <- q[which.min(out$total)]
  return(out)
}

#' Sensitivity of the classic EOQ to its inputs
#'
#' @param demand Demand per year.
#' @param ordering Fixed cost of placing one order.
#' @param holding Cost of holding one unit for one year.
#' @param change Relative changes applied to one input at a time.
#' @return A data frame with columns `input`, `change`, `q` and `cost`, one
#'   row per input among `demand`, `ordering` and `holding` and change.
#' @export
eoq.sensitivity <- function(demand, ordering, holding, change = c(-0.2, 0, 0.2)) {
  eoq <- function(d, k, h) sqrt(2 * d * k / h)
  cost <- function(d, k, h) sqrt(2 * d * k * h)
  rows <- list()

  # Vary demand
  d <- demand * (1 + change)
  rows$demand <- data.frame(input = "demand", change = change,
    q = eoq(d, ordering, holding), cost = cost(d, ordering, holding))
  # Vary ordering
  k <- ordering * (1 + change)
  rows$ordering <- data.frame(input = "ordering", change = change,
    q = eoq(demand, k, holding), cost = cost(demand, k, holding))
  # Vary holding
  h <- holding * (1 + change)
  rows$holding <- data.frame(input = "holding", change = change,
    q = eoq(demand, ordering, h), cost = cost(demand, ordering, h))

  out <- do.call(rbind, unname(rows))
  base <- cost(demand, ordering, holding)
  out$cost.change <- out$cost / base - 1
  return(out)
}

#' Consolidate purchase orders
#'
#' @param orders A data frame with columns `supplier`, `date`, `lane`,
#'   `item.group`, `quantity` and `weight`.
#' @param by Consolidation key, one of `"supplier"`, `"week"`, `"lane"` or
#'   `"item.group"`.
#' @return A data frame with columns `key`, `orders`, `quantity` and
#'   `weight`, one row per consolidated shipment.
#' @export
consolidate.orders <- function(orders, by = c("supplier", "week", "lane", "item.group")) {
  by <- match.arg(by)

  key <- switch(by,
    supplier = {
      # One shipment per supplier
      k <- as.character(orders$supplier)
      ifelse(is.na(k), "unknown", k)
    },
    week = {
      # One shipment per ISO week of the order date
      k <- format(as.Date(orders$date), "%G-W%V")
      ifelse(is.na(k), "unknown", k)
    },
    lane = {
      # One shipment per transport lane
      k <- as.character(orders$lane)
      ifelse(is.na(k), "unknown", k)
    },
    item.group = {
      # One shipment per item group
      k <- as.character(orders$item.group)
      ifelse(is.na(k), "unknown", k)
    }
  )

  groups <- split(orders, key)
  out <- data.frame(
    key = names(groups),
    orders = vapply(groups, nrow, integer(1)),
    quantity = vapply(groups, function(g) sum(g$quantity), numeric(1)),
    weight = vapply(groups, function(g) sum(g$weight), numeric(1))
  )
  rownames(out) <- NULL
  return(out)
}

#' Freight charge of a shipment by transport mode
#'
#' @param weight Shipment weight in kg.
#' @param mode Transport mode, one of `"parcel"`, `"ltl"`, `"ftl"` or `"air"`.
#' @param rates A list of tariffs per mode, each with `base`, `per.kg` and
#'   `max.kg`.
#' @return The charge, with the mode and the billed weight as attributes.
#' @export
freight.charge <- function(weight, mode = c("parcel", "ltl", "ftl", "air"),
                           rates = list(
                             parcel = list(base = 6, per.kg = 1.2, max.kg = 30),
                             ltl = list(base = 45, per.kg = 0.18, max.kg = 2500),
                             ftl = list(base = 900, per.kg = 0, max.kg = 24000),
                             air = list(base = 60, per.kg = 4.5, max.kg = 5000)
                           )) {
  mode <- match.arg(mode)
  tariff <- rates[[mode]]
  if (weight > tariff$max.kg) {
    stop(sprintf("Weight %g kg exceeds the %s limit of %g kg.", weight, mode, tariff$max.kg),
      call. = FALSE)
  }

  billed <- switch(mode,
    parcel = {
      # Billed per started kg
      ceiling(weight)
    },
    ltl = {
      # Billed per started 100 kg with a 100 kg minimum
      max(100, ceiling(weight / 100) * 100)
    },
    ftl = {
      # Billed as one full truck regardless of weight
      tariff$max.kg
    },
    air = {
      # Billed per started half kg with a 1 kg minimum
      max(1, ceiling(weight * 2) / 2)
    }
  )

  charge <- tariff$base + billed * tariff$per.kg
  return(structure(charge, mode = mode, billed = billed))
}
