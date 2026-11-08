#' Create a trade blotter
#'
#' @param date Trade dates, coercible with [as.Date()].
#' @param side `"buy"` or `"sell"` for each trade.
#' @param qty Positive quantities.
#' @param price Positive execution prices.
#' @param fee Non-negative fees; recycled.
#' @return A data frame with columns `date`, `side`, `qty`, `price` and
#'   `fee`, sorted by `date`.
#' @export
new.trades <- function(date, side, qty, price, fee = 0) {
  # Check `date`
  date <- as.Date(date)
  if (anyNA(date)) {
    stop(sprintf("`%s` has invalid values.", "date"), call. = FALSE)
  }
  # Check `side`
  if (anyNA(side) || !all(side %in% c("buy", "sell"))) {
    stop(sprintf("`%s` has invalid values.", "side"), call. = FALSE)
  }
  # Check `qty`
  if (!is.numeric(qty) || anyNA(qty) || any(qty <= 0)) {
    stop(sprintf("`%s` has invalid values.", "qty"), call. = FALSE)
  }
  # Check `price`
  if (!is.numeric(price) || anyNA(price) || any(price <= 0)) {
    stop(sprintf("`%s` has invalid values.", "price"), call. = FALSE)
  }
  # Check `fee`
  if (!is.numeric(fee) || anyNA(fee) || any(fee < 0)) {
    stop(sprintf("`%s` has invalid values.", "fee"), call. = FALSE)
  }

  out <- data.frame(date = date, side = side, qty = qty, price = price, fee = fee,
    stringsAsFactors = FALSE)
  out <- out[order(out$date), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Cash flows of a trade blotter
#'
#' Buys pay the notional plus the fee; sells receive the notional less the
#' fee.
#'
#' @param trades A blotter as returned by [new.trades()].
#' @return A numeric vector of signed cash flows, one per trade.
#' @export
trade.cash.flows <- function(trades) {
  notional <- trades$qty * trades$price
  ifelse(trades$side == "buy", -(notional + trades$fee), notional - trades$fee)
}

#' Apply one trade to an average-cost position
#'
#' A buy adds to the position at its own price, moving the average cost; a
#' sell removes units at the current average cost and realises the
#' difference to its price. Fees are added to the cost of buys and
#' deducted from the proceeds of sells.
#'
#' @param position A list with elements `qty`, `avg.cost`, `realized`,
#'   `bought`, `sold` and `fees`.
#' @param trade A single row of a blotter as returned by [new.trades()].
#' @return The updated `position` list.
#' @export
apply.trade <- function(position, trade) {
  if (trade$side == "buy") {
    # Buy: grow the position and blend the average cost
    cost <- trade$qty * trade$price + trade$fee
    total <- position$qty * position$avg.cost + cost
    position$qty <- position$qty + trade$qty
    position$avg.cost <- total / position$qty
    position$bought <- position$bought + trade$qty
  } else {
    # Sell: shrink the position and realise against the average cost
    if (trade$qty > position$qty) {
      stop("Cannot sell more than the position holds.", call. = FALSE)
    }
    proceeds <- trade$qty * trade$price - trade$fee
    position$realized <- position$realized + proceeds - trade$qty * position$avg.cost
    position$qty <- position$qty - trade$qty
    if (position$qty == 0) position$avg.cost <- 0
    position$sold <- position$sold + trade$qty
  }
  position$fees <- position$fees + trade$fee
  position
}

#' Realised profit and loss with first-in first-out lots
#'
#' @param trades A blotter as returned by [new.trades()].
#' @return A list with elements `realized` (total realised profit),
#'   `lots` (a data frame of open lots with columns `qty` and `price`) and
#'   `matched` (number of lot matches made).
#' @export
fifo.realized <- function(trades) {
  lots <- data.frame(qty = numeric(), price = numeric())
  realized <- 0
  matched <- 0L
  for (i in seq_len(nrow(trades))) {
    t <- trades[i, ]
    if (t$side == "buy") {
      lots <- rbind(lots, data.frame(qty = t$qty, price = t$price + t$fee / t$qty))
      next
    }
    remaining <- t$qty
    unit.proceeds <- t$price - t$fee / t$qty
    while (remaining > 0) {
      if (nrow(lots) == 0L) stop("Cannot sell more than the position holds.", call. = FALSE)
      take <- min(remaining, lots$qty[1L])
      realized <- realized + take * (unit.proceeds - lots$price[1L])
      lots$qty[1L] <- lots$qty[1L] - take
      remaining <- remaining - take
      matched <- matched + 1L
      if (lots$qty[1L] == 0) lots <- lots[-1L, , drop = FALSE]
    }
  }
  rownames(lots) <- NULL
  list(realized = realized, lots = lots, matched = matched)
}

#' Fill an order against an order book
#'
#' A buy walks the asks from the lowest price upwards; a sell walks the
#' bids from the highest price downwards. Unfilled quantity is reported
#' rather than raising an error.
#'
#' @param side `"buy"` or `"sell"`.
#' @param qty Quantity to fill.
#' @param book A list with elements `bids` and `asks`, each a data frame
#'   with columns `price` and `qty`.
#' @return A list with elements `filled`, `unfilled`, `avg.price` and
#'   `levels` (number of price levels touched).
#' @export
fill.order <- function(side = c("buy", "sell"), qty, book) {
  side <- match.arg(side)
  filled <- 0
  notional <- 0
  levels <- 0L
  if (side == "buy") {
    asks <- book$asks[order(book$asks$price), , drop = FALSE]
    for (i in seq_len(nrow(asks))) {
      if (filled >= qty) break
      take <- min(qty - filled, asks$qty[i])
      filled <- filled + take
      notional <- notional + take * asks$price[i]
      levels <- levels + 1L
    }
  } else {
    bids <- book$bids[order(book$bids$price, decreasing = TRUE), , drop = FALSE]
    for (i in seq_len(nrow(bids))) {
      if (filled >= qty) break
      take <- min(qty - filled, bids$qty[i])
      filled <- filled + take
      notional <- notional + take * bids$price[i]
      levels <- levels + 1L
    }
  }
  list(filled = filled, unfilled = qty - filled,
    avg.price = if (filled > 0) notional / filled else NA_real_, levels = levels)
}

#' Brokerage fees under different fee models
#'
#' Rates come from the table `rates`: `flat` 5 per trade, `per.share` 0.01
#' per unit, `percent` 0.1 percent of notional. `tiered` charges the
#' `percent` rate on notional up to 10000 and half of it above.
#'
#' @param trades A blotter as returned by [new.trades()].
#' @param model One of `"flat"`, `"per.share"`, `"percent"` or `"tiered"`.
#' @param minimum Smallest fee charged per trade.
#' @return `trades` with the `fee` column replaced.
#' @export
apply.fee.model <- function(trades, model = c("flat", "per.share", "percent", "tiered"),
                            minimum = 1) {
  model <- match.arg(model)
  rates <- c(flat = 5, per.share = 0.01, percent = 0.001, tier.limit = 10000)
  notional <- trades$qty * trades$price
  fee <- switch(model,
    flat = {
      rep(rates[["flat"]], nrow(trades))
    },
    per.share = {
      trades$qty * rates[["per.share"]]
    },
    percent = {
      notional * rates[["percent"]]
    },
    tiered = {
      lower <- pmin(notional, rates[["tier.limit"]])
      upper <- pmax(notional - rates[["tier.limit"]], 0)
      lower * rates[["percent"]] + upper * rates[["percent"]] / 2
    }
  )
  trades$fee <- round(pmax(fee, minimum), 2)
  trades
}
