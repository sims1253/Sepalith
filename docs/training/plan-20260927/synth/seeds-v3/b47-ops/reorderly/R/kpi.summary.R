#' Key performance indicators of a simulated inventory
#'
#' @param sim A data frame with columns `demand`, `on.hand`, `backorder` and
#'   `order`, one row per period.
#' @param kpis Indicators to compute, any of `"fill.rate"`, `"cycle.service"`,
#'   `"avg.inventory"`, `"turns"`, `"stockout.periods"` and `"avg.backorder"`.
#' @return A named numeric vector with one element per entry of `kpis`.
#' @export
kpi.summary <- function(sim, kpis = c("fill.rate", "cycle.service", "avg.inventory",
                                      "turns", "stockout.periods", "avg.backorder")) {
  kpis <- match.arg(kpis, several.ok = TRUE)
  out <- stats::setNames(numeric(length(kpis)), kpis)
  new.backorders <- pmax(diff(c(0, sim$backorder)), 0)

  for (kpi in kpis) {
    out[[kpi]] <- switch(kpi,
      fill.rate = {
        # Share of demand served from stock without delay
        served <- sum(sim$demand) - sum(new.backorders)
        if (sum(sim$demand) == 0) 1 else served / sum(sim$demand)
      },
      cycle.service = {
        # Share of periods without any new backorder
        clean <- new.backorders == 0
        mean(clean)
      },
      avg.inventory = {
        # Average stock on hand at the end of a period
        stock <- sim$on.hand
        mean(stock)
      },
      turns = {
        # Demand divided by average stock on hand
        stock <- sim$on.hand
        if (mean(stock) == 0) Inf else sum(sim$demand) / mean(stock)
      },
      stockout.periods = {
        # Periods ending with zero stock on hand
        stock <- sim$on.hand
        sum(stock <= 0)
      },
      avg.backorder = {
        # Average backorder at the end of a period
        owed <- sim$backorder
        mean(owed)
      }
    )
  }
  return(out)
}

#' Compare indicators with their targets
#'
#' @param kpi A named numeric vector as returned by [kpi.summary()].
#' @param targets Named numeric targets for some or all indicators.
#' @param direction Named vector saying whether `"higher"` or `"lower"`
#'   values are better for each indicator.
#' @return A data frame with columns `kpi`, `value`, `target`, `direction`,
#'   `gap` (positive when better than target) and `met`.
#' @export
kpi.targets.check <- function(kpi, targets,
                              direction = c(fill.rate = "higher", cycle.service = "higher",
                                            avg.inventory = "lower", turns = "higher",
                                            stockout.periods = "lower",
                                            avg.backorder = "lower")) {
  shared <- intersect(names(kpi), names(targets))
  if (length(shared) == 0) {
    stop("No indicator in `kpi` has a target.", call. = FALSE)
  }
  out <- data.frame(kpi = shared, value = unname(kpi[shared]),
    target = unname(targets[shared]), direction = unname(direction[shared]),
    gap = NA_real_, met = NA)

  for (i in seq_len(nrow(out))) {
    if (out$direction[i] == "higher") {
      # Higher is better: gap is value above target
      out$gap[i] <- out$value[i] - out$target[i]
      out$met[i] <- out$value[i] >= out$target[i]
    } else {
      # Lower is better: gap is value below target
      out$gap[i] <- out$target[i] - out$value[i]
      out$met[i] <- out$value[i] <= out$target[i]
    }
  }
  return(out)
}

#' Text table of indicators for reports
#'
#' @param kpi A named numeric vector as returned by [kpi.summary()].
#' @param digits Decimal places for unit and ratio values.
#' @return A data frame with columns `kpi`, `label` and `text`.
#' @export
render.kpi.table <- function(kpi, digits = 1) {
  labels <- c(fill.rate = "Fill rate", cycle.service = "Cycle service level",
    avg.inventory = "Average inventory", turns = "Inventory turns",
    stockout.periods = "Stock-out periods", avg.backorder = "Average backorder")
  text <- character(length(kpi))
  names(text) <- names(kpi)

  # fill.rate as a percentage
  if ("fill.rate" %in% names(kpi)) {
    text[["fill.rate"]] <- sprintf("%.*f%%", digits, 100 * kpi[["fill.rate"]])
  }
  # cycle.service as a percentage
  if ("cycle.service" %in% names(kpi)) {
    text[["cycle.service"]] <- sprintf("%.*f%%", digits, 100 * kpi[["cycle.service"]])
  }
  # avg.inventory in units
  if ("avg.inventory" %in% names(kpi)) {
    text[["avg.inventory"]] <- sprintf("%.*f units", digits, kpi[["avg.inventory"]])
  }
  # turns as a multiple
  if ("turns" %in% names(kpi)) {
    text[["turns"]] <- sprintf("%.*fx", digits, kpi[["turns"]])
  }
  # stockout.periods as a whole count
  if ("stockout.periods" %in% names(kpi)) {
    text[["stockout.periods"]] <- sprintf("%d", as.integer(kpi[["stockout.periods"]]))
  }
  # avg.backorder in units
  if ("avg.backorder" %in% names(kpi)) {
    text[["avg.backorder"]] <- sprintf("%.*f units", digits, kpi[["avg.backorder"]])
  }

  return(data.frame(kpi = names(kpi), label = unname(labels[names(kpi)]),
    text = unname(text)))
}

#' Change of indicators between two periods
#'
#' @param before,after Named numeric vectors as returned by [kpi.summary()]
#'   for the earlier and the later period.
#' @return A data frame with columns `kpi`, `before`, `after`, `change` and
#'   `pct.change`; indicators present in only one input are dropped with a
#'   warning.
#' @export
kpi.period.compare <- function(before, after) {
  # Indicators missing from the later period
  only.before <- setdiff(names(before), names(after))
  if (length(only.before) > 0) {
    warning(sprintf("Dropped from `%s`: %s", "before", paste(only.before, collapse = ", ")),
      call. = FALSE)
  }
  # Indicators missing from the earlier period
  only.after <- setdiff(names(after), names(before))
  if (length(only.after) > 0) {
    warning(sprintf("Dropped from `%s`: %s", "after", paste(only.after, collapse = ", ")),
      call. = FALSE)
  }

  shared <- intersect(names(before), names(after))
  change <- after[shared] - before[shared]
  pct.change <- ifelse(before[shared] == 0, NA_real_, 100 * change / abs(before[shared]))
  return(data.frame(kpi = shared, before = unname(before[shared]),
    after = unname(after[shared]), change = unname(change),
    pct.change = unname(pct.change)))
}

#' Weighted scorecard across items
#'
#' @param kpi.table A data frame with an `item` column and one column per
#'   indicator `fill.rate`, `turns`, `avg.inventory` and `stockout.periods`.
#' @param weights Weight of each indicator in the score.
#' @return `kpi.table` with a `score` column between 0 and 1 (higher is
#'   better), sorted by decreasing score.
#' @export
kpi.scorecard <- function(kpi.table,
                          weights = c(fill.rate = 0.4, turns = 0.2, avg.inventory = 0.2,
                                      stockout.periods = 0.2)) {
  unit.scale <- function(x) {
    spread <- max(x) - min(x)
    if (spread == 0) return(rep(1, length(x)))
    return((x - min(x)) / spread)
  }
  score <- numeric(nrow(kpi.table))

  # fill.rate: higher is better
  part <- unit.scale(kpi.table$fill.rate)
  score <- score + weights[["fill.rate"]] * part
  # turns: higher is better
  part <- unit.scale(kpi.table$turns)
  score <- score + weights[["turns"]] * part
  # avg.inventory: lower is better
  part <- 1 - unit.scale(kpi.table$avg.inventory)
  score <- score + weights[["avg.inventory"]] * part
  # stockout.periods: lower is better
  part <- 1 - unit.scale(kpi.table$stockout.periods)
  score <- score + weights[["stockout.periods"]] * part

  kpi.table$score <- score / sum(weights)
  kpi.table <- kpi.table[order(-kpi.table$score, kpi.table$item), , drop = FALSE]
  rownames(kpi.table) <- NULL
  return(kpi.table)
}
