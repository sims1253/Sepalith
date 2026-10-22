#' Tabulate events and numbers at risk
#'
#' @param time Numeric vector of follow-up times.
#' @param status Event indicator, `1` for an event and `0` for censoring.
#' @return A data frame with columns `time`, `n.risk`, `n.event` and
#'   `n.censor`, one row per distinct time.
#' @keywords internal
risk.table = function(time, status) {
  if (length(time) != length(status)) {
    stop("`time` and `status` must have the same length.", call. = FALSE)
  }
  if (!all(status %in% c(0, 1))) {
    stop("`status` must be coded 0/1.", call. = FALSE)
  }
  ut = sort(unique(time))
  n.event = vapply(ut, function(t) sum(time == t & status == 1), numeric(1))
  n.censor = vapply(ut, function(t) sum(time == t & status == 0), numeric(1))
  n.risk = vapply(ut, function(t) sum(time >= t), numeric(1))
  data.frame(time = ut, n.risk = n.risk, n.event = n.event, n.censor = n.censor)
}

#' Kaplan-Meier estimate of the survival function
#'
#' @param time Numeric vector of follow-up times.
#' @param status Event indicator, `1` for an event and `0` for censoring.
#' @param conf.level Confidence level for the log-log pointwise interval.
#' @return An object of class `nq.km`.
#' @family survival estimators
#' @export
kaplan.meier = function(time, status, conf.level = 0.95) {
  tab = risk.table(time, status)
  tab = tab[tab$n.event > 0, , drop = FALSE]
  tab$surv = cumprod(1 - tab$n.event / tab$n.risk)
  tab$var.log = cumsum(tab$n.event / (tab$n.risk * (tab$n.risk - tab$n.event)))
  z = stats::qnorm(1 - (1 - conf.level) / 2)
  se.ll = sqrt(tab$var.log) / abs(log(tab$surv))
  tab$lower = tab$surv^exp(z * se.ll)
  tab$upper = tab$surv^exp(-z * se.ll)
  structure(list(table = tab, n = length(time), events = sum(status),
    conf.level = conf.level), class = "nq.km")
}

#' Nelson-Aalen estimate of the cumulative hazard
#'
#' @param time Numeric vector of follow-up times.
#' @param status Event indicator, `1` for an event and `0` for censoring.
#' @param conf.level Confidence level for the log-scale pointwise interval.
#' @return An object of class `nq.na`.
#' @family survival estimators
#' @export
nelson.aalen = function(time, status, conf.level = 0.95) {
  tab = risk.table(time, status)
  tab = tab[tab$n.event > 0, , drop = FALSE]
  tab$cumhaz = cumsum(tab$n.event / tab$n.risk)
  tab$var = cumsum(tab$n.event / tab$n.risk^2)
  z = stats::qnorm(1 - (1 - conf.level) / 2)
  se.log = sqrt(tab$var) / tab$cumhaz
  tab$lower = tab$cumhaz * exp(-z * se.log)
  tab$upper = tab$cumhaz * exp(z * se.log)
  structure(list(table = tab, n = length(time), events = sum(status),
    conf.level = conf.level), class = "nq.na")
}

#' Quantile of a Kaplan-Meier curve
#'
#' Returns the smallest event time at which the estimated survival drops to
#' `1 - prob` or below.
#'
#' @param km An object of class `nq.km`.
#' @param prob Probability of the quantile, e.g. `0.5` for the median.
#' @return The estimated quantile, or `NA` if the curve never reaches it.
#' @export
km.quantile = function(km, prob = 0.5) {
  stopifnot(inherits(km, "nq.km"), prob > 0, prob < 1)
  tab = km$table
  hit = which(tab$surv <= 1 - prob + 1e-12)
  if (length(hit) == 0L) {
    return(NA_real_)
  }
  tab$time[hit[1L]]
}

#' Two-sample log-rank test
#'
#' @param time Numeric vector of follow-up times.
#' @param status Event indicator, `1` for an event and `0` for censoring.
#' @param group Vector of group labels with exactly two levels.
#' @return A list with the chi-squared statistic, observed and expected
#'   counts for the first group, and the p-value.
#' @export
logrank.test = function(time, status, group) {
  group = as.factor(group)
  if (nlevels(group) != 2L) {
    stop("`group` must have exactly two levels.", call. = FALSE)
  }
  g1 = group == levels(group)[1L]
  ut = sort(unique(time[status == 1]))
  parts = vapply(ut, function(t) {
    at.risk = time >= t
    n = sum(at.risk)
    n1 = sum(at.risk & g1)
    d = sum(time == t & status == 1)
    d1 = sum(time == t & status == 1 & g1)
    v = if (n > 1) d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1) else 0
    c(observed = d1, expected = d * n1 / n, var = v)
  }, numeric(3))
  totals = rowSums(parts)
  stat = (totals[["observed"]] - totals[["expected"]])^2 / totals[["var"]]
  list(statistic = stat, observed = totals[["observed"]], expected = totals[["expected"]],
    p.value = stats::pchisq(stat, df = 1, lower.tail = FALSE))
}

#' Print a Kaplan-Meier estimate
#'
#' @param x An object of class `nq.km`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family survival print methods
#' @export
print.nq.km = function(x, digits = 3, ...) {
  cat("Kaplan-Meier estimate: n =", x$n, ", events =", x$events, "\n")
  cat("Median survival:", format(km.quantile(x, 0.5), digits = digits), "\n\n")
  cols = c("time", "n.risk", "n.event", "surv", "lower", "upper")
  print(format(x$table[cols], digits = digits), row.names = FALSE)
  invisible(x)
}

#' Print a Nelson-Aalen estimate
#'
#' @param x An object of class `nq.na`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family survival print methods
#' @export
print.nq.na = function(x, digits = 3, ...) {
  cat("Nelson-Aalen estimate: n =", x$n, ", events =", x$events, "\n")
  cat("Final cumulative hazard:", format(utils::tail(x$table$cumhaz, 1), digits = digits),
    "\n\n")
  cols = c("time", "n.risk", "n.event", "cumhaz", "lower", "upper")
  print(format(x$table[cols], digits = digits), row.names = FALSE)
  invisible(x)
}
