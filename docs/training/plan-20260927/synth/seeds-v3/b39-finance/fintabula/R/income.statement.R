#' Sum trial balance lines by account prefix
#'
#' @param tb A trial balance data frame with columns `code` and `balance`,
#'   balances in the normal direction of each account.
#' @param prefixes Character vector of account code prefixes.
#' @return The summed balance of all accounts whose code starts with one of
#'   `prefixes`.
#' @keywords internal
line.amount <- function(tb, prefixes) {
  hit <- Reduce(`|`, lapply(prefixes, function(p) startsWith(tb$code, p)), FALSE)
  sum(tb$balance[hit])
}

#' Build an income statement
#'
#' The statement has one line per item, in this order: `revenue`,
#' `cost.of.sales`, `gross.profit`, `operating.expenses`,
#' `operating.income`, `interest.expense`, `pretax.income`, `tax` and
#' `net.income`.
#'
#' @param tb A trial balance data frame with columns `code` and `balance`,
#'   balances in the normal direction of each account.
#' @param map A named list of account code prefixes with elements
#'   `revenue`, `cost.of.sales`, `operating.expenses`, `interest.expense`
#'   and `tax`.
#' @return A named numeric vector of class `fin.statement`.
#' @export
build.income.statement <- function(tb, map) {
  stmt <- c()

  # Revenue
  stmt["revenue"] <- line.amount(tb, map$revenue)
  # Cost of sales
  stmt["cost.of.sales"] <- line.amount(tb, map$cost.of.sales)
  # Gross profit
  stmt["gross.profit"] <- stmt[["revenue"]] - stmt[["cost.of.sales"]]
  # Operating expenses
  stmt["operating.expenses"] <- line.amount(tb, map$operating.expenses)
  # Operating income
  stmt["operating.income"] <- stmt[["gross.profit"]] - stmt[["operating.expenses"]]
  # Interest expense
  stmt["interest.expense"] <- line.amount(tb, map$interest.expense)
  # Pretax income
  stmt["pretax.income"] <- stmt[["operating.income"]] - stmt[["interest.expense"]]
  # Tax
  stmt["tax"] <- line.amount(tb, map$tax)
  # Net income
  stmt["net.income"] <- stmt[["pretax.income"]] - stmt[["tax"]]

  structure(stmt, class = c("fin.statement", "numeric"))
}

#' Margin ratios from an income statement
#'
#' @param stmt An income statement as returned by
#'   [build.income.statement()].
#' @param digits Number of decimal places kept in the ratios.
#' @return A named numeric vector with elements `gross.margin`,
#'   `operating.margin`, `pretax.margin`, `net.margin` and `tax.rate`; a
#'   ratio is `NA` when its denominator is zero.
#' @export
margin.ratios <- function(stmt, digits = 4L) {
  ratios <- c()

  # Gross margin
  den <- stmt[["revenue"]]
  ratios["gross.margin"] <- if (den == 0) NA_real_ else stmt[["gross.profit"]] / den
  # Operating margin
  den <- stmt[["revenue"]]
  ratios["operating.margin"] <- if (den == 0) NA_real_ else stmt[["operating.income"]] / den
  # Pretax margin
  den <- stmt[["revenue"]]
  ratios["pretax.margin"] <- if (den == 0) NA_real_ else stmt[["pretax.income"]] / den
  # Net margin
  den <- stmt[["revenue"]]
  ratios["net.margin"] <- if (den == 0) NA_real_ else stmt[["net.income"]] / den
  # Tax rate
  den <- stmt[["pretax.income"]]
  ratios["tax.rate"] <- if (den == 0) NA_real_ else stmt[["tax"]] / den

  round(ratios, digits)
}

#' Compare two statements line by line
#'
#' @param current,prior Statements with the same line names, as returned by
#'   [build.income.statement()].
#' @param threshold Absolute percentage change above which a line is
#'   flagged.
#' @return A data frame with columns `item`, `current`, `prior`, `change`,
#'   `pct.change` and `flag`, one row per line. `pct.change` is `NA` when
#'   the prior amount is zero; such lines are flagged when they changed.
#' @export
statement.variance <- function(current, prior, threshold = 10) {
  if (!identical(names(current), names(prior))) {
    stop("`current` and `prior` must have the same lines.", call. = FALSE)
  }
  current <- unclass(current)
  prior <- unclass(prior)
  change <- current - prior
  pct.change <- ifelse(prior == 0, NA_real_, 100 * change / abs(prior))

  # New lines have no percentage, so flag them on any change
  flag <- ifelse(is.na(pct.change), change != 0, abs(pct.change) > threshold)
  data.frame(
    item = names(current),
    current = unname(current),
    prior = unname(prior),
    change = unname(change),
    pct.change = unname(round(pct.change, 1)),
    flag = unname(flag),
    stringsAsFactors = FALSE
  )
}

#' Format a financial statement for printing
#'
#' Subtotal lines (`gross.profit`, `operating.income`, `pretax.income`,
#' `net.income` and any line whose name starts with `total.`) are preceded
#' by a rule; negative amounts are shown in parentheses.
#'
#' @param x A statement of class `fin.statement`.
#' @param width Width of the label column.
#' @param title Optional title, printed centred above the statement.
#' @param ... Unused, for compatibility with [format()].
#' @return A character vector of formatted lines.
#' @export
format.fin.statement <- function(x, width = 24L, title = NULL, ...) {
  subtotals <- c("gross.profit", "operating.income", "pretax.income", "net.income")
  total.width <- width + 16L
  values <- unclass(x)

  # Labels: dots to spaces, first letter upper case
  labels <- gsub(".", " ", names(values), fixed = TRUE)
  labels <- paste0(toupper(substring(labels, 1, 1)), substring(labels, 2))
  # Amounts: thousands separators, negatives in parentheses
  amounts <- formatC(abs(values), format = "f", digits = 2, big.mark = ",")
  amounts <- ifelse(values < 0, sprintf("(%s)", amounts), amounts)

  out <- character()
  if (!is.null(title)) {
    margin <- max(0L, (total.width - nchar(title)) %/% 2L)
    out <- c(out, paste0(strrep(" ", margin), title), strrep("=", total.width))
  }
  for (i in seq_along(values)) {
    if (names(values)[i] %in% subtotals || startsWith(names(values)[i], "total.")) {
      out <- c(out, strrep("-", total.width))
    }
    out <- c(out, sprintf("%-*s%16s", width, labels[i], amounts[i]))
  }
  out
}

#' @rdname format.fin.statement
#' @export
print.fin.statement <- function(x, ...) {
  cat(format(x, ...), sep = "\n")
  invisible(x)
}
