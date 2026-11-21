#' Build a balance sheet
#'
#' The sheet has one line per item, in this order: `cash`, `receivables`,
#' `inventory`, `total.current.assets`, `fixed.assets`, `total.assets`,
#' `payables`, `short.term.debt`, `total.current.liabilities`,
#' `long.term.debt`, `total.liabilities`, `equity` and
#' `total.liabilities.equity`.
#'
#' @param tb A trial balance data frame with columns `code` and `balance`,
#'   balances in the normal direction of each account.
#' @param map A named list of account code prefixes with elements `cash`,
#'   `receivables`, `inventory`, `fixed.assets`, `payables`,
#'   `short.term.debt`, `long.term.debt` and `equity`.
#' @return A named numeric vector of class `fin.statement`.
#' @export
build.balance.sheet <- function(tb, map) {
  bs <- c()

  # Current assets
  bs["cash"] <- line.amount(tb, map$cash)
  bs["receivables"] <- line.amount(tb, map$receivables)
  bs["inventory"] <- line.amount(tb, map$inventory)
  bs["total.current.assets"] <- bs[["cash"]] + bs[["receivables"]] + bs[["inventory"]]
  # Non-current assets
  bs["fixed.assets"] <- line.amount(tb, map$fixed.assets)
  bs["total.assets"] <- bs[["total.current.assets"]] + bs[["fixed.assets"]]

  # Current liabilities
  bs["payables"] <- line.amount(tb, map$payables)
  bs["short.term.debt"] <- line.amount(tb, map$short.term.debt)
  bs["total.current.liabilities"] <- bs[["payables"]] + bs[["short.term.debt"]]
  # Non-current liabilities
  bs["long.term.debt"] <- line.amount(tb, map$long.term.debt)
  bs["total.liabilities"] <- bs[["total.current.liabilities"]] + bs[["long.term.debt"]]
  # Equity
  bs["equity"] <- line.amount(tb, map$equity)
  bs["total.liabilities.equity"] <- bs[["total.liabilities"]] + bs[["equity"]]

  structure(bs, class = c("fin.statement", "numeric"))
}

#' Check the subtotals of a balance sheet
#'
#' Recomputes `total.current.assets`, `total.assets`,
#' `total.current.liabilities`, `total.liabilities` and
#' `total.liabilities.equity` from their components, and checks that
#' `total.assets` equals `total.liabilities.equity`.
#'
#' @param bs A balance sheet as returned by [build.balance.sheet()].
#' @param tolerance Largest absolute difference treated as zero.
#' @return A character vector of problems, empty when the sheet is
#'   consistent.
#' @export
check.balance.sheet <- function(bs, tolerance = 0.005) {
  problems <- character()
  differs <- function(a, b) abs(a - b) > tolerance

  # Check `total.current.assets`
  expected <- bs[["cash"]] + bs[["receivables"]] + bs[["inventory"]]
  if (differs(bs[["total.current.assets"]], expected)) {
    problems <- c(problems, sprintf("`%s` should be %.2f", "total.current.assets", expected))
  }
  # Check `total.assets`
  expected <- bs[["total.current.assets"]] + bs[["fixed.assets"]]
  if (differs(bs[["total.assets"]], expected)) {
    problems <- c(problems, sprintf("`%s` should be %.2f", "total.assets", expected))
  }
  # Check `total.current.liabilities`
  expected <- bs[["payables"]] + bs[["short.term.debt"]]
  if (differs(bs[["total.current.liabilities"]], expected)) {
    problems <- c(problems, sprintf("`%s` should be %.2f", "total.current.liabilities",
      expected))
  }
  # Check `total.liabilities`
  expected <- bs[["total.current.liabilities"]] + bs[["long.term.debt"]]
  if (differs(bs[["total.liabilities"]], expected)) {
    problems <- c(problems, sprintf("`%s` should be %.2f", "total.liabilities", expected))
  }
  # Check `total.liabilities.equity`
  expected <- bs[["total.liabilities"]] + bs[["equity"]]
  if (differs(bs[["total.liabilities.equity"]], expected)) {
    problems <- c(problems, sprintf("`%s` should be %.2f", "total.liabilities.equity",
      expected))
  }

  if (differs(bs[["total.assets"]], bs[["total.liabilities.equity"]])) {
    problems <- c(problems, sprintf("assets and liabilities plus equity differ by %.2f",
      bs[["total.assets"]] - bs[["total.liabilities.equity"]]))
  }
  problems
}

#' Liquidity and leverage ratios
#'
#' @param bs A balance sheet as returned by [build.balance.sheet()].
#' @param digits Number of decimal places kept in the ratios.
#' @return A named numeric vector with elements `current.ratio`,
#'   `quick.ratio`, `cash.ratio`, `debt.to.equity` and `equity.ratio`; a
#'   ratio is `NA` when its denominator is zero.
#' @export
liquidity.ratios <- function(bs, digits = 4L) {
  ratios <- c()
  debt <- bs[["short.term.debt"]] + bs[["long.term.debt"]]
  quick <- bs[["cash"]] + bs[["receivables"]]

  # Current ratio
  den <- bs[["total.current.liabilities"]]
  ratios["current.ratio"] <- if (den == 0) NA_real_ else bs[["total.current.assets"]] / den
  # Quick ratio
  den <- bs[["total.current.liabilities"]]
  ratios["quick.ratio"] <- if (den == 0) NA_real_ else quick / den
  # Cash ratio
  den <- bs[["total.current.liabilities"]]
  ratios["cash.ratio"] <- if (den == 0) NA_real_ else bs[["cash"]] / den
  # Debt to equity
  den <- bs[["equity"]]
  ratios["debt.to.equity"] <- if (den == 0) NA_real_ else debt / den
  # Equity ratio
  den <- bs[["total.assets"]]
  ratios["equity.ratio"] <- if (den == 0) NA_real_ else bs[["equity"]] / den

  round(ratios, digits)
}

#' Cash effect of working capital movements
#'
#' An increase in an operating asset ties up cash, so it enters with a
#' negative sign; an increase in an operating liability releases cash, so
#' it enters with a positive sign.
#'
#' @param current,prior Balance sheets as returned by
#'   [build.balance.sheet()].
#' @param assets Names of the operating asset lines.
#' @param liabilities Names of the operating liability lines.
#' @return A data frame with columns `item`, `type`, `prior`, `current`,
#'   `change` and `effect`, one row per line in `assets` and
#'   `liabilities`, with attribute `total` (the summed `effect`).
#' @export
working.capital.effect <- function(current, prior, assets = c("receivables", "inventory"),
                                   liabilities = "payables") {
  rows <- list()
  # Operating assets: increases use cash
  for (item in assets) {
    change <- current[[item]] - prior[[item]]
    rows[[item]] <- data.frame(item = item, type = "asset", prior = prior[[item]],
      current = current[[item]], change = change, effect = -change)
  }
  # Operating liabilities: increases provide cash
  for (item in liabilities) {
    change <- current[[item]] - prior[[item]]
    rows[[item]] <- data.frame(item = item, type = "liability", prior = prior[[item]],
      current = current[[item]], change = change, effect = change)
  }

  out <- do.call(rbind, unname(rows))
  rownames(out) <- NULL
  attr(out, "total") <- sum(out$effect)
  out
}

#' Classify trial balance accounts into statement lines
#'
#' @param tb A trial balance data frame with columns `code` and `balance`,
#'   balances in the normal direction of each account.
#' @param map A named list of account code prefixes, one element per line.
#' @return `tb` with an added column `line`; accounts that match no prefix
#'   get `NA`, accounts that match several prefixes keep the first match
#'   and trigger a warning.
#' @export
classify.accounts <- function(tb, map) {
  line <- rep(NA_character_, nrow(tb))
  clash <- character()
  for (name in names(map)) {
    hit <- Reduce(`|`, lapply(map[[name]], function(p) startsWith(tb$code, p)), FALSE)
    clash <- c(clash, tb$code[hit & !is.na(line)])
    line[hit & is.na(line)] <- name
  }
  if (length(clash) > 0L) {
    warning("Accounts matched by several lines: ", paste(unique(clash), collapse = ", "),
      call. = FALSE)
  }
  tb$line <- line
  tb
}
