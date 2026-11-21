#' Account class from an account code
#'
#' The first digit of the code decides the class: `1` asset, `2` liability,
#' `3` equity, `4` revenue, `5` and `6` expense.
#'
#' @param code A character vector of account codes.
#' @return A character vector of classes, `NA` for unknown leading digits.
#' @keywords internal
account_class <- function(code) {
  classes <- c("1" = "asset", "2" = "liability", "3" = "equity", "4" = "revenue",
    "5" = "expense", "6" = "expense")
  unname(classes[substr(code, 1L, 1L)])
}

#' Normal balance side of an account class
#'
#' @param class One of `"asset"`, `"liability"`, `"equity"`, `"revenue"` or
#'   `"expense"`.
#' @return A list with elements `side` (`"debit"` or `"credit"`), `sign`
#'   (`1` when debits increase the account, `-1` otherwise), `statement`
#'   (`"balance"` or `"income"`) and `temporary` (`TRUE` for accounts that
#'   are closed at period end).
#' @keywords internal
normal_side <- function(class = c("asset", "liability", "equity", "revenue", "expense")) {
  class <- match.arg(class)
  switch(class,
    asset = list(
      side = "debit",
      sign = 1,
      statement = "balance",
      temporary = FALSE
    ),
    liability = list(
      side = "credit",
      sign = -1,
      statement = "balance",
      temporary = FALSE
    ),
    equity = list(
      side = "credit",
      sign = -1,
      statement = "balance",
      temporary = FALSE
    ),
    revenue = list(
      side = "credit",
      sign = -1,
      statement = "income",
      temporary = TRUE
    ),
    expense = list(
      side = "debit",
      sign = 1,
      statement = "income",
      temporary = TRUE
    )
  )
}

#' Create a chart of accounts
#'
#' @param code Account codes, four digits each.
#' @param name Account names.
#' @param class Account classes; when `NULL` they are derived from `code`
#'   with `account_class()`.
#' @return A data frame with columns `code`, `name`, `class` and `side`,
#'   sorted by `code`.
#' @export
new_chart <- function(code, name, class = NULL) {
  code <- as.character(code)
  n <- length(code)

  # Check `code`
  if (anyNA(code) || any(!grepl("^[0-9]{4}$", code)) || anyDuplicated(code)) {
    stop(sprintf("`%s` has invalid or duplicated entries.", "code"), call. = FALSE)
  }
  # Check `name`
  name <- as.character(name)
  if (length(name) != n || anyNA(name) || any(!nzchar(name))) {
    stop(sprintf("`%s` has invalid or duplicated entries.", "name"), call. = FALSE)
  }
  # Check `class`
  if (is.null(class)) class <- account_class(code)
  known <- c("asset", "liability", "equity", "revenue", "expense")
  if (length(class) != n || anyNA(class) || !all(class %in% known)) {
    stop(sprintf("`%s` has invalid or duplicated entries.", "class"), call. = FALSE)
  }

  side <- vapply(class, function(cl) normal_side(cl)$side, character(1), USE.NAMES = FALSE)
  chart <- data.frame(code = code, name = name, class = class, side = side,
    stringsAsFactors = FALSE)
  chart[order(chart$code), , drop = FALSE]
}

#' Summarise a chart of accounts by class
#'
#' @param chart A chart of accounts as returned by [new_chart()].
#' @return A data frame with one row per class (`asset`, `liability`,
#'   `equity`, `revenue`, `expense`) and columns `class`, `n`, `first` and
#'   `last` (lowest and highest code, `NA` for empty classes).
#' @export
chart_summary <- function(chart) {
  rows <- list()

  # Class `asset`
  codes <- sort(chart$code[chart$class == "asset"])
  rows$asset <- data.frame(class = "asset", n = length(codes), first = codes[1L],
    last = rev(codes)[1L])
  # Class `liability`
  codes <- sort(chart$code[chart$class == "liability"])
  rows$liability <- data.frame(class = "liability", n = length(codes), first = codes[1L],
    last = rev(codes)[1L])
  # Class `equity`
  codes <- sort(chart$code[chart$class == "equity"])
  rows$equity <- data.frame(class = "equity", n = length(codes), first = codes[1L],
    last = rev(codes)[1L])
  # Class `revenue`
  codes <- sort(chart$code[chart$class == "revenue"])
  rows$revenue <- data.frame(class = "revenue", n = length(codes), first = codes[1L],
    last = rev(codes)[1L])
  # Class `expense`
  codes <- sort(chart$code[chart$class == "expense"])
  rows$expense <- data.frame(class = "expense", n = length(codes), first = codes[1L],
    last = rev(codes)[1L])

  out <- do.call(rbind, unname(rows))
  rownames(out) <- NULL
  out
}

#' Trial balance from journal lines
#'
#' @param entries A data frame of journal lines with columns `account`,
#'   `debit` and `credit`.
#' @param chart A chart of accounts as returned by [new_chart()].
#' @return A data frame with columns `code`, `name`, `class`, `debit`,
#'   `credit` and `balance` (in the normal direction of the account), one
#'   row per account in `chart`, plus attribute `balanced`.
#' @export
trial_balance <- function(entries, chart) {
  n <- nrow(chart)
  debit <- numeric(n)
  credit <- numeric(n)
  for (i in seq_len(n)) {
    rows <- entries$account == chart$code[i]
    debit[i] <- sum(entries$debit[rows])
    credit[i] <- sum(entries$credit[rows])
  }
  sign <- ifelse(chart$side == "debit", 1, -1)

  out <- data.frame(
    code = chart$code,
    name = chart$name,
    class = chart$class,
    debit = debit,
    credit = credit,
    balance = sign * (debit - credit),
    stringsAsFactors = FALSE
  )
  attr(out, "balanced") <- isTRUE(all.equal(sum(debit), sum(credit)))
  out
}

#' Close revenue and expense accounts into retained earnings
#'
#' Revenue accounts carry credit balances and are closed with a debit;
#' expense accounts carry debit balances and are closed with a credit. The
#' net of both goes to `retained`.
#'
#' @param balances A trial balance as returned by [trial_balance()].
#' @param date The closing date.
#' @param retained Account code of retained earnings.
#' @return A data frame of closing journal lines with columns `date`,
#'   `account`, `debit` and `credit`.
#' @export
closing_entries <- function(balances, date, retained = "3100") {
  date <- as.Date(date)
  lines <- list()

  # Close revenue accounts with a debit
  revenue <- balances[balances$class == "revenue" & balances$balance != 0, ]
  for (i in seq_len(nrow(revenue))) {
    lines[[length(lines) + 1L]] <- data.frame(date = date, account = revenue$code[i],
      debit = revenue$balance[i], credit = 0, stringsAsFactors = FALSE)
  }
  # Close expense accounts with a credit
  expense <- balances[balances$class == "expense" & balances$balance != 0, ]
  for (i in seq_len(nrow(expense))) {
    lines[[length(lines) + 1L]] <- data.frame(date = date, account = expense$code[i],
      debit = 0, credit = expense$balance[i], stringsAsFactors = FALSE)
  }

  income <- sum(revenue$balance) - sum(expense$balance)
  lines[[length(lines) + 1L]] <- data.frame(date = date, account = retained,
    debit = max(-income, 0), credit = max(income, 0), stringsAsFactors = FALSE)
  do.call(rbind, lines)
}
