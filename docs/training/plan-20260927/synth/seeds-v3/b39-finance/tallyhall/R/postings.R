#' Create a posting batch
#'
#' Scalars are recycled to the length of `amount`, so a batch for a single
#' date or a single account can be written compactly.
#'
#' @param date Posting dates, coercible with [as.Date()].
#' @param account Account codes as a character vector.
#' @param amount Non-negative posting amounts.
#' @param side Either `"debit"` or `"credit"` for each posting.
#' @param memo Optional free-text memo for each posting.
#' @return A data frame with columns `date`, `account`, `amount`, `side` and
#'   `memo`, one row per posting.
#' @export
new_posting_batch <- function(date, account, amount, side, memo = "") {
  if (!is.numeric(amount) || anyNA(amount) || any(amount < 0)) {
    stop("`amount` must be a non-negative numeric vector.", call. = FALSE)
  }
  n <- length(amount)

  # Recycle and check `date`
  date <- as.Date(date)
  if (length(date) == 1L) date <- rep(date, n)
  if (length(date) != n || anyNA(date)) {
    stop(sprintf("`%s` must have length 1 or %d with no missing values.", "date", n),
      call. = FALSE)
  }
  # Recycle and check `account`
  account <- as.character(account)
  if (length(account) == 1L) account <- rep(account, n)
  if (length(account) != n || anyNA(account)) {
    stop(sprintf("`%s` must have length 1 or %d with no missing values.", "account", n),
      call. = FALSE)
  }
  # Recycle and check `side`
  side <- tolower(as.character(side))
  if (length(side) == 1L) side <- rep(side, n)
  if (length(side) != n || anyNA(side) || !all(side %in% c("debit", "credit"))) {
    stop(sprintf("`%s` must have length 1 or %d with no missing values.", "side", n),
      call. = FALSE)
  }
  # Recycle and check `memo`
  memo <- as.character(memo)
  if (length(memo) == 1L) memo <- rep(memo, n)
  if (length(memo) != n || anyNA(memo)) {
    stop(sprintf("`%s` must have length 1 or %d with no missing values.", "memo", n),
      call. = FALSE)
  }

  return(data.frame(
    date = date,
    account = account,
    amount = amount,
    side = side,
    memo = memo,
    stringsAsFactors = FALSE
  ))
}

#' Apply a posting batch to account balances
#'
#' @details
#' Balances are kept in their natural sign. `cash`, `receivables` and
#' `inventory` are debit-normal, so debits increase them. `payables` and
#' `revenue` are credit-normal, so credits increase them.
#'
#' @param balances A named list of current balances with elements `cash`,
#'   `receivables`, `inventory`, `payables` and `revenue`.
#' @param batch A posting batch as returned by [new_posting_batch()].
#' @return A list with elements `balances` (the updated list) and `posted`
#'   (the number of postings applied to each account).
#' @export
apply_postings <- function(balances, batch) {
  signed <- ifelse(batch$side == "debit", 1, -1) * batch$amount
  posted <- list()

  # Debit-normal: cash
  hit <- batch$account == "cash"
  balances$cash <- balances$cash + sum(signed[hit])
  posted$cash <- sum(hit)
  # Debit-normal: receivables
  hit <- batch$account == "receivables"
  balances$receivables <- balances$receivables + sum(signed[hit])
  posted$receivables <- sum(hit)
  # Debit-normal: inventory
  hit <- batch$account == "inventory"
  balances$inventory <- balances$inventory + sum(signed[hit])
  posted$inventory <- sum(hit)
  # Credit-normal: payables
  hit <- batch$account == "payables"
  balances$payables <- balances$payables - sum(signed[hit])
  posted$payables <- sum(hit)
  # Credit-normal: revenue
  hit <- batch$account == "revenue"
  balances$revenue <- balances$revenue - sum(signed[hit])
  posted$revenue <- sum(hit)

  unknown <- setdiff(unique(batch$account), names(posted))
  if (length(unknown) > 0L) {
    warning("Skipped postings to unknown accounts: ",
      paste(unknown, collapse = ", "), call. = FALSE)
  }
  return(list(balances = balances, posted = posted))
}

#' Summarise a posting batch
#'
#' The debit side and the credit side are summarised separately and then
#' compared.
#'
#' @param batch A posting batch as returned by [new_posting_batch()].
#' @return A list with elements `n` (number of postings), `debits` and
#'   `credits` (each a list with `n`, `total`, `largest` and `accounts`),
#'   `net` (debit total minus credit total), `first` and `last` (earliest and
#'   latest posting date) and `balanced` (`TRUE` when `net` is zero to the
#'   cent).
#' @export
posting_summary <- function(batch) {
  is_debit <- batch$side == "debit"

  # Debit side
  amounts <- batch$amount[is_debit]
  debits <- list(
    n = length(amounts),
    total = sum(amounts),
    largest = if (length(amounts)) max(amounts) else 0,
    accounts = sort(unique(batch$account[is_debit]))
  )
  # Credit side
  amounts <- batch$amount[!is_debit]
  credits <- list(
    n = length(amounts),
    total = sum(amounts),
    largest = if (length(amounts)) max(amounts) else 0,
    accounts = sort(unique(batch$account[!is_debit]))
  )

  net <- debits$total - credits$total
  dates <- range(batch$date)
  return(list(
    n = nrow(batch),
    debits = debits,
    credits = credits,
    net = net,
    first = dates[1L],
    last = dates[2L],
    balanced = abs(round(net, 2)) < 0.005
  ))
}

#' Latest running balance of one account
#'
#' @param ledger A running ledger data frame with columns `date`, `account`,
#'   `debit`, `credit`, `balance` and `memo`.
#' @param account The account name.
#' @return The last recorded balance of `account`, or `0` if it has none.
#' @keywords internal
last_balance <- function(ledger, account) {
  previous <- ledger$balance[ledger$account == account]
  if (length(previous) == 0L) {
    return(0)
  }
  return(previous[length(previous)])
}

#' Post a two-sided business event to a running ledger
#'
#' Each helper appends a debit row and a credit row and carries the running
#' balance of both accounts forward. Balances are kept in their natural
#' sign: `cash`, `receivables` and `inventory` grow with debits, `payables`
#' and `revenue` grow with credits.
#'
#' * `post_cash_receipt()`: debit `cash`, credit `receivables`;
#' * `post_cash_payment()`: debit `payables`, credit `cash`;
#' * `post_credit_sale()`: debit `receivables`, credit `revenue`;
#' * `post_credit_purchase()`: debit `inventory`, credit `payables`.
#'
#' @param ledger A running ledger data frame with columns `date`, `account`,
#'   `debit`, `credit`, `balance` and `memo`.
#' @param amount A single positive amount.
#' @param date The posting date.
#' @param memo Optional memo copied onto both rows.
#' @return `ledger` with two rows appended.
#' @rdname post_event
#' @export
post_cash_receipt <- function(ledger, amount, date, memo = "") {
  if (!is.numeric(amount) || length(amount) != 1L || is.na(amount) || amount <= 0) {
    stop("`amount` must be a single positive number.", call. = FALSE)
  }
  date <- as.Date(date)

  # Debit `cash`
  debit_row <- data.frame(
    date = date,
    account = "cash",
    debit = amount,
    credit = 0,
    balance = last_balance(ledger, "cash") + amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  # Credit `receivables`
  credit_row <- data.frame(
    date = date,
    account = "receivables",
    debit = 0,
    credit = amount,
    balance = last_balance(ledger, "receivables") - amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  return(rbind(ledger, debit_row, credit_row))
}

#' @rdname post_event
#' @export
post_cash_payment <- function(ledger, amount, date, memo = "") {
  if (!is.numeric(amount) || length(amount) != 1L || is.na(amount) || amount <= 0) {
    stop("`amount` must be a single positive number.", call. = FALSE)
  }
  date <- as.Date(date)

  # Debit `payables`
  debit_row <- data.frame(
    date = date,
    account = "payables",
    debit = amount,
    credit = 0,
    balance = last_balance(ledger, "payables") - amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  # Credit `cash`
  credit_row <- data.frame(
    date = date,
    account = "cash",
    debit = 0,
    credit = amount,
    balance = last_balance(ledger, "cash") - amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  return(rbind(ledger, debit_row, credit_row))
}

#' @rdname post_event
#' @export
post_credit_sale <- function(ledger, amount, date, memo = "") {
  if (!is.numeric(amount) || length(amount) != 1L || is.na(amount) || amount <= 0) {
    stop("`amount` must be a single positive number.", call. = FALSE)
  }
  date <- as.Date(date)

  # Debit `receivables`
  debit_row <- data.frame(
    date = date,
    account = "receivables",
    debit = amount,
    credit = 0,
    balance = last_balance(ledger, "receivables") + amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  # Credit `revenue`
  credit_row <- data.frame(
    date = date,
    account = "revenue",
    debit = 0,
    credit = amount,
    balance = last_balance(ledger, "revenue") + amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  return(rbind(ledger, debit_row, credit_row))
}

#' @rdname post_event
#' @export
post_credit_purchase <- function(ledger, amount, date, memo = "") {
  if (!is.numeric(amount) || length(amount) != 1L || is.na(amount) || amount <= 0) {
    stop("`amount` must be a single positive number.", call. = FALSE)
  }
  date <- as.Date(date)

  # Debit `inventory`
  debit_row <- data.frame(
    date = date,
    account = "inventory",
    debit = amount,
    credit = 0,
    balance = last_balance(ledger, "inventory") + amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  # Credit `payables`
  credit_row <- data.frame(
    date = date,
    account = "payables",
    debit = 0,
    credit = amount,
    balance = last_balance(ledger, "payables") + amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  return(rbind(ledger, debit_row, credit_row))
}
