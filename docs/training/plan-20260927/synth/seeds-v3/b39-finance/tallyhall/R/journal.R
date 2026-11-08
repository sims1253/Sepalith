#' Build a two-line journal entry
#'
#' The debit line and the credit line mirror each other: same date, same
#' amount, same memo, opposite side.
#'
#' @param debit_account Account code debited.
#' @param credit_account Account code credited.
#' @param amount A single positive amount.
#' @param date The entry date.
#' @param memo Optional memo copied onto both lines.
#' @return A data frame with columns `date`, `account`, `debit`, `credit` and
#'   `memo`, and two rows: the debit line first, then the credit line.
#' @export
journal_entry <- function(debit_account, credit_account, amount, date, memo = "") {
  if (!is.numeric(amount) || length(amount) != 1L || is.na(amount) || amount <= 0) {
    stop("`amount` must be a single positive number.", call. = FALSE)
  }
  if (identical(debit_account, credit_account)) {
    stop("`debit_account` and `credit_account` must differ.", call. = FALSE)
  }
  date <- as.Date(date)

  debit_line <- data.frame(
    date = date,
    account = debit_account,
    debit = amount,
    credit = 0,
    memo = memo,
    stringsAsFactors = FALSE
  )
  credit_line <- data.frame(
    date = date,
    account = credit_account,
    debit = 0,
    credit = amount,
    memo = memo,
    stringsAsFactors = FALSE
  )
  return(rbind(debit_line, credit_line))
}

#' Reverse a journal entry
#'
#' @param entry A journal entry data frame as returned by [journal_entry()].
#' @param date Date of the reversal; defaults to the original entry date.
#' @return A data frame of the same shape as `entry` with the `debit` and
#'   `credit` columns swapped and the memo prefixed with `"Reversal: "`.
#' @export
reverse_entry <- function(entry, date = entry$date) {
  reversed <- entry
  reversed$debit <- entry$credit
  reversed$credit <- entry$debit
  reversed$date <- as.Date(date)
  reversed$memo <- paste0("Reversal: ", entry$memo)
  return(reversed)
}

#' Total each side of a set of journal entries by account
#'
#' @param entries A data frame of journal lines with columns `account`,
#'   `debit` and `credit`.
#' @return A data frame with columns `account`, `debit`, `credit`, `net`
#'   (debit minus credit) and `side` (`"debit"`, `"credit"` or `"zero"`),
#'   one row per account, sorted by account.
#' @export
side_totals <- function(entries) {
  accounts <- sort(unique(entries$account))
  debit <- numeric(length(accounts))
  credit <- numeric(length(accounts))

  # Accumulate the debit side
  for (i in seq_along(accounts)) {
    rows <- entries$account == accounts[i]
    debit[i] <- sum(entries$debit[rows], na.rm = TRUE)
  }
  # Accumulate the credit side
  for (i in seq_along(accounts)) {
    rows <- entries$account == accounts[i]
    credit[i] <- sum(entries$credit[rows], na.rm = TRUE)
  }

  net <- debit - credit
  side <- ifelse(net > 0, "debit", ifelse(net < 0, "credit", "zero"))
  return(data.frame(
    account = accounts,
    debit = debit,
    credit = credit,
    net = net,
    side = side,
    stringsAsFactors = FALSE
  ))
}

#' Build a period-end adjusting entry
#'
#' @details
#' The accounts used for each type come from `accounts`:
#' * `"accrued_expense"`: debit `expense`, credit `accrued_liability`;
#' * `"accrued_revenue"`: debit `accrued_asset`, credit `revenue`;
#' * `"prepaid_expense"`: debit `expense`, credit `prepaid_asset`;
#' * `"deferred_revenue"`: debit `deferred_liability`, credit `revenue`.
#'
#' @param type One of `"accrued_expense"`, `"accrued_revenue"`,
#'   `"prepaid_expense"` or `"deferred_revenue"`.
#' @param amount A single positive amount.
#' @param date The entry date.
#' @param accounts A named list of account codes with elements `expense`,
#'   `revenue`, `accrued_asset`, `accrued_liability`, `prepaid_asset` and
#'   `deferred_liability`.
#' @return A two-line journal entry as returned by [journal_entry()].
#' @export
adjusting_entry <- function(type = c("accrued_expense", "accrued_revenue",
                                     "prepaid_expense", "deferred_revenue"),
                            amount, date, accounts) {
  type <- match.arg(type)
  memo <- paste("Adjustment:", gsub("_", " ", type))

  switch(type,
    accrued_expense = {
      journal_entry(
        debit_account = accounts$expense,
        credit_account = accounts$accrued_liability,
        amount = amount, date = date, memo = memo
      )
    },
    accrued_revenue = {
      journal_entry(
        debit_account = accounts$accrued_asset,
        credit_account = accounts$revenue,
        amount = amount, date = date, memo = memo
      )
    },
    prepaid_expense = {
      journal_entry(
        debit_account = accounts$expense,
        credit_account = accounts$prepaid_asset,
        amount = amount, date = date, memo = memo
      )
    },
    deferred_revenue = {
      journal_entry(
        debit_account = accounts$deferred_liability,
        credit_account = accounts$revenue,
        amount = amount, date = date, memo = memo
      )
    }
  )
}

#' Split one amount across several accounts on the credit side
#'
#' @param debit_account Account code debited with the full amount.
#' @param credit_accounts Account codes credited.
#' @param weights Non-negative weights, one per element of
#'   `credit_accounts`; they are normalised to sum to one.
#' @param amount A single positive amount.
#' @param date The entry date.
#' @param memo Optional memo copied onto every line.
#' @return A journal data frame with one debit line followed by one credit
#'   line per credit account. Rounding differences go to the last line.
#' @export
split_entry <- function(debit_account, credit_accounts, weights, amount, date, memo = "") {
  if (length(weights) != length(credit_accounts) || any(weights < 0)) {
    stop("`weights` must be non-negative and match `credit_accounts`.", call. = FALSE)
  }
  shares <- round(amount * weights / sum(weights), 2)
  shares[length(shares)] <- amount - sum(shares[-length(shares)])
  date <- as.Date(date)

  debit_line <- data.frame(
    date = date,
    account = debit_account,
    debit = amount,
    credit = 0,
    memo = memo,
    stringsAsFactors = FALSE
  )
  credit_lines <- data.frame(
    date = rep(date, length(shares)),
    account = credit_accounts,
    debit = 0,
    credit = shares,
    memo = memo,
    stringsAsFactors = FALSE
  )
  return(rbind(debit_line, credit_lines))
}

#' Format journal lines for display
#'
#' Debit lines are printed flush left; credit lines are indented by
#' `indent` spaces, following the usual journal convention. The output
#' starts with a header line and ends with a rule and a totals line.
#'
#' @param entry A journal data frame with columns `date`, `account`,
#'   `debit` and `credit`.
#' @param indent Number of spaces used to indent credit lines.
#' @param title Optional title printed above the header.
#' @return A character vector of formatted lines.
#' @export
format_entry <- function(entry, indent = 4L, title = NULL) {
  pad <- strrep(" ", indent)
  header <- sprintf("%-10s  %-*s %12s %12s", "Date", 13L + indent, "Account", "Debit",
    "Credit")
  body <- character(nrow(entry))

  for (i in seq_len(nrow(entry))) {
    if (entry$debit[i] > 0) {
      # Debit line: amount in the debit column
      body[i] <- sprintf("%-10s  Dr %-*s %12.2f %12s", format(entry$date[i]),
        10L + indent, entry$account[i], entry$debit[i], "")
    } else {
      # Credit line: indented, amount in the credit column
      body[i] <- sprintf("%-10s  %sCr %-10s %12s %12.2f", format(entry$date[i]), pad,
        entry$account[i], "", entry$credit[i])
    }
  }

  rule <- strrep("-", nchar(header))
  totals <- sprintf("%-*s %12.2f %12.2f", 25L + indent, "Total", sum(entry$debit),
    sum(entry$credit))
  return(c(title, header, body, rule, totals))
}
