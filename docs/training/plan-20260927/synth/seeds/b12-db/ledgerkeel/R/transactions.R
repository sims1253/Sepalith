#' Run code inside a savepoint
#'
#' Creates a named savepoint, evaluates `code`, and releases the savepoint
#' on success or rolls back to it on error. Must be called inside an open
#' transaction.
#'
#' @param con A `DBIConnection` object.
#' @param name Name of the savepoint.
#' @param code A function taking the connection as its only argument.
#' @return The value returned by `code`.
#' @export
with_savepoint <- function(con, name, code) {
  code <- match.fun(code)
  sp <- DBI::dbQuoteIdentifier(con, name)
  DBI::dbExecute(con, paste("SAVEPOINT", sp))
  result <- tryCatch(
    code(con),
    error = function(e) {
      DBI::dbExecute(con, paste("ROLLBACK TO SAVEPOINT", sp))
      stop(e)
    }
  )
  DBI::dbExecute(con, paste("RELEASE SAVEPOINT", sp))
  return(result)
}

#' Is an error a serialization failure?
#'
#' Recognises the Postgres serialization and deadlock errors that are safe
#' to retry.
#'
#' @param e A condition object.
#' @return `TRUE` or `FALSE`.
#' @keywords internal
is_retryable <- function(e) {
  msg <- conditionMessage(e)
  patterns <- c("could not serialize", "deadlock detected", "40001", "40P01")
  return(any(vapply(patterns, grepl, logical(1), x = msg, fixed = TRUE)))
}

#' Run a transaction with retries
#'
#' Wraps [DBI::dbWithTransaction()] and retries when the database reports a
#' serialization failure or deadlock, backing off between attempts.
#'
#' @param con A `DBIConnection` object.
#' @param code A function taking the connection as its only argument.
#' @param attempts Maximum number of attempts.
#' @param wait Initial wait between attempts, in seconds; doubles each time.
#' @return The value returned by `code`.
#' @export
with_retry <- function(con, code, attempts = 3L, wait = 0.2) {
  code <- match.fun(code)
  for (i in seq_len(attempts)) {
    outcome <- tryCatch(
      list(ok = TRUE, value = DBI::dbWithTransaction(con, code(con))),
      error = function(e) list(ok = FALSE, error = e)
    )
    if (outcome$ok) {
      return(outcome$value)
    }
    if (!is_retryable(outcome$error) || i == attempts) {
      stop(outcome$error)
    }
    Sys.sleep(wait * 2^(i - 1))
  }
}

#' Fetch an account balance
#'
#' Sums the signed amounts of all journal lines posted to `account_id`.
#'
#' @param con A `DBIConnection` object.
#' @param account_id Identifier of the account.
#' @return A single number; zero for an account without postings.
#' @export
account_balance <- function(con, account_id) {
  res <- DBI::dbGetQuery(
    con,
    "SELECT COALESCE(SUM(amount), 0) AS balance FROM journal_lines WHERE account_id = ?",
    params = list(account_id)
  )
  return(as.numeric(res$balance))
}

#' Post a debit line
#'
#' Inserts a negative journal line for `account_id` under `entry_id`.
#'
#' @param con A `DBIConnection` object.
#' @param entry_id Identifier of the journal entry.
#' @param account_id Identifier of the account.
#' @param amount Positive amount to post.
#' @return The number of rows inserted.
#' @family postings
#' @keywords internal
post_debit <- function(con, entry_id, account_id, amount) {
  stopifnot(is.numeric(amount), length(amount) == 1, amount > 0)
  n <- DBI::dbExecute(
    con,
    "INSERT INTO journal_lines (entry_id, account_id, amount) VALUES (?, ?, ?)",
    params = list(entry_id, account_id, -amount)
  )
  return(n)
}

#' Post a credit line
#'
#' Inserts a positive journal line for `account_id` under `entry_id`.
#'
#' @param con A `DBIConnection` object.
#' @param entry_id Identifier of the journal entry.
#' @param account_id Identifier of the account.
#' @param amount Positive amount to post.
#' @return The number of rows inserted.
#' @family postings
#' @keywords internal
post_credit <- function(con, entry_id, account_id, amount) {
  stopifnot(is.numeric(amount), length(amount) == 1, amount > 0)
  n <- DBI::dbExecute(
    con,
    "INSERT INTO journal_lines (entry_id, account_id, amount) VALUES (?, ?, ?)",
    params = list(entry_id, account_id, amount)
  )
  return(n)
}

#' Transfer an amount between two accounts
#'
#' Creates a journal entry with one debit and one credit line in a single
#' transaction. The source account must hold enough funds unless
#' `allow_overdraft` is `TRUE`.
#'
#' @param con A `DBIConnection` object.
#' @param from_account Identifier of the account to debit.
#' @param to_account Identifier of the account to credit.
#' @param amount Positive amount to post.
#' @param memo Free-text description stored with the entry.
#' @param allow_overdraft Whether the source balance may go negative.
#' @return The identifier of the new journal entry.
#' @export
transfer <- function(con, from_account, to_account, amount, memo = "",
                     allow_overdraft = FALSE) {
  if (identical(from_account, to_account)) {
    stop("Source and target account must differ.", call. = FALSE)
  }
  with_retry(con, function(con) {
    if (!allow_overdraft && account_balance(con, from_account) < amount) {
      stop("Insufficient funds in account ", from_account, ".", call. = FALSE)
    }
    entry <- DBI::dbGetQuery(
      con,
      "INSERT INTO journal_entries (memo, posted_at) VALUES (?, CURRENT_TIMESTAMP)
       RETURNING entry_id",
      params = list(memo)
    )
    post_debit(con, entry$entry_id, from_account, amount)
    post_credit(con, entry$entry_id, to_account, amount)
    entry$entry_id
  })
}

#' Check that every journal entry balances
#'
#' Finds journal entries whose lines do not sum to zero, which would mean a
#' half-written transfer slipped through.
#'
#' @param con A `DBIConnection` object.
#' @param tolerance Largest absolute imbalance treated as zero.
#' @return A data frame with columns `entry_id` and `imbalance`; empty when
#'   the ledger balances.
#' @export
unbalanced_entries <- function(con, tolerance = 1e-9) {
  sql <- glue::glue_sql(
    "SELECT entry_id, SUM(amount) AS imbalance
     FROM journal_lines
     GROUP BY entry_id
     HAVING ABS(SUM(amount)) > {tolerance}
     ORDER BY entry_id",
    .con = con
  )
  return(DBI::dbGetQuery(con, sql))
}
