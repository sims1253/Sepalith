#' Record ledger problems
#'
#' @param rows Integer row numbers that failed a check.
#' @param field The name of the ledger column that was checked.
#' @param problem A short description of the failure.
#' @return A data frame with columns `row`, `field` and `problem`, with one
#'   row per element of `rows`.
#' @keywords internal
ledger_issue <- function(rows, field, problem) {
  data.frame(
    row = as.integer(rows),
    field = rep(field, length(rows)),
    problem = rep(problem, length(rows)),
    stringsAsFactors = FALSE
  )
}

#' Validate a ledger field by field
#'
#' Each field is checked on its own, so one bad column does not hide
#' problems in the others. The fields are checked in this order: `date`,
#' `account`, `debit`, `credit`, `currency` and `reference`. Account codes
#' are four digits, optionally followed by a dash and a two-digit
#' sub-account (`"1200"`, `"1200-03"`); currency codes are three upper-case
#' letters.
#'
#' @param ledger A ledger data frame with columns `date`, `account`, `debit`,
#'   `credit`, `currency` and `reference`.
#' @param period Optional length-2 `Date` vector; when given, `date` must
#'   fall inside it.
#' @return A data frame of problems as built by `ledger_issue()`; it has zero
#'   rows when the ledger is valid.
#' @export
validate_ledger <- function(ledger, period = NULL) {
  fields <- c("date", "account", "debit", "credit", "currency", "reference")
  missing <- setdiff(fields, names(ledger))
  if (length(missing) > 0L) {
    stop("Ledger is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  issues <- list()

  # Field `date`
  bad <- which(is.na(ledger$date))
  issues$date <- ledger_issue(bad, "date", "missing value")
  if (!is.null(period)) {
    outside <- which(ledger$date < period[1L] | ledger$date > period[2L])
    issues$date <- rbind(issues$date, ledger_issue(outside, "date", "outside period"))
  }
  # Field `account`
  bad <- which(is.na(ledger$account) | !grepl("^[0-9]{4}(-[0-9]{2})?$", ledger$account))
  issues$account <- ledger_issue(bad, "account", "invalid code")
  # Field `debit`
  bad <- which(is.na(ledger$debit) | ledger$debit < 0)
  issues$debit <- ledger_issue(bad, "debit", "missing or negative")
  # Field `credit`
  bad <- which(is.na(ledger$credit) | ledger$credit < 0)
  issues$credit <- ledger_issue(bad, "credit", "missing or negative")
  # Field `currency`
  bad <- which(is.na(ledger$currency) | !grepl("^[A-Z]{3}$", ledger$currency))
  issues$currency <- ledger_issue(bad, "currency", "invalid code")
  # Field `reference`
  bad <- which(duplicated(ledger$reference) | is.na(ledger$reference))
  issues$reference <- ledger_issue(bad, "reference", "missing or duplicated")

  out <- do.call(rbind, unname(issues))
  out[order(out$row, match(out$field, fields)), , drop = FALSE]
}

#' Check that the debit and credit sides of a ledger agree
#'
#' Both sides are checked the same way: missing amounts count as zero and
#' are reported, negative amounts are reported, and the side is totalled.
#'
#' @param ledger A ledger data frame with columns `date`, `account`, `debit`,
#'   `credit`, `currency` and `reference`.
#' @param tolerance Largest absolute difference treated as zero.
#' @return A list with elements `debits`, `credits`, `difference`,
#'   `missing` and `negative` (each a list of row numbers with elements
#'   `debit` and `credit`), `one_sided` (row numbers with both or neither
#'   side filled) and `ok`.
#' @export
validate_balance <- function(ledger, tolerance = 0.005) {
  missing <- list()
  negative <- list()

  # Debit side
  debit <- ledger$debit
  missing$debit <- which(is.na(debit))
  debit[missing$debit] <- 0
  negative$debit <- which(debit < 0)
  debits <- sum(debit)
  # Credit side
  credit <- ledger$credit
  missing$credit <- which(is.na(credit))
  credit[missing$credit] <- 0
  negative$credit <- which(credit < 0)
  credits <- sum(credit)

  # Every row should carry exactly one side
  one_sided <- which((debit > 0) == (credit > 0))
  difference <- debits - credits
  problems <- length(unlist(missing)) + length(unlist(negative)) + length(one_sided)
  list(
    debits = debits,
    credits = credits,
    difference = difference,
    missing = missing,
    negative = negative,
    one_sided = one_sided,
    ok = abs(difference) <= tolerance && problems == 0L
  )
}

#' Describe each ledger field
#'
#' @param ledger A ledger data frame with columns `date`, `account`, `debit`,
#'   `credit`, `currency` and `reference`.
#' @return A data frame with one row per field (`date`, `account`, `debit`,
#'   `credit`, `currency`, `reference`) and columns `field`, `missing`,
#'   `distinct` and `example`.
#' @export
ledger_field_summary <- function(ledger) {
  rows <- list()

  # Field `date`
  x <- ledger$date
  rows$date <- data.frame(field = "date", missing = sum(is.na(x)),
    distinct = length(unique(x[!is.na(x)])), example = format(x[!is.na(x)][1L]))
  # Field `account`
  x <- ledger$account
  rows$account <- data.frame(field = "account", missing = sum(is.na(x)),
    distinct = length(unique(x[!is.na(x)])), example = format(x[!is.na(x)][1L]))
  # Field `debit`
  x <- ledger$debit
  rows$debit <- data.frame(field = "debit", missing = sum(is.na(x)),
    distinct = length(unique(x[!is.na(x)])), example = format(x[!is.na(x)][1L]))
  # Field `credit`
  x <- ledger$credit
  rows$credit <- data.frame(field = "credit", missing = sum(is.na(x)),
    distinct = length(unique(x[!is.na(x)])), example = format(x[!is.na(x)][1L]))
  # Field `currency`
  x <- ledger$currency
  rows$currency <- data.frame(field = "currency", missing = sum(is.na(x)),
    distinct = length(unique(x[!is.na(x)])), example = format(x[!is.na(x)][1L]))
  # Field `reference`
  x <- ledger$reference
  rows$reference <- data.frame(field = "reference", missing = sum(is.na(x)),
    distinct = length(unique(x[!is.na(x)])), example = format(x[!is.na(x)][1L]))

  out <- do.call(rbind, unname(rows))
  rownames(out) <- NULL
  out
}

#' Repair common ledger formatting problems
#'
#' Applies one fix per field and leaves everything else untouched:
#' `account` is trimmed, `debit` and `credit` are rounded to cents with
#' missing values set to zero, `currency` is upper-cased and blank
#' `reference` values are replaced by a generated `"AUTO-<row>"` reference.
#'
#' @param ledger A ledger data frame with columns `date`, `account`, `debit`,
#'   `credit`, `currency` and `reference`.
#' @return A list with elements `ledger` (the repaired data frame) and
#'   `changed` (the number of changed values per field).
#' @export
repair_ledger <- function(ledger) {
  changed <- list()

  # Repair `account`
  fixed <- trimws(ledger$account)
  changed$account <- sum(fixed != ledger$account, na.rm = TRUE)
  ledger$account <- fixed
  # Repair `debit`
  fixed <- round(ifelse(is.na(ledger$debit), 0, ledger$debit), 2)
  changed$debit <- sum(fixed != ledger$debit | is.na(ledger$debit))
  ledger$debit <- fixed
  # Repair `credit`
  fixed <- round(ifelse(is.na(ledger$credit), 0, ledger$credit), 2)
  changed$credit <- sum(fixed != ledger$credit | is.na(ledger$credit))
  ledger$credit <- fixed
  # Repair `currency`
  fixed <- toupper(trimws(ledger$currency))
  changed$currency <- sum(fixed != ledger$currency, na.rm = TRUE)
  ledger$currency <- fixed
  # Repair `reference`
  blank <- is.na(ledger$reference) | !nzchar(trimws(ledger$reference))
  ledger$reference[blank] <- paste0("AUTO-", which(blank))
  changed$reference <- sum(blank)

  list(ledger = ledger, changed = changed)
}

#' Stop when a ledger fails validation
#'
#' @param ledger A ledger data frame with columns `date`, `account`, `debit`,
#'   `credit`, `currency` and `reference`.
#' @param period Optional length-2 `Date` vector passed to
#'   [validate_ledger()].
#' @param max_shown Maximum number of problems listed in the error message.
#' @return `ledger`, invisibly, when there are no problems.
#' @export
assert_ledger <- function(ledger, period = NULL, max_shown = 5L) {
  issues <- validate_ledger(ledger, period)
  balance <- validate_balance(ledger)
  if (nrow(issues) == 0L && balance$ok) {
    return(invisible(ledger))
  }
  lines <- sprintf("row %d, `%s`: %s", issues$row, issues$field, issues$problem)
  if (!balance$ok) {
    lines <- c(lines, sprintf("debits and credits differ by %.2f", balance$difference))
  }
  if (length(lines) > max_shown) {
    lines <- c(head(lines, max_shown), sprintf("... and %d more", length(lines) - max_shown))
  }
  stop("Ledger failed validation:\n", paste0("* ", lines, collapse = "\n"), call. = FALSE)
}
