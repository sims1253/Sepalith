#' @title Two-letter weekday codes in ISO order
#' @format A character vector of length 7.
#' @keywords internal
WEEKDAY.CODES <- c("MO", "TU", "WE", "TH", "FR", "SA", "SU")

#' @title Parse a recurrence rule string
#' @description Reads a small subset of iCalendar rules: the fields `FREQ`,
#'   `INTERVAL`, `COUNT`, `UNTIL` and `BYDAY`, separated by `;`.
#' @param x A single string such as `"FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,TH"`.
#' @return A list with elements `freq`, `interval`, `count`, `until` and
#'   `byday`; absent fields are `NULL` except `interval`, which defaults to 1.
#' @export
rrule.parse <- function(x) {
  parts <- strsplit(strsplit(x, ";", fixed = TRUE)[[1L]], "=", fixed = TRUE)
  values <- vapply(parts, function(p) if (length(p) == 2L) p[2L] else NA_character_,
    character(1))
  names(values) <- toupper(vapply(parts, `[`, character(1), 1L))
  if (anyNA(values)) {
    stop("Every field must have the form NAME=VALUE.", call. = FALSE)
  }
  fields <- c("FREQ", "INTERVAL", "COUNT", "UNTIL", "BYDAY")
  unknown <- setdiff(names(values), fields)
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown field(s): %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }

  rule <- list(freq = NULL, interval = 1L, count = NULL, until = NULL, byday = NULL)
  if (!is.na(values["FREQ"])) {
    rule$freq <- tolower(values[["FREQ"]])
  }
  if (!is.na(values["INTERVAL"])) {
    rule$interval <- as.integer(values[["INTERVAL"]])
  }
  if (!is.na(values["COUNT"])) {
    rule$count <- as.integer(values[["COUNT"]])
  }
  if (!is.na(values["UNTIL"])) {
    rule$until <- as.Date(values[["UNTIL"]], format = "%Y%m%d")
  }
  if (!is.na(values["BYDAY"])) {
    rule$byday <- match(strsplit(values[["BYDAY"]], ",", fixed = TRUE)[[1L]], WEEKDAY.CODES)
  }
  rrule.validate(rule)
}

#' @title Check a parsed recurrence rule
#' @param rule A list as returned by [rrule.parse()].
#' @return `rule`, unchanged, or an error naming the first bad field.
#' @keywords internal
rrule.validate <- function(rule) {
  freqs <- c("daily", "weekly", "monthly", "yearly")
  # Check `freq`
  if (is.null(rule$freq) || !(rule$freq %in% freqs)) {
    stop("`freq` must be one of daily, weekly, monthly or yearly.", call. = FALSE)
  }
  # Check `interval`
  if (is.na(rule$interval) || rule$interval < 1L) {
    stop("`interval` must be a positive whole number.", call. = FALSE)
  }
  # Check `count`
  if (!is.null(rule$count) && (is.na(rule$count) || rule$count < 1L)) {
    stop("`count` must be a positive whole number.", call. = FALSE)
  }
  # Check `until`
  if (!is.null(rule$until) && is.na(rule$until)) {
    stop("`until` must be a date in YYYYMMDD form.", call. = FALSE)
  }
  # Check `byday`
  if (!is.null(rule$byday) && (anyNA(rule$byday) || rule$freq != "weekly")) {
    stop("`byday` must list weekday codes and needs a weekly rule.", call. = FALSE)
  }
  rule
}

#' @title Write a recurrence rule back to a string
#' @param rule A list as returned by [rrule.parse()].
#' @return A single string; `rrule.parse(rrule.format(r))` gives back `r`.
#' @export
rrule.format <- function(rule) {
  rule <- rrule.validate(rule)
  parts <- character(0)
  if (!is.null(rule$freq)) {
    parts <- c(parts, paste0("FREQ=", toupper(rule$freq)))
  }
  if (rule$interval != 1L) {
    parts <- c(parts, paste0("INTERVAL=", rule$interval))
  }
  if (!is.null(rule$count)) {
    parts <- c(parts, paste0("COUNT=", rule$count))
  }
  if (!is.null(rule$until)) {
    parts <- c(parts, paste0("UNTIL=", format(rule$until, "%Y%m%d")))
  }
  if (!is.null(rule$byday)) {
    parts <- c(parts, paste0("BYDAY=", paste(WEEKDAY.CODES[rule$byday], collapse = ",")))
  }
  paste(parts, collapse = ";")
}

#' @title Occurrence dates of the k-th recurrence step
#' @param start First occurrence as a `Date`.
#' @param freq One of `"daily"`, `"weekly"`, `"monthly"` or `"yearly"`.
#' @param k Integer vector of step numbers (0 is `start` itself).
#' @return A `Date` vector; monthly and yearly steps that land on a day the
#'   month lacks give `NA`, as in iCalendar.
#' @keywords internal
rrule.step <- function(start, freq = c("daily", "weekly", "monthly", "yearly"), k) {
  freq <- match.arg(freq)
  lt <- as.POSIXlt(start)
  switch(freq,
    daily = {
      start + k
    },
    weekly = {
      start + 7L * k
    },
    monthly = {
      idx <- (lt$year + 1900L) * 12L + lt$mon + k
      as.Date(sprintf("%04d-%02d-%02d", idx %/% 12L, idx %% 12L + 1L, lt$mday),
        optional = TRUE)
    },
    yearly = {
      as.Date(sprintf("%04d-%02d-%02d", lt$year + 1900L + k, lt$mon + 1L, lt$mday),
        optional = TRUE)
    }
  )
}

#' @title Expand a recurrence rule into dates
#' @param rule A list as returned by [rrule.parse()], or a rule string.
#' @param start First occurrence as a `Date`.
#' @param limit Maximum number of dates to return when the rule has neither
#'   `count` nor `until`.
#' @return A sorted `Date` vector.
#' @export
rrule.expand <- function(rule, start, limit = 100L) {
  if (is.character(rule)) {
    rule <- rrule.parse(rule)
  }
  start <- as.Date(start)
  n.max <- if (is.null(rule$count)) limit else rule$count
  until <- if (is.null(rule$until)) as.Date(Inf) else rule$until
  out <- as.Date(character())
  k <- 0L
  while (length(out) < n.max) {
    step <- rrule.step(start, rule$freq, k * rule$interval)
    if (!is.null(rule$byday)) {
      monday <- step - (as.integer(format(step, "%u")) - 1L)
      step <- monday + rule$byday - 1L
      step <- step[step >= start]
    }
    step <- step[!is.na(step)]
    if (length(step) > 0L && min(step) > until) {
      break
    }
    out <- c(out, step[step <= until])
    k <- k + 1L
  }
  head(sort(out), n.max)
}
