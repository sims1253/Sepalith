#' Format rates for display
#'
#' @param x A numeric vector of proportions.
#' @param style Display style: one of `"percent"`, `"permille"`,
#'   `"fraction"` or `"ratio"`.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `x`; `NA` values become
#'   `"-"`.
#' @export
#' @family formatters
formatRate = function(x, style = c("percent", "permille", "fraction", "ratio"),
                      digits = 1) {
  style = match.arg(style)
  scale = c(percent = 100, permille = 1000, fraction = 1, ratio = 1)
  scaled = x * scale[[style]]
  out = switch(style,
    percent = {
      paste0(formatC(scaled, format = "f", digits = digits), "%")
    },
    permille = {
      paste0(formatC(scaled, format = "f", digits = digits), " per mille")
    },
    fraction = {
      formatC(scaled, format = "f", digits = digits + 2)
    },
    ratio = {
      denominator = ifelse(scaled > 0, round(1 / scaled), NA)
      paste0("1 in ", formatC(denominator, format = "d", big.mark = ","))
    }
  )
  out[is.na(x)] = "-"
  return(out)
}

#' Describe progress against a target
#'
#' @param achieved Number of completed interviews.
#' @param target Target number of interviews.
#' @param slack Share of the target by which a cell may differ and still be
#'   on track.
#' @return A list with elements `pct`, `label` and `status` (`"under"`,
#'   `"on"` or `"over"`).
#' @export
#' @family formatters
formatProgress = function(achieved, target, slack = 0.05) {
  if (is.na(achieved) || is.na(target) || target <= 0) {
    return(list(pct = NA_real_, label = "no target", status = "on"))
  }
  pct = achieved / target
  if (pct < 1 - slack) {
    missing = target - achieved
    label = sprintf("%d of %d (%d short)", as.integer(achieved), as.integer(target),
                    as.integer(missing))
    status = "under"
  } else if (pct > 1 + slack) {
    extra = achieved - target
    label = sprintf("%d of %d (%d over)", as.integer(achieved), as.integer(target),
                    as.integer(extra))
    status = "over"
  } else {
    label = sprintf("%d of %d", as.integer(achieved), as.integer(target))
    status = "on"
  }
  return(list(pct = pct, label = label, status = status))
}

#' Format a column for a table cell
#'
#' @param x A vector of values from one column.
#' @param type Kind of column: one of `"count"`, `"rate"`, `"date"`,
#'   `"duration"` or `"text"`.
#' @param na Text shown for missing values.
#' @return A list with elements `text` (character), `align` (`"left"` or
#'   `"right"`) and `sortKey` (numeric or character, used for ordering).
#' @export
#' @family formatters
formatCells = function(x, type = c("count", "rate", "date", "duration", "text"),
                       na = "-") {
  type = match.arg(type)
  missing = is.na(x)
  result = switch(type,
    count = {
      text = formatC(as.numeric(x), format = "d", big.mark = ",")
      list(text = text, align = "right", sortKey = as.numeric(x))
    },
    rate = {
      text = formatRate(as.numeric(x), "percent")
      list(text = text, align = "right", sortKey = as.numeric(x))
    },
    date = {
      text = format(as.Date(x), "%d %b %Y")
      list(text = text, align = "right", sortKey = as.numeric(as.Date(x)))
    },
    duration = {
      minutes = as.integer(x %/% 60)
      text = sprintf("%d:%02d", minutes, as.integer(round(x - 60 * minutes)))
      list(text = text, align = "right", sortKey = as.numeric(x))
    },
    text = {
      text = as.character(x)
      list(text = text, align = "left", sortKey = tolower(text))
    }
  )
  result$text[missing] = na
  return(result)
}

#' Format the summary of a fieldwork wave
#'
#' @param summary A list with elements `waveId`, `startDate`, `endDate`,
#'   `targetN`, `completes` and `responseRate`.
#' @return A named character vector with one `"Label: value"` line per
#'   element of `summary`.
#' @export
#' @family formatters
formatWaveSummary = function(summary) {
  labels = c(
    waveId = "Wave", startDate = "Start", endDate = "End",
    targetN = "Target", completes = "Completes", responseRate = "Response rate"
  )
  lines = character(0)

  # Identifier and dates
  lines[["waveId"]] = sprintf("%s: %s", labels[["waveId"]],
                              formatCells(summary$waveId, "text")$text)
  lines[["startDate"]] = sprintf("%s: %s", labels[["startDate"]],
                                 formatCells(summary$startDate, "date")$text)
  lines[["endDate"]] = sprintf("%s: %s", labels[["endDate"]],
                               formatCells(summary$endDate, "date")$text)

  # Counts and rates
  lines[["targetN"]] = sprintf("%s: %s", labels[["targetN"]],
                               formatCells(summary$targetN, "count")$text)
  lines[["completes"]] = sprintf("%s: %s", labels[["completes"]],
                                 formatCells(summary$completes, "count")$text)
  lines[["responseRate"]] = sprintf("%s: %s", labels[["responseRate"]],
                                    formatCells(summary$responseRate, "rate")$text)
  return(lines)
}

#' Format the interviewer table
#'
#' @param data A data frame with columns `interviewer`, `completes`,
#'   `refusals`, `rate`, `lastContact` and `avgDuration`.
#' @param na Text shown for missing values.
#' @return A list with elements `table` (the formatted data frame) and
#'   `align` (a named character vector of column alignments).
#' @export
#' @family formatters
formatInterviewerTable = function(data, na = "-") {
  types = c(
    interviewer = "text", completes = "count", refusals = "count",
    rate = "rate", lastContact = "date", avgDuration = "duration"
  )
  missing = setdiff(names(types), names(data))
  if (length(missing) > 0L) {
    stop("Missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  table = data[names(types)]
  align = character(0)
  for (column in names(types)) {
    cells = formatCells(data[[column]], types[[column]], na = na)
    table[[column]] = cells$text
    align[[column]] = cells$align
  }
  ranking = order(-data$completes, data$interviewer)
  table = table[ranking, , drop = FALSE]
  rownames(table) = NULL
  return(list(table = table, align = align))
}

#' Count interview outcomes
#'
#' @param outcomes A character vector of outcomes, each one of `"complete"`,
#'   `"partial"`, `"refusal"`, `"noncontact"` or `"ineligible"`.
#' @param digits Number of decimal places in the shares.
#' @return A data frame with columns `outcome`, `n` and `share`, with one
#'   row each for `complete`, `partial`, `refusal`, `noncontact` and
#'   `ineligible`.
#' @export
#' @family formatters
formatOutcomeCounts = function(outcomes, digits = 1) {
  outcomes = as.character(outcomes)
  known = c("complete", "partial", "refusal", "noncontact", "ineligible")
  unknown = setdiff(unique(outcomes[!is.na(outcomes)]), known)
  if (length(unknown) > 0L) {
    stop("Unknown outcomes: ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  total = sum(!is.na(outcomes))
  rows = list()

  # Count `complete`
  n = sum(outcomes == "complete", na.rm = TRUE)
  rows$complete = data.frame(outcome = "complete", n = n,
                             share = formatRate(n / total, digits = digits))

  # Count `partial`
  n = sum(outcomes == "partial", na.rm = TRUE)
  rows$partial = data.frame(outcome = "partial", n = n,
                            share = formatRate(n / total, digits = digits))

  # Count `refusal`
  n = sum(outcomes == "refusal", na.rm = TRUE)
  rows$refusal = data.frame(outcome = "refusal", n = n,
                            share = formatRate(n / total, digits = digits))

  # Count `noncontact`
  n = sum(outcomes == "noncontact", na.rm = TRUE)
  rows$noncontact = data.frame(outcome = "noncontact", n = n,
                               share = formatRate(n / total, digits = digits))

  # Count `ineligible`
  n = sum(outcomes == "ineligible", na.rm = TRUE)
  rows$ineligible = data.frame(outcome = "ineligible", n = n,
                               share = formatRate(n / total, digits = digits))
  result = do.call(rbind, rows)
  rownames(result) = NULL
  return(result)
}
