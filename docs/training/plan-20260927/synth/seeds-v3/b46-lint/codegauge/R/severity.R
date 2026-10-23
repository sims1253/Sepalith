#' Count lints per severity
#'
#' @description
#' Summarises a lint table with one row per severity level, most severe first.
#' @param lints A data frame of lints with columns `line`, `column`, `rule`,
#'   `severity` and `message`.
#' @return A data frame with columns `severity` (`error`, `warning`, `style`,
#'   `info`), `n`, `lines` (distinct lines) and `rules` (distinct rules,
#'   comma separated).
#' @family severity reports
#' @export
severityCounts = function(lints) {
  levels = c("error", "warning", "style", "info")
  unknown = setdiff(unique(lints$severity), levels)
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown severity: %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }

  # Errors
  hit = lints$severity == "error"
  errorRow = data.frame(severity = "error", n = sum(hit),
    lines = length(unique(lints$line[hit])),
    rules = paste(sort(unique(lints$rule[hit])), collapse = ", "))
  # Warnings
  hit = lints$severity == "warning"
  warningRow = data.frame(severity = "warning", n = sum(hit),
    lines = length(unique(lints$line[hit])),
    rules = paste(sort(unique(lints$rule[hit])), collapse = ", "))
  # Style
  hit = lints$severity == "style"
  styleRow = data.frame(severity = "style", n = sum(hit),
    lines = length(unique(lints$line[hit])),
    rules = paste(sort(unique(lints$rule[hit])), collapse = ", "))
  # Info
  hit = lints$severity == "info"
  infoRow = data.frame(severity = "info", n = sum(hit),
    lines = length(unique(lints$line[hit])),
    rules = paste(sort(unique(lints$rule[hit])), collapse = ", "))

  rbind(errorRow, warningRow, styleRow, infoRow)
}

#' Format lints for a consumer
#'
#' @description
#' Renders a lint table as plain text, GitHub annotations, Checkstyle XML,
#' a Markdown table or CSV. Each format maps the severities onto its own
#' levels.
#' @param lints A data frame of lints with columns `line`, `column`, `rule`,
#'   `severity` and `message`.
#' @param file The file name to report the lints against.
#' @param format The output format.
#' @return A character vector of output lines.
#' @family severity reports
#' @export
formatLints = function(lints, file, format = c("plain", "github", "checkstyle", "markdown",
                                               "csv")) {
  format = match.arg(format)
  levels = c("error", "warning", "style", "info")
  severity = factor(lints$severity, levels = levels)

  switch(format,
    plain = {
      level = toupper(substr(levels, 1L, 1L))[severity]
      sprintf("%s:%d:%d: %s %s [%s]", file, lints$line, lints$column, level,
        lints$message, lints$rule)
    },
    github = {
      level = c("error", "warning", "notice", "notice")[severity]
      sprintf("::%s file=%s,line=%d,col=%d::%s [%s]", level, file, lints$line,
        lints$column, lints$message, lints$rule)
    },
    checkstyle = {
      level = c("error", "warning", "info", "info")[severity]
      body = sprintf(paste0("    <error line=\"%d\" column=\"%d\" severity=\"%s\"",
        " message=\"%s\" source=\"%s\"/>"), lints$line, lints$column, level,
        lints$message, lints$rule)
      c("<checkstyle>", sprintf("  <file name=\"%s\">", file), body, "  </file>", "</checkstyle>")
    },
    markdown = {
      level = levels[severity]
      body = sprintf("| %d:%d | %s | %s | %s |", lints$line, lints$column, level,
        lints$rule, lints$message)
      c(sprintf("**%s**", file), "", "| Position | Severity | Rule | Message |",
        "|---|---|---|---|", body)
    },
    csv = {
      level = levels[severity]
      body = sprintf("%s,%d,%d,%s,%s,\"%s\"", file, lints$line, lints$column, level,
        lints$rule, gsub("\"", "\"\"", lints$message))
      c("file,line,column,severity,rule,message", body)
    }
  )
}

#' Write a lint report grouped by severity
#'
#' @description
#' Lists the lints in one section per severity, skipping empty sections.
#' @param lints A data frame of lints with columns `line`, `column`, `rule`,
#'   `severity` and `message`.
#' @param path A file to write the report to, or `NULL` to only return it.
#' @return The report lines; invisibly when written to `path`.
#' @family severity reports
#' @export
reportBySeverity = function(lints, path = NULL) {
  labels = c(error = "Errors", warning = "Warnings", style = "Style", info = "Notes")
  lints = lints[order(lints$line, lints$column), , drop = FALSE]
  out = sprintf("%d lints", nrow(lints))

  # Errors
  rows = lints[lints$severity == "error", , drop = FALSE]
  if (nrow(rows) > 0L) {
    out = c(out, "", sprintf("%s (%d)", labels[["error"]], nrow(rows)),
      sprintf("  %d:%d %s [%s]", rows$line, rows$column, rows$message, rows$rule))
  }
  # Warnings
  rows = lints[lints$severity == "warning", , drop = FALSE]
  if (nrow(rows) > 0L) {
    out = c(out, "", sprintf("%s (%d)", labels[["warning"]], nrow(rows)),
      sprintf("  %d:%d %s [%s]", rows$line, rows$column, rows$message, rows$rule))
  }
  # Style
  rows = lints[lints$severity == "style", , drop = FALSE]
  if (nrow(rows) > 0L) {
    out = c(out, "", sprintf("%s (%d)", labels[["style"]], nrow(rows)),
      sprintf("  %d:%d %s [%s]", rows$line, rows$column, rows$message, rows$rule))
  }
  # Info
  rows = lints[lints$severity == "info", , drop = FALSE]
  if (nrow(rows) > 0L) {
    out = c(out, "", sprintf("%s (%d)", labels[["info"]], nrow(rows)),
      sprintf("  %d:%d %s [%s]", rows$line, rows$column, rows$message, rows$rule))
  }

  if (is.null(path)) {
    return(out)
  }
  writeLines(out, path)
  invisible(out)
}

#' Compare lint results before and after a change
#'
#' @description
#' Counts the lints per severity in both runs and lists the rules that were
#' fixed or introduced.
#' @param before,after Data frames of lints with columns `line`, `column`,
#'   `rule`, `severity` and `message`.
#' @return A list with elements `table` (columns `severity`, `before`,
#'   `after`, `change` and `trend`), `fixed` and `introduced`.
#' @family severity reports
#' @export
compareSeverity = function(before, after) {
  levels = c("error", "warning", "style", "info")
  old = stats::setNames(integer(4), levels)
  new = stats::setNames(integer(4), levels)

  # Count the lints before the change
  for (level in levels) {
    old[[level]] = sum(before$severity == level)
  }
  # Count the lints after the change
  for (level in levels) {
    new[[level]] = sum(after$severity == level)
  }

  change = new - old
  trend = ifelse(change < 0L, "better", ifelse(change > 0L, "worse", "same"))
  table = data.frame(
    severity = levels, before = unname(old), after = unname(new),
    change = unname(change), trend = unname(trend), stringsAsFactors = FALSE
  )
  list(
    table = table,
    fixed = sort(setdiff(before$rule, after$rule)),
    introduced = sort(setdiff(after$rule, before$rule))
  )
}

#' Decide whether a lint run passes
#'
#' @description
#' Every severity at or above `failOn` is checked against its budget.
#' @param lints A data frame of lints with columns `line`, `column`, `rule`,
#'   `severity` and `message`.
#' @param failOn The least severe level that can fail the run.
#' @param budget A named numeric vector with the number of lints allowed per
#'   severity: `error`, `warning`, `style` and `info`.
#' @return A list with elements `pass`, `status` (0 or 1) and `reasons`.
#' @family severity reports
#' @export
lintVerdict = function(lints, failOn = c("error", "warning", "style", "info"),
                       budget = c(error = 0, warning = 0, style = 0, info = 0)) {
  failOn = match.arg(failOn)
  levels = c("error", "warning", "style", "info")
  checked = levels[seq_len(match(failOn, levels))]
  reasons = character()

  # Check `error`
  n = sum(lints$severity == "error")
  if ("error" %in% checked && n > budget[["error"]]) {
    reasons = c(reasons, sprintf("%d %s lints, budget %g", n, "error", budget[["error"]]))
  }
  # Check `warning`
  n = sum(lints$severity == "warning")
  if ("warning" %in% checked && n > budget[["warning"]]) {
    reasons = c(reasons, sprintf("%d %s lints, budget %g", n, "warning", budget[["warning"]]))
  }
  # Check `style`
  n = sum(lints$severity == "style")
  if ("style" %in% checked && n > budget[["style"]]) {
    reasons = c(reasons, sprintf("%d %s lints, budget %g", n, "style", budget[["style"]]))
  }
  # Check `info`
  n = sum(lints$severity == "info")
  if ("info" %in% checked && n > budget[["info"]]) {
    reasons = c(reasons, sprintf("%d %s lints, budget %g", n, "info", budget[["info"]]))
  }

  pass = length(reasons) == 0L
  list(pass = pass, status = if (pass) 0L else 1L, reasons = reasons)
}
