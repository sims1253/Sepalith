#' Render an HTTP status badge
#'
#' Produces a small coloured label whose class reflects the status family,
#' for use in crawl and link-check reports.
#'
#' @param status Integer HTTP status code, or `NA` for failed requests.
#' @return An `htmltools` tag.
#' @importFrom htmltools tags
#' @export
status_badge = function(status) {
  family = if (is.na(status)) "error" else switch(
    as.character(status %/% 100),
    "2" = "ok",
    "3" = "redirect",
    "4" = "client",
    "5" = "server",
    "error"
  )

  label = if (is.na(status)) "failed" else as.character(status)
  tags$span(class = paste("qw-badge", paste0("qw-badge-", family)), label)
}

#' Render a metric card
#'
#' Shows a single headline number with a label and, optionally, the change
#' compared with a previous run.
#'
#' @param label Short text describing the metric.
#' @param value The current value.
#' @param previous Optional value from the previous run.
#' @return An `htmltools` tag.
#' @importFrom htmltools tags
#' @export
metric_card = function(label, value, previous = NULL) {
  # The delta is only shown when there is something to compare against.
  delta = NULL
  if (!is.null(previous) && !is.na(previous)) {
    change = value - previous
    direction = if (change > 0) "up" else if (change < 0) "down" else "flat"
    delta = tags$span(
      class = paste("qw-delta", paste0("qw-delta-", direction)),
      sprintf("%+g", change)
    )
  }

  tags$div(
    class = "qw-card",
    tags$div(class = "qw-card-label", label),
    tags$div(class = "qw-card-value", format(value, big.mark = ",")),
    delta
  )
}

#' Render a table of checked links
#'
#' Builds an HTML table with one row per URL, its status badge and the
#' response time, linking every URL to its target.
#'
#' @param results A data frame with columns `url`, `status` and `elapsed`.
#' @param max_rows Maximum number of rows to show.
#' @return An `htmltools` tag.
#' @importFrom htmltools tags tagList
#' @export
link_table = function(results, max_rows = 200L) {
  missing_cols = setdiff(c("url", "status", "elapsed"), names(results))
  if (length(missing_cols) > 0) {
    stop("`results` is missing columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  shown = utils::head(results, max_rows)
  rows = vector("list", nrow(shown))
  for (i in seq_len(nrow(shown))) {
    rows[[i]] = tags$tr(
      tags$td(tags$a(href = shown$url[i], shown$url[i])),
      tags$td(status_badge(shown$status[i])),
      tags$td(sprintf("%.0f ms", shown$elapsed[i] * 1000))
    )
  }

  # Say how much was cut off rather than truncating silently.
  footer = NULL
  if (nrow(results) > max_rows) {
    hidden = nrow(results) - max_rows
    footer = tags$p(class = "qw-note", sprintf("%d more rows not shown.", hidden))
  }

  tagList(
    tags$table(
      class = "qw-table",
      tags$thead(tags$tr(tags$th("URL"), tags$th("Status"), tags$th("Time"))),
      tags$tbody(rows)
    ),
    footer
  )
}

#' Summarise a crawl as a row of metric cards
#'
#' Counts successful, redirected and failed requests and shows them next to
#' the totals of a previous crawl when one is supplied.
#'
#' @param results A data frame with columns `url`, `status` and `elapsed`.
#' @param previous Optional data frame of the same shape from an earlier run.
#' @return An `htmltools` tag.
#' @importFrom htmltools tags
#' @export
crawl_summary = function(results, previous = NULL) {
  count_by = function(df) {
    if (is.null(df)) {
      return(list(total = NA, ok = NA, redirect = NA, failed = NA))
    }
    # A missing status means the request itself failed.
    family = df$status %/% 100
    list(
      total = nrow(df),
      ok = sum(family == 2, na.rm = TRUE),
      redirect = sum(family == 3, na.rm = TRUE),
      failed = sum(is.na(family) | family >= 4)
    )
  }

  now = count_by(results)
  before = count_by(previous)
  tags$div(
    class = "qw-summary",
    metric_card("URLs checked", now$total, before$total),
    metric_card("OK", now$ok, before$ok),
    metric_card("Redirects", now$redirect, before$redirect),
    metric_card("Failed", now$failed, before$failed)
  )
}

#' Assemble a complete report page
#'
#' Wraps report snippets in a minimal HTML page with an inline stylesheet
#' and a generation timestamp.
#'
#' @param title Page title, also used as the main heading.
#' @param ... Tags to place in the page body.
#' @param css Character string with extra CSS rules.
#' @return An `htmltools` tag list representing the whole page.
#' @importFrom htmltools tags tagList HTML
#' @export
report_page = function(title, ..., css = "") {
  base_css = paste(
    "body { font-family: sans-serif; margin: 2rem; color: #222; }",
    ".qw-summary { display: flex; gap: 1rem; margin-bottom: 1.5rem; }",
    ".qw-card { border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem 1rem; }",
    ".qw-card-value { font-size: 1.6rem; font-weight: bold; }",
    ".qw-badge { padding: 0 0.4rem; border-radius: 4px; color: #fff; }",
    ".qw-badge-ok { background: #2e7d32; } .qw-badge-redirect { background: #1565c0; }",
    ".qw-badge-client, .qw-badge-server, .qw-badge-error { background: #c62828; }",
    sep = "\n"
  )

  tags$html(
    tags$head(
      tags$meta(charset = "utf-8"),
      tags$title(title),
      tags$style(HTML(paste(base_css, css, sep = "\n")))
    ),
    tags$body(
      tags$h1(title),
      tags$p(class = "qw-note", paste("Generated", format(Sys.time(), "%Y-%m-%d %H:%M"))),
      tagList(...)
    )
  )
}

#' Write a report page to disk
#'
#' Renders the page and saves it as a standalone HTML file.
#'
#' @param page A page built with [report_page()].
#' @param path Destination file path.
#' @return The path, invisibly.
#' @importFrom htmltools save_html
#' @export
save_report = function(page, path) {
  # Make sure the target directory exists before writing.
  dir = dirname(path)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  save_html(page, file = path)
  invisible(path)
}
