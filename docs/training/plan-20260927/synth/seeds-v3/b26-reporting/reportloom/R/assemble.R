#' Create an empty report
#'
#' @export
#' @param title A single string used as the report title.
#' @param author A single string naming the author.
#' @param date A single `Date` shown under the title.
#' @param subtitle An optional single string shown under the title.
#' @param keywords A character vector of keywords for the metadata.
#' @param sections A list of `rl_section` objects.
#' @return An object of class `rl_report` with elements `title`, `author`,
#'   `date`, `subtitle`, `keywords` and `sections`.
new.report <- function(title, author, date = Sys.Date(), subtitle = NULL,
                       keywords = character(), sections = list()) {
  # Check `title`
  if (!is.character(title) || length(title) != 1L || is.na(title)) {
    stop("`title` must be a single string.", call. = FALSE)
  }
  # Check `author`
  if (!is.character(author) || length(author) != 1L || is.na(author)) {
    stop("`author` must be a single string.", call. = FALSE)
  }
  # Check `date`
  if (!inherits(date, "Date") || length(date) != 1L || is.na(date)) {
    stop("`date` must be a single Date.", call. = FALSE)
  }
  # Check `subtitle`
  if (!is.null(subtitle) && (!is.character(subtitle) || length(subtitle) != 1L)) {
    stop("`subtitle` must be NULL or a single string.", call. = FALSE)
  }
  # Check `keywords`
  if (!is.character(keywords) || anyNA(keywords)) {
    stop("`keywords` must be a character vector without missing values.", call. = FALSE)
  }
  # Check `sections`
  if (!is.list(sections) || !all(vapply(sections, inherits, logical(1), "rl_section"))) {
    stop("`sections` must be a list of rl_section objects.", call. = FALSE)
  }

  structure(
    list(
      title = title,
      author = author,
      date = date,
      subtitle = subtitle,
      keywords = unique(keywords),
      sections = sections
    ),
    class = "rl_report"
  )
}

#' Add a section to a report
#'
#' @export
#' @param report An `rl_report` object created by [new.report()].
#' @param section An `rl_section` object.
#' @param after Position after which to insert `section`; `NULL` appends.
#' @return The modified `report`.
report.add <- function(report, section, after = NULL) {
  stopifnot(inherits(report, "rl_report"), inherits(section, "rl_section"))
  if (is.null(after)) {
    after <- length(report$sections)
  }
  report$sections <- append(report$sections, list(section), after = after)
  report
}

#' Find structural problems in a report
#'
#' @description
#' Checks, in order: an empty `title`, an empty `author`, no sections,
#' duplicated section ids, skipped heading levels and empty sections.
#'
#' @export
#' @param report An `rl_report` object created by [new.report()].
#' @return A character vector with one message per problem; empty when the
#'   report is fine.
report.problems <- function(report) {
  stopifnot(inherits(report, "rl_report"))
  problems <- character()
  # Empty `title`
  if (!nzchar(trimws(report$title))) {
    problems <- c(problems, "The report has an empty title.")
  }
  # Empty `author`
  if (!nzchar(trimws(report$author))) {
    problems <- c(problems, "The report has an empty author.")
  }
  # No sections
  if (length(report$sections) == 0L) {
    return(c(problems, "The report has no sections."))
  }

  ids <- vapply(report$sections, `[[`, character(1), "id")
  levels <- vapply(report$sections, `[[`, integer(1), "level")
  sizes <- vapply(report$sections, function(section) length(section$blocks), integer(1))
  # Duplicated section ids
  duplicated.ids <- unique(ids[duplicated(ids)])
  if (length(duplicated.ids) > 0L) {
    problems <- c(problems, sprintf("Section `%s` is used more than once.", duplicated.ids))
  }
  # Skipped heading levels
  jumps <- which(diff(c(1L, levels)) > 1L)
  if (length(jumps) > 0L) {
    problems <- c(problems, sprintf("Section `%s` skips a heading level.", ids[jumps]))
  }
  # Empty sections
  empty <- ids[sizes == 0L]
  if (length(empty) > 0L) {
    problems <- c(problems, sprintf("Section `%s` has no content.", empty))
  }

  problems
}

#' Opening lines of a rendered report
#'
#' @export
#' @param report An `rl_report` object created by [new.report()].
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"text"`.
#' @return A character vector with one element per output line.
front.matter <- function(report, format = c("markdown", "html", "latex", "text")) {
  format <- match.arg(format)
  date <- format(report$date, "%Y-%m-%d")
  switch(format,
    markdown = c(
      "---",
      sprintf("title: \"%s\"", report$title),
      sprintf("author: \"%s\"", report$author),
      sprintf("date: \"%s\"", date),
      "---"
    ),
    html = c(
      "<!DOCTYPE html>",
      "<html>",
      sprintf("<head><title>%s</title></head>", report$title),
      "<body>",
      sprintf("<p class=\"byline\">%s, %s</p>", report$author, date)
    ),
    latex = c(
      "\\documentclass{article}",
      sprintf("\\title{%s}", report$title),
      sprintf("\\author{%s}", report$author),
      sprintf("\\date{%s}", date),
      "\\begin{document}",
      "\\maketitle"
    ),
    text = c(
      report$title,
      strrep("=", nchar(report$title)),
      sprintf("%s, %s", report$author, date)
    )
  )
}

#' Closing lines of a rendered report
#'
#' @export
#' @inheritParams front.matter
#' @param note An optional single string printed at the end.
#' @return A character vector with one element per output line.
back.matter <- function(report, format = c("markdown", "html", "latex", "text"), note = NULL) {
  format <- match.arg(format)
  if (format == "html" && !is.null(note)) {
    note <- sprintf("<p class=\"note\">%s</p>", note)
  }
  closing <- switch(format,
    markdown = c("---", note),
    html = c(note, "</body>", "</html>"),
    latex = c(note, "\\end{document}"),
    text = c(strrep("-", nchar(report$title)), note)
  )
  closing
}

#' Table of contents
#'
#' @description
#' One entry per section, indented by two spaces for every heading level
#' below the first.
#'
#' @export
#' @param report An `rl_report` object created by [new.report()].
#' @param format One of `"markdown"`, `"html"` or `"text"`.
#' @return A character vector with one element per output line.
report.toc <- function(report, format = c("markdown", "html", "text")) {
  format <- match.arg(format)
  entries <- character(length(report$sections))
  for (i in seq_along(report$sections)) {
    section <- report$sections[[i]]
    indent <- strrep("  ", section$level - 1L)
    entries[i] <- switch(format,
      markdown = sprintf("%s- [%s](#%s)", indent, section$title, section$id),
      html = sprintf("%s<li><a href=\"#%s\">%s</a></li>", indent, section$id, section$title),
      text = sprintf("%s%d. %s", indent, i, section$title)
    )
  }

  if (format == "html") {
    entries <- c("<ul class=\"toc\">", entries, "</ul>")
  }

  entries
}

#' Render a whole report
#'
#' @description
#' Steps: front matter, table of contents, every section in order, back
#' matter. LaTeX output builds its own contents with `\tableofcontents`.
#'
#' @export
#' @param report An `rl_report` object created by [new.report()].
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"text"`.
#' @param toc Whether to include a table of contents.
#' @param note An optional single string printed at the end.
#' @return A character vector with one element per output line.
render.report <- function(report, format = c("markdown", "html", "latex", "text"),
                          toc = TRUE, note = NULL) {
  format <- match.arg(format)
  stopifnot(inherits(report, "rl_report"))
  # 1. Front matter
  lines <- front.matter(report, format)

  # 2. Table of contents
  if (toc && length(report$sections) > 0L) {
    if (format == "latex") {
      lines <- c(lines, "\\tableofcontents")
    } else {
      lines <- c(lines, "", report.toc(report, format))
    }
  }

  # 3. Every section in order
  for (section in report$sections) {
    lines <- c(lines, "", render.section(section, format))
  }

  # 4. Back matter
  c(lines, "", back.matter(report, format, note))
}

#' Write a report in several formats
#'
#' @export
#' @param report An `rl_report` object created by [new.report()].
#' @param dir Directory to write into; created when missing.
#' @param formats Formats to write, any of `"markdown"`, `"html"`, `"latex"`
#'   and `"text"`.
#' @param stem File name without extension; derived from the title when
#'   `NULL`.
#' @return A named character vector of written paths, invisibly.
write.report <- function(report, dir, formats = c("markdown", "html"), stem = NULL) {
  extensions <- c(markdown = "md", html = "html", latex = "tex", text = "txt")
  unknown <- setdiff(formats, names(extensions))
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown format(s): %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }

  if (is.null(stem)) {
    stem <- gsub("[^A-Za-z0-9]+", "_", tolower(report$title))
  }

  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  paths <- character(length(formats))
  names(paths) <- formats
  for (format in formats) {
    paths[[format]] <- file.path(dir, paste0(stem, ".", extensions[[format]]))
    writeLines(render.report(report, format), paths[[format]])
  }

  invisible(paths)
}
