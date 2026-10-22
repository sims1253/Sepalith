#' Classify the lines of a NEWS.md file
#'
#' @description Indented lines that are not bullets count as continuations of
#'   the bullet above them.
#' @param lines A character vector of NEWS.md lines.
#' @return A character vector the same length as `lines`, with values
#'   `"version"`, `"section"`, `"bullet"`, `"continuation"` or `"other"`.
#' @keywords internal
news_line_kind <- function(lines) {
  kind <- rep("other", length(lines))
  # Later rules win, so the order of these assignments matters.
  kind[grepl("^#\\s+\\S", lines)] <- "version"
  kind[grepl("^##+\\s+\\S", lines)] <- "section"
  kind[grepl("^\\s{0,3}[*-]\\s+", lines)] <- "bullet"
  kind[grepl("^\\s{2,}\\S", lines) & kind == "other"] <- "continuation"
  kind
}

#' Split a NEWS.md version heading into package, version and date
#'
#' @description Understands headings such as `# mypkg 1.2.0`,
#'   `# mypkg 1.2.0 (2026-03-01)` and `# mypkg (development version)`.
#' @param heading A single heading line, including the leading `#`.
#' @return A list with elements `package`, `version` and `date`; `version` is
#'   `NA` when the heading carries none.
#' @keywords internal
news_parse_heading <- function(heading) {
  text <- sub("^#\\s+", "", heading) |> trimws()

  # Pull out an ISO date first, then drop the parenthesised part holding it.
  date <- regmatches(text, regexpr("[0-9]{4}-[0-9]{2}-[0-9]{2}", text))
  text <- sub("\\s*\\([^)]*[0-9]{4}-[0-9]{2}-[0-9]{2}[^)]*\\)\\s*$", "", text)
  if (grepl("development version", text, fixed = TRUE)) {
    version <- "dev"
    package <- sub("\\s*\\(development version\\)$", "", text)
  } else if (grepl("[0-9]+(\\.[0-9]+)+$", text)) {
    version <- sub("^.*?\\s*v?([0-9]+(\\.[0-9]+)+)$", "\\1", text, perl = TRUE)
    package <- sub("\\s*v?[0-9]+(\\.[0-9]+)+$", "", text)
  } else {
    # A heading without any version, e.g. "# mypkg".
    version <- NA_character_
    package <- text
  }

  list(
    package = if (nzchar(package)) package else NA_character_,
    version = version,
    date = if (length(date) == 1L) as.Date(date) else as.Date(NA)
  )
}

#' Parse NEWS.md into a tidy table of entries
#'
#' Each bullet becomes one row. Continuation lines are folded into the
#' preceding bullet; bullets that appear before any `##` heading get the
#' section `NA`.
#'
#' @param lines A character vector of NEWS.md lines.
#' @return A data frame with columns `version`, `date`, `section` and `item`.
#' @export
news_parse <- function(lines) {
  kind <- news_line_kind(lines)
  n_bullets <- sum(kind == "bullet")
  version <- section <- item <- character(n_bullets)
  date <- rep(as.Date(NA), n_bullets)

  # State carried from the most recent headings while walking the lines.
  cur_version <- NA_character_
  cur_date <- as.Date(NA)
  cur_section <- NA_character_
  k <- 0L

  for (i in seq_along(lines)) {
    if (kind[[i]] == "version") {
      head <- news_parse_heading(lines[[i]])
      cur_version <- head$version
      cur_date <- head$date
      cur_section <- NA_character_
    } else if (kind[[i]] == "section") {
      cur_section <- sub("^##+\\s+", "", lines[[i]]) |> trimws()
    } else if (kind[[i]] == "bullet") {
      k <- k + 1L
      version[[k]] <- cur_version
      date[[k]] <- cur_date
      section[[k]] <- cur_section
      item[[k]] <- sub("^\\s{0,3}[*-]\\s+", "", lines[[i]])
    } else if (kind[[i]] == "continuation" && k > 0L) {
      item[[k]] <- paste(item[[k]], trimws(lines[[i]]))
    }
  }

  data.frame(
    version = version,
    date = date,
    section = section,
    item = item,
    stringsAsFactors = FALSE
  )
}

#' Read and parse a NEWS.md file
#'
#' @param path Path to a NEWS.md file.
#' @return A data frame with columns `version`, `date`, `section` and `item`.
#' @export
news_read <- function(path = "NEWS.md") {
  if (!file.exists(path)) {
    stop("Cannot find '", path, "'.", call. = FALSE)
  }
  readLines(path, warn = FALSE, encoding = "UTF-8") |>
    news_parse()
}

#' List the versions mentioned in a NEWS.md file
#'
#' @param path Path to a NEWS.md file.
#' @return A character vector of versions, newest first as written.
#' @export
news_versions <- function(path = "NEWS.md") {
  lines <- readLines(path, warn = FALSE)
  headings <- lines[news_line_kind(lines) == "version"]
  vapply(headings, \(h) news_parse_heading(h)$version, character(1)) |>
    unname()
}

#' Get the entries for one version of a NEWS.md file
#'
#' @param version The version to extract, e.g. `"1.2.0"` or `"dev"`.
#' @param path Path to a NEWS.md file.
#' @return A data frame with columns `version`, `date`, `section` and `item`.
#' @export
news_entries_for <- function(version, path = "NEWS.md") {
  news <- news_read(path)
  if (!version %in% news$version) {
    stop("Version '", version, "' is not in '", path, "'.", call. = FALSE)
  }
  news[news$version == version, , drop = FALSE]
}

#' Get the entries for the most recent release in a NEWS.md file
#'
#' Development headings are skipped.
#'
#' @param path Path to a NEWS.md file.
#' @return A data frame with columns `version`, `date`, `section` and `item`.
#' @export
news_latest_release <- function(path = "NEWS.md") {
  releases <- setdiff(news_versions(path), c("dev", NA_character_))
  if (length(releases) == 0L) {
    stop("'", path, "' does not mention any released version.", call. = FALSE)
  }

  # Compare as versions: "1.10.0" is newer than "1.9.0".
  latest <- releases[order(numeric_version(releases), decreasing = TRUE)][[1L]]
  news_entries_for(latest, path = path)
}

#' Add a bullet to the top version of a NEWS.md file
#'
#' The bullet is inserted under the requested `##` section of the first
#' version heading, creating the section if needed.
#'
#' @param text The bullet text, without the leading `*`.
#' @param section The `##` section to add the bullet to.
#' @param path Path to a NEWS.md file.
#' @return `path`, invisibly.
#' @export
news_add_bullet <- function(text, section = "Bug fixes", path = "NEWS.md") {
  lines <- readLines(path, warn = FALSE)
  kind <- news_line_kind(lines)
  tops <- which(kind == "version")
  if (length(tops) == 0L) {
    stop("'", path, "' has no version heading.", call. = FALSE)
  }

  # Only look for the section inside the first (newest) version block.
  block_end <- if (length(tops) > 1L) tops[[2L]] - 1L else length(lines)
  in_block <- seq_along(lines) > tops[[1L]] & seq_along(lines) <= block_end
  heading_text <- trimws(sub("^##+\\s+", "", lines))
  target <- which(in_block & kind == "section" & heading_text == section)
  bullet <- paste("*", text)

  if (length(target) == 0L) {
    insert_at <- tops[[1L]]
    new_lines <- c("", paste("##", section), "", bullet)
  } else {
    insert_at <- target[[1L]]
    # Keep the blank line that follows the section heading.
    if (insert_at < length(lines) && lines[[insert_at + 1L]] == "") {
      insert_at <- insert_at + 1L
    }
    new_lines <- bullet
  }

  lines <- append(lines, new_lines, after = insert_at)
  writeLines(lines, path)
  invisible(path)
}
