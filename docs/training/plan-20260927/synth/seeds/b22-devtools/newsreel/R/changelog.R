#' @importFrom magrittr %>%
NULL

#' Parse conventional-commit subject lines
#'
#' Splits subjects of the form `type(scope)!: description` into their parts.
#' Subjects that do not follow the convention get type `"other"`.
#'
#' @param subject A character vector of commit subject lines.
#' @return A tibble with columns `type`, `scope`, `breaking` and
#'   `description`.
#' @export
parse_commit_subject <- function(subject) {
  pattern <- "^([a-z]+)(\\(([^)]+)\\))?(!)?:\\s*(.+)$"
  m <- stringr::str_match(subject, pattern)
  conventional <- !is.na(m[, 1])
  tibble::tibble(
    type = dplyr::if_else(conventional, m[, 2], "other"),
    scope = m[, 4],
    breaking = conventional & !is.na(m[, 5]),
    description = dplyr::if_else(conventional, m[, 6], subject)
  )
}

#' Map a commit type to a changelog section
#'
#' @param type A single conventional-commit type.
#' @return The section heading the commit belongs under, or `NA` for types
#'   that are left out of the changelog.
#' @keywords internal
commit_section <- function(type) {
  switch(type,
    feat = "New features",
    fix = "Bug fixes",
    perf = "Performance",
    docs = "Documentation",
    refactor = ,
    style = "Internal changes",
    deprecate = "Deprecations",
    test = ,
    ci = ,
    chore = NA_character_,
    "Other changes"
  )
}

#' Group parsed commits into changelog sections
#'
#' Breaking changes are pulled into their own section regardless of type.
#'
#' @param commits A data frame of commits with at least a `subject` column.
#' @return A tibble with columns `section`, `scope` and `description`,
#'   ordered by section.
#' @export
changelog_group <- function(commits) {
  order <- c(
    "Breaking changes", "New features", "Bug fixes", "Performance",
    "Deprecations", "Documentation", "Internal changes", "Other changes"
  )
  parse_commit_subject(commits$subject) %>%
    dplyr::mutate(
      section = purrr::map_chr(type, commit_section),
      section = dplyr::if_else(breaking, "Breaking changes", section)
    ) %>%
    dplyr::filter(!is.na(section)) %>%
    dplyr::mutate(section = factor(section, levels = order)) %>%
    dplyr::arrange(section, scope) %>%
    dplyr::select(section, scope, description)
}

#' Render grouped commits as Markdown
#'
#' @param grouped A tibble of grouped commits as returned by [changelog_group()].
#' @param heading The top-level heading, e.g. `"mypkg 1.2.0"`.
#' @return A character vector of Markdown lines.
#' @family changelog renderers
#' @export
changelog_render_markdown <- function(grouped, heading) {
  out <- c(paste("#", heading), "")
  for (sec in levels(droplevels(grouped$section))) {
    rows <- grouped %>% dplyr::filter(section == sec)
    bullets <- dplyr::if_else(
      is.na(rows$scope),
      paste0("* ", rows$description),
      paste0("* **", rows$scope, "**: ", rows$description)
    )
    out <- c(out, paste("##", sec), "", bullets, "")
  }
  out
}

#' Render grouped commits as plain text
#'
#' @param grouped A tibble of grouped commits as returned by [changelog_group()].
#' @param heading The top-level heading, e.g. `"mypkg 1.2.0"`.
#' @return A character vector of plain-text lines.
#' @family changelog renderers
#' @export
changelog_render_plain <- function(grouped, heading) {
  out <- c(heading, strrep("=", nchar(heading)), "")
  for (sec in levels(droplevels(grouped$section))) {
    rows <- grouped %>% dplyr::filter(section == sec)
    bullets <- dplyr::if_else(
      is.na(rows$scope),
      paste0("  - ", rows$description),
      paste0("  - [", rows$scope, "] ", rows$description)
    )
    out <- c(out, sec, strrep("-", nchar(sec)), bullets, "")
  }
  out
}

#' Suggest the next version from a set of commits
#'
#' Any breaking change bumps the major version, any feature bumps the minor
#' version, and anything else bumps the patch version.
#'
#' @param commits A data frame of commits with at least a `subject` column.
#' @param current The current version string.
#' @return The suggested next version string.
#' @export
next_version_from_commits <- function(commits, current) {
  parsed <- parse_commit_subject(commits$subject)
  v <- unclass(numeric_version(current))[[1]]
  length(v) <- 3
  v[is.na(v)] <- 0L
  if (any(parsed$breaking)) {
    v <- c(v[1] + 1L, 0L, 0L)
  } else if (any(parsed$type == "feat")) {
    v <- c(v[1], v[2] + 1L, 0L)
  } else {
    v <- c(v[1], v[2], v[3] + 1L)
  }
  paste(v, collapse = ".")
}

#' Build a changelog from commits
#'
#' @param commits A data frame of commits with at least a `subject` column.
#' @param heading The top-level heading, e.g. `"mypkg 1.2.0"`.
#' @param format Output format.
#' @return A character vector of lines.
#' @export
changelog_from_commits <- function(commits, heading, format = c("markdown", "plain")) {
  format <- match.arg(format)
  grouped <- changelog_group(commits)
  if (nrow(grouped) == 0L) {
    rlang::inform("No user-facing commits found.")
  }
  switch(format,
    markdown = changelog_render_markdown(grouped, heading),
    plain = changelog_render_plain(grouped, heading)
  )
}
