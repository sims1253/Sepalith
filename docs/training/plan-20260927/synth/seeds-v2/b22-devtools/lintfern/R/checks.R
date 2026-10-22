#' Build a table of lint hits
#'
#' @param line Integer line numbers of the hits.
#' @param column Integer column numbers of the hits.
#' @param message Character messages, recycled to the number of hits.
#' @return A data frame with columns `line`, `column` and `message`.
#' @keywords internal
lintHits <- function(line = integer(), column = integer(), message = character()) {
  n <- length(line)
  if (length(column) != n) {
    stop("`line` and `column` must have the same length.", call. = FALSE)
  }

  data.frame(
    line = as.integer(line),
    column = as.integer(column),
    message = rep_len(as.character(message), n),
    stringsAsFactors = FALSE
  )
}

#' Find function arguments that are never used
#'
#' An argument counts as used if its name appears anywhere in the function
#' body. Functions whose body forwards `...` are not exempt: unused named
#' arguments are still reported.
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @return A data frame of lint hits as returned by [lintHits()].
#' @export
checkUnusedArguments <- function(pd) {
  defs <- findFunctionDefs(pd)
  hits <- list()

  for (i in seq_len(nrow(defs))) {
    formals <- functionFormals(pd, defs$id[[i]])
    used <- functionBodySymbols(pd, defs$id[[i]])
    unused <- formals[!formals$text %in% c(used, "..."), , drop = FALSE]
    if (nrow(unused) == 0L) {
      next
    }

    label <- if (is.na(defs$name[[i]])) "anonymous function" else defs$name[[i]]
    hits[[length(hits) + 1L]] <- lintHits(
      unused$line1,
      unused$col1,
      sprintf("Argument '%s' of %s is never used.", unused$text, label)
    )
  }

  if (length(hits) == 0L) {
    return(lintHits())
  }
  do.call(rbind, hits)
}

#' Find functions with too many lines of code
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param maxLines Maximum number of code lines allowed per function.
#' @return A data frame of lint hits as returned by [lintHits()].
#' @export
checkLongFunctions <- function(pd, maxLines = 50L) {
  defs <- findFunctionDefs(pd)
  # Anonymous functions are judged as part of whatever encloses them.
  defs <- defs[!is.na(defs$name), , drop = FALSE]
  sizes <- vapply(defs$id, function(id) functionCodeLines(pd, id), integer(1))
  long <- sizes > maxLines
  if (!any(long)) {
    return(lintHits())
  }

  lintHits(
    defs$line1[long],
    rep(1L, sum(long)),
    sprintf(
      "Function '%s' has %d lines of code (limit %d).",
      defs$name[long], sizes[long], as.integer(maxLines)
    )
  )
}

#' Find TODO-style comments
#'
#' @param pd A data frame of parse data as returned by [readSourceTokens()].
#' @param tags Comment tags to look for, matched case-insensitively.
#' @return A data frame of lint hits as returned by [lintHits()].
#' @export
checkTodoComments <- function(pd, tags = c("TODO", "FIXME", "XXX")) {
  comments <- pd[pd$token == "COMMENT", , drop = FALSE]
  # Group 1 is the tag, group 2 the note after an optional colon. Roxygen
  # comments (#') are included on purpose.
  pattern <- paste0("^#+'?\\s*(", paste(tags, collapse = "|"), ")\\b:?\\s*(.*)$")
  matched <- grepl(pattern, comments$text, ignore.case = TRUE, perl = TRUE)
  if (!any(matched)) {
    return(lintHits())
  }

  comments <- comments[matched, , drop = FALSE]
  tag <- toupper(sub(pattern, "\\1", comments$text, ignore.case = TRUE, perl = TRUE))
  note <- sub(pattern, "\\2", comments$text, ignore.case = TRUE, perl = TRUE)
  lintHits(
    comments$line1,
    comments$col1,
    ifelse(nzchar(note), sprintf("%s: %s", tag, note), sprintf("Bare %s comment.", tag))
  )
}

#' Find lines that exceed a maximum width
#'
#' @param lines A character vector of source lines.
#' @param maxWidth Maximum number of characters allowed per line.
#' @return A data frame of lint hits as returned by [lintHits()].
#' @export
checkLineLength <- function(lines, maxWidth = 80L) {
  # Display width, so wide characters count double like they do on screen.
  widths <- nchar(lines, type = "width")
  bad <- which(widths > maxWidth)
  lintHits(
    bad,
    rep(as.integer(maxWidth) + 1L, length(bad)),
    sprintf("Line is %d characters wide (limit %d).", widths[bad], as.integer(maxWidth))
  )
}

#' Find lines with trailing whitespace
#'
#' @param lines A character vector of source lines.
#' @return A data frame of lint hits as returned by [lintHits()].
#' @export
checkTrailingWhitespace <- function(lines) {
  bad <- which(grepl("[ \t]+$", lines))
  trimmed <- nchar(sub("[ \t]+$", "", lines[bad]))
  lintHits(
    bad,
    trimmed + 1L,
    "Trailing whitespace."
  )
}

#' Find lines indented with tab characters
#'
#' @param lines A character vector of source lines.
#' @return A data frame of lint hits as returned by [lintHits()].
#' @export
checkTabIndent <- function(lines) {
  bad <- which(grepl("^ *\t", lines))
  column <- regexpr("\t", lines[bad], fixed = TRUE)
  lintHits(
    bad,
    as.integer(column),
    "Indent with spaces, not tabs."
  )
}
