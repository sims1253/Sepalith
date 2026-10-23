#' Fix the code that breaks one lint rule
#'
#' Rewrites the tokens a rule complains about, editing each line from the
#' right so the columns from the parse data stay valid.
#' @param text A character vector of R source lines.
#' @param rule The rule to fix.
#' @return A list with elements `rule`, `before` (the input), `after` (the
#'   fixed lines) and `changed` (input line numbers that were edited).
#' @export
fix.rule <- function(text, rule = c("assign_eq", "true_false", "semicolon", "trailing_space",
                                    "single_quotes")) {
  rule <- match.arg(rule)
  before <- text
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, c("line1", "col1", "line2", "col2", "token", "text")]

  switch(rule,
    assign_eq = {
      hits <- pd[pd$token == "EQ_ASSIGN", , drop = FALSE]
      value <- rep("<-", nrow(hits))
    },
    true_false = {
      hits <- pd[pd$token == "SYMBOL" & pd$text %in% c("T", "F"), , drop = FALSE]
      value <- ifelse(hits$text == "T", "TRUE", "FALSE")
    },
    semicolon = {
      hits <- pd[pd$token == "';'", , drop = FALSE]
      rest <- substring(text[hits$line1], hits$col2 + 1L)
      hits$col2 <- hits$col2 + nchar(rest) - nchar(sub("^ +", "", rest))
      indent <- sub("^( *).*$", "\\1", text[hits$line1])
      value <- ifelse(nzchar(trimws(rest)), paste0("\n", indent), "")
    },
    trailing_space = {
      pos <- regexpr(" +$", text)
      hits <- data.frame(line1 = which(pos > 0L), col1 = as.integer(pos[pos > 0L]))
      hits$col2 <- nchar(text[hits$line1])
      value <- rep("", nrow(hits))
    },
    single_quotes = {
      hits <- pd[pd$token == "STR_CONST" & startsWith(pd$text, "'") &
        pd$line1 == pd$line2, , drop = FALSE]
      inner <- substr(hits$text, 2L, nchar(hits$text) - 1L)
      inner <- gsub("\"", "\\\\\"", gsub("\\\\'", "'", inner))
      value <- paste0("\"", inner, "\"")
    }
  )

  # Apply the edits from the right so earlier columns stay valid
  for (k in order(hits$line1, hits$col1, decreasing = TRUE)) {
    l <- hits$line1[k]
    text[l] <- paste0(substr(text[l], 1L, hits$col1[k] - 1L), value[k],
      substring(text[l], hits$col2[k] + 1L))
  }
  after <- unlist(lapply(strsplit(text, "\n", fixed = TRUE), function(x) {
    if (length(x) == 0L) "" else x
  }))
  list(rule = rule, before = before, after = after, changed = sort(unique(hits$line1)))
}

#' Fix several lint rules in turn
#'
#' Applies [fix.rule()] once per rule, feeding each result into the next.
#' @param text A character vector of R source lines.
#' @param rules The rules to fix, in order.
#' @return A list with elements `before`, `after` and `log`, a data frame
#'   with columns `rule`, `lines_before`, `lines_after` and `edited`.
#' @export
fix.all <- function(text, rules = c("assign_eq", "true_false", "semicolon", "trailing_space",
                                    "single_quotes")) {
  rules <- match.arg(rules, several.ok = TRUE)
  current <- text
  lines.before <- lines.after <- edited <- integer(length(rules))

  for (i in seq_along(rules)) {
    fixed <- fix.rule(current, rules[i])
    lines.before[i] <- length(fixed$before)
    lines.after[i] <- length(fixed$after)
    edited[i] <- length(fixed$changed)
    current <- fixed$after
  }

  log <- data.frame(
    rule = rules, lines_before = lines.before, lines_after = lines.after,
    edited = edited, stringsAsFactors = FALSE
  )
  list(before = text, after = current, log = log)
}

#' Fix blank lines
#'
#' Trims blank lines at the start and end of a file and shortens runs of
#' blank lines inside it.
#' @param text A character vector of R source lines.
#' @param max.blank The longest run of blank lines to keep inside the file.
#' @param at Where to trim blank lines: `start`, `end` or `both`.
#' @return A list with elements `before`, `after` and `removed`, the number
#'   of lines dropped at the `start`, in the `middle` and at the `end`.
#' @export
fix.blank.lines <- function(text, max.blank = 1L, at = c("both", "start", "end")) {
  at <- match.arg(at)
  blank <- !nzchar(trimws(text))
  keep <- rep(TRUE, length(text))
  filled <- which(!blank)

  # Drop the blank lines at the start of the file
  if (at %in% c("both", "start")) {
    first <- if (length(filled) > 0L) filled[1L] else length(text) + 1L
    keep[seq_len(first - 1L)] <- FALSE
  }
  removed.start <- sum(!keep)
  # Drop the blank lines at the end of the file
  if (at %in% c("both", "end")) {
    last <- if (length(filled) > 0L) filled[length(filled)] else 0L
    keep[seq_along(text) > last] <- FALSE
  }
  removed.end <- sum(!keep) - removed.start

  # Shorten the runs of blank lines that are still kept
  run <- 0L
  removed.middle <- 0L
  for (i in seq_along(text)) {
    run <- if (blank[i] && keep[i]) run + 1L else 0L
    if (run > max.blank) {
      keep[i] <- FALSE
      removed.middle <- removed.middle + 1L
    }
  }

  removed <- c(start = removed.start, middle = removed.middle, end = removed.end)
  list(before = text, after = text[keep], removed = removed)
}

#' Fix the quoting of strings
#'
#' Converts single-line string literals to one quote character, escaping
#' the new quote and unescaping the old one inside the string.
#' @param text A character vector of R source lines.
#' @param to The quote character to use: `double` or `single`.
#' @return A list with elements `before`, `after` and `changed` (line
#'   numbers that were edited).
#' @export
fix.quotes <- function(text, to = c("double", "single")) {
  to <- match.arg(to)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal & pd$token == "STR_CONST" & pd$line1 == pd$line2, , drop = FALSE]

  if (to == "double") {
    # Single-quoted strings become double-quoted
    hits <- pd[startsWith(pd$text, "'"), , drop = FALSE]
    inner <- substr(hits$text, 2L, nchar(hits$text) - 1L)
    inner <- gsub("\"", "\\\\\"", gsub("\\\\'", "'", inner))
    value <- paste0("\"", inner, "\"")
  } else {
    # Double-quoted strings become single-quoted
    hits <- pd[startsWith(pd$text, "\""), , drop = FALSE]
    inner <- substr(hits$text, 2L, nchar(hits$text) - 1L)
    inner <- gsub("'", "\\\\'", gsub("\\\\\"", "\"", inner))
    value <- paste0("'", inner, "'")
  }

  after <- text
  for (k in order(hits$line1, hits$col1, decreasing = TRUE)) {
    l <- hits$line1[k]
    after[l] <- paste0(substr(after[l], 1L, hits$col1[k] - 1L), value[k],
      substring(after[l], hits$col2[k] + 1L))
  }
  list(before = text, after = after, changed = sort(unique(hits$line1)))
}

#' Fix the spelling of anonymous functions
#'
#' Switches anonymous functions between the `function` keyword and the `\`
#' shorthand. Functions assigned to a name always keep `function`.
#' @param text A character vector of R source lines.
#' @param to The spelling to use: `lambda` for `\(x)` or `function`.
#' @return A list with elements `before`, `after` and `changed` (line
#'   numbers that were edited).
#' @export
fix.lambda <- function(text, to = c("lambda", "function")) {
  to <- match.arg(to)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  prev <- c("", pd$token)[seq_len(nrow(pd))]

  if (to == "lambda") {
    # Anonymous `function` keywords become `\`
    hits <- pd[pd$token == "FUNCTION" & !prev %in% c("LEFT_ASSIGN", "EQ_ASSIGN"), , drop = FALSE]
    value <- rep("\\", nrow(hits))
  } else {
    # `\` shorthands become `function`
    hits <- pd[pd$token == "'\\\\'", , drop = FALSE]
    value <- rep("function", nrow(hits))
  }

  after <- text
  for (k in order(hits$line1, hits$col1, decreasing = TRUE)) {
    l <- hits$line1[k]
    after[l] <- paste0(substr(after[l], 1L, hits$col1[k] - 1L), value[k],
      substring(after[l], hits$col2[k] + 1L))
  }
  list(before = text, after = after, changed = sort(unique(hits$line1)))
}
