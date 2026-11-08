#' Fix the spacing around operators
#'
#' Puts exactly one space, or none, on each side of the operators of one
#' class. Operators at the start or end of a line keep their line break.
#' @param text A character vector of R source lines.
#' @param operator The class of operators to fix.
#' @return A list with elements `before`, `after` and `changed` (line
#'   numbers that were edited).
#' @export
fix.spacing <- function(text, operator = c("assign", "comparison", "arithmetic", "comma",
                                           "pipe")) {
  operator <- match.arg(operator)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]

  switch(operator,
    assign = {
      tokens <- c("LEFT_ASSIGN", "RIGHT_ASSIGN", "EQ_ASSIGN")
      space.before <- TRUE
      space.after <- TRUE
    },
    comparison = {
      tokens <- c("EQ", "NE", "LT", "GT", "LE", "GE")
      space.before <- TRUE
      space.after <- TRUE
    },
    arithmetic = {
      tokens <- c("'+'", "'*'", "'/'")
      space.before <- TRUE
      space.after <- TRUE
    },
    comma = {
      tokens <- "','"
      space.before <- FALSE
      space.after <- TRUE
    },
    pipe = {
      tokens <- c("PIPE", "SPECIAL")
      space.before <- TRUE
      space.after <- TRUE
    }
  )

  hits <- pd[pd$token %in% tokens, , drop = FALSE]
  after <- text
  for (k in order(hits$line1, hits$col1, decreasing = TRUE)) {
    l <- hits$line1[k]
    left <- substr(after[l], 1L, hits$col1[k] - 1L)
    right <- substring(after[l], hits$col2[k] + 1L)
    # Space before the operator, unless it starts the line
    if (nzchar(trimws(left))) {
      left <- if (space.before) sub(" *$", " ", left) else sub(" +$", "", left)
    }
    # Space after the operator, unless it ends the line
    if (nzchar(trimws(right))) {
      right <- if (space.after) sub("^ *", " ", right) else sub("^ +", "", right)
    }
    after[l] <- paste0(left, hits$text[k], right)
  }
  list(before = text, after = after, changed = which(after != text))
}

#' Fix the spacing inside brackets
#'
#' Removes the space just inside parentheses and square brackets, and puts
#' one space just inside braces. Empty brackets end up empty.
#' @param text A character vector of R source lines.
#' @param bracket The kind of bracket to fix.
#' @return A list with elements `before`, `after` and `changed` (line
#'   numbers that were edited).
#' @export
fix.brackets <- function(text, bracket = c("paren", "square", "brace")) {
  bracket <- match.arg(bracket)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]

  switch(bracket,
    paren = {
      open <- "'('"
      close <- "')'"
      inside <- ""
    },
    square = {
      open <- c("'['", "LBB")
      close <- "']'"
      inside <- ""
    },
    brace = {
      open <- "'{'"
      close <- "'}'"
      inside <- " "
    }
  )

  hits <- pd[pd$token %in% c(open, close), , drop = FALSE]
  after <- text
  for (k in order(hits$line1, hits$col1, decreasing = TRUE)) {
    l <- hits$line1[k]
    left <- substr(after[l], 1L, hits$col1[k] - 1L)
    right <- substring(after[l], hits$col2[k] + 1L)
    if (hits$token[k] %in% open) {
      # Space after an opening bracket, unless the line ends there
      if (nzchar(trimws(right)) && !startsWith(trimws(right), "#")) {
        right <- sub("^ *", if (grepl("^ *[])}]", right)) "" else inside, right)
      }
    } else {
      # Space before a closing bracket, unless the line starts there
      if (nzchar(trimws(left))) {
        left <- sub(" *$", if (grepl("[[({] *$", left)) "" else inside, left)
      }
    }
    after[l] <- paste0(left, hits$text[k], right)
  }
  list(before = text, after = after, changed = which(after != text))
}

#' Fix the indentation of R code
#'
#' Re-indents every line by its bracket depth. Lines inside multi-line
#' strings are left alone.
#' @param text A character vector of R source lines.
#' @param width The number of spaces per level.
#' @return A list with elements `before`, `after` and `changed` (line
#'   numbers that were edited).
#' @export
fix.indent <- function(text, width = 2L) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  n <- length(text)

  # Brackets opened on each line; `[[` opens two levels
  weight <- ifelse(pd$token == "LBB", 2L, 1L)
  is.open <- pd$token %in% c("'{'", "'('", "'['", "LBB")
  opened <- vapply(seq_len(n), function(i) sum(weight[is.open & pd$line1 == i]), integer(1))
  # Brackets closed on each line
  is.close <- pd$token %in% c("'}'", "')'", "']'")
  closed <- vapply(seq_len(n), function(i) sum(is.close & pd$line1 == i), integer(1))

  # Closing brackets at the start of a line dedent that line
  leading <- integer(n)
  for (l in unique(pd$line1)) {
    closing <- pd$token[pd$line1 == l] %in% c("'}'", "')'", "']'")
    leading[l] <- if (all(closing)) length(closing) else which(!closing)[1L] - 1L
  }
  depth <- cumsum(c(0L, opened - closed)[seq_len(n)]) - leading

  # Lines inside multi-line strings keep their text as it is
  strings <- pd[pd$line2 > pd$line1, , drop = FALSE]
  inner <- unlist(Map(function(from, to) seq_len(to - from) + from, strings$line1, strings$line2))
  after <- text
  todo <- nzchar(trimws(text)) & !(seq_len(n) %in% inner)
  after[todo] <- paste0(strrep(" ", width * depth[todo]), trimws(text[todo], "left"))
  list(before = text, after = after, changed = which(after != text))
}

#' Fix the spacing around keywords
#'
#' Puts one space between `if`, `for` and `while` and their condition, none
#' between `function` and its arguments, and one space on each side of
#' `else`.
#' @param text A character vector of R source lines.
#' @return A list with elements `before`, `after` and `changed` (line
#'   numbers that were edited).
#' @export
fix.keywords <- function(text) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  hits <- pd[pd$token %in% c("IF", "FOR", "WHILE", "FUNCTION", "ELSE"), , drop = FALSE]
  after <- text

  for (k in order(hits$line1, hits$col1, decreasing = TRUE)) {
    l <- hits$line1[k]
    left <- substr(after[l], 1L, hits$col1[k] - 1L)
    right <- substring(after[l], hits$col2[k] + 1L)
    switch(hits$token[k],
      IF = {
        right <- sub("^ *", " ", right)
      },
      FOR = {
        right <- sub("^ *", " ", right)
      },
      WHILE = {
        right <- sub("^ *", " ", right)
      },
      FUNCTION = {
        right <- sub("^ +", "", right)
      },
      ELSE = {
        if (nzchar(trimws(left))) left <- sub(" *$", " ", left)
        if (nzchar(trimws(right))) right <- sub("^ *", " ", right)
      }
    )
    after[l] <- paste0(left, hits$text[k], right)
  }
  list(before = text, after = after, changed = which(after != text))
}

#' Fix the pipe operator
#'
#' Switches between the magrittr pipe `%>%` and the native pipe `|>`. A
#' bare function name on the right of `%>%` gets the parentheses `|>`
#' requires.
#' @param text A character vector of R source lines.
#' @param to The pipe to use: `native` or `magrittr`.
#' @return A list with elements `before`, `after` and `changed` (line
#'   numbers that were edited).
#' @export
fix.pipes <- function(text, to = c("native", "magrittr")) {
  to <- match.arg(to)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]

  if (to == "native") {
    # `%>%` becomes `|>`, and `x %>% f` becomes `x |> f()`
    is.hit <- pd$token == "SPECIAL" & pd$text == "%>%"
    value <- rep("|>", sum(is.hit))
    bare <- which(is.hit) + 1L
    bare <- bare[pd$token[bare] == "SYMBOL"]
  } else {
    # `|>` becomes `%>%`; calls on the right already have their parentheses
    is.hit <- pd$token == "PIPE"
    value <- rep("%>%", sum(is.hit))
    bare <- integer()
  }

  edits <- rbind(
    data.frame(line = pd$line1[is.hit], col1 = pd$col1[is.hit], col2 = pd$col2[is.hit],
      value = value),
    data.frame(line = pd$line1[bare], col1 = pd$col2[bare] + 1L, col2 = pd$col2[bare],
      value = rep("()", length(bare)))
  )
  after <- text
  for (k in order(edits$line, edits$col1, decreasing = TRUE)) {
    l <- edits$line[k]
    after[l] <- paste0(substr(after[l], 1L, edits$col1[k] - 1L), edits$value[k],
      substring(after[l], edits$col2[k] + 1L))
  }
  list(before = text, after = after, changed = which(after != text))
}
