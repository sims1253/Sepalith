#' Summarise the terminal tokens of R code
#'
#' Tallies the five token classes a linter looks at most often.
#' @param text A character vector of R source lines.
#' @return A list of data frames `symbols`, `calls`, `numbers`, `strings`
#'   and `comments`, each with columns `text`, `n` and `line` (first use).
#' @export
token_summary <- function(text) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, c("line1", "col1", "token", "text")]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]

  # Symbols used as variables
  rows <- pd[pd$token == "SYMBOL", , drop = FALSE]
  u <- unique(rows$text)
  symbols <- data.frame(text = u, n = tabulate(match(rows$text, u), length(u)),
    line = rows$line1[match(u, rows$text)], stringsAsFactors = FALSE)

  # Functions that are called
  rows <- pd[pd$token == "SYMBOL_FUNCTION_CALL", , drop = FALSE]
  u <- unique(rows$text)
  calls <- data.frame(text = u, n = tabulate(match(rows$text, u), length(u)),
    line = rows$line1[match(u, rows$text)], stringsAsFactors = FALSE)

  # Numeric and logical constants
  rows <- pd[pd$token == "NUM_CONST", , drop = FALSE]
  u <- unique(rows$text)
  numbers <- data.frame(text = u, n = tabulate(match(rows$text, u), length(u)),
    line = rows$line1[match(u, rows$text)], stringsAsFactors = FALSE)

  # String constants
  rows <- pd[pd$token == "STR_CONST", , drop = FALSE]
  u <- unique(rows$text)
  strings <- data.frame(text = u, n = tabulate(match(rows$text, u), length(u)),
    line = rows$line1[match(u, rows$text)], stringsAsFactors = FALSE)

  # Comments
  rows <- pd[pd$token == "COMMENT", , drop = FALSE]
  u <- unique(rows$text)
  comments <- data.frame(text = u, n = tabulate(match(rows$text, u), length(u)),
    line = rows$line1[match(u, rows$text)], stringsAsFactors = FALSE)

  return(list(
    symbols = symbols, calls = calls, numbers = numbers,
    strings = strings, comments = comments
  ))
}

#' Describe each terminal token of R code
#'
#' Walks the parse data in source order and classifies every token.
#' @param text A character vector of R source lines.
#' @return A data frame with columns `line`, `col`, `token`, `text`, `kind`
#'   (`assign`, `literal`, `comment`, `call`, `symbol` or `other`) and
#'   `detail`.
#' @export
describe_tokens <- function(text) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  n <- nrow(pd)
  kind <- character(n)
  detail <- character(n)

  for (i in seq_len(n)) {
    txt <- pd$text[i]
    switch(pd$token[i],
      LEFT_ASSIGN = {
        kind[i] <- "assign"
        detail[i] <- if (txt == "<<-") "superassign" else "left"
      },
      RIGHT_ASSIGN = {
        kind[i] <- "assign"
        detail[i] <- if (txt == "->>") "superassign" else "right"
      },
      EQ_ASSIGN = {
        kind[i] <- "assign"
        detail[i] <- "equals"
      },
      NUM_CONST = {
        kind[i] <- "literal"
        detail[i] <- typeof(str2lang(txt))
      },
      STR_CONST = {
        kind[i] <- "literal"
        detail[i] <- typeof(str2lang(txt))
      },
      NULL_CONST = {
        kind[i] <- "literal"
        detail[i] <- typeof(str2lang(txt))
      },
      COMMENT = {
        kind[i] <- "comment"
        detail[i] <- if (startsWith(txt, "#'")) "roxygen" else "plain"
      },
      SYMBOL_FUNCTION_CALL = {
        kind[i] <- "call"
        detail[i] <- if (i > 1L && pd$token[i - 1L] == "NS_GET") "namespaced" else "bare"
      },
      SYMBOL = {
        kind[i] <- "symbol"
        detail[i] <- if (txt %in% c("T", "F")) "abbreviation" else "name"
      },
      {
        kind[i] <- "other"
        detail[i] <- pd$token[i]
      }
    )
  }

  return(data.frame(
    line = pd$line1, col = pd$col1, token = pd$token, text = pd$text,
    kind = kind, detail = detail, stringsAsFactors = FALSE
  ))
}

#' Measure bracket nesting in R code
#'
#' Tracks the depth of parentheses, braces and square brackets token by token.
#' @param text A character vector of R source lines.
#' @return A data frame with one row per bracket kind (`paren`, `brace`,
#'   `square`) and columns `kind`, `opened`, `max_depth` and `max_line`.
#' @export
bracket_depth <- function(text) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  kinds <- c("paren", "brace", "square")
  depth <- opened <- max_depth <- max_line <- stats::setNames(integer(3), kinds)

  for (i in seq_len(nrow(pd))) {
    line <- pd$line1[i]
    switch(pd$token[i],
      "'('" = {
        depth[["paren"]] <- depth[["paren"]] + 1L
        opened[["paren"]] <- opened[["paren"]] + 1L
        if (depth[["paren"]] > max_depth[["paren"]]) {
          max_depth[["paren"]] <- depth[["paren"]]
          max_line[["paren"]] <- line
        }
      },
      "')'" = {
        depth[["paren"]] <- depth[["paren"]] - 1L
      },
      "'{'" = {
        depth[["brace"]] <- depth[["brace"]] + 1L
        opened[["brace"]] <- opened[["brace"]] + 1L
        if (depth[["brace"]] > max_depth[["brace"]]) {
          max_depth[["brace"]] <- depth[["brace"]]
          max_line[["brace"]] <- line
        }
      },
      "'}'" = {
        depth[["brace"]] <- depth[["brace"]] - 1L
      },
      # `[[` closes with two `]` tokens, so it opens two levels at once
      LBB = {
        depth[["square"]] <- depth[["square"]] + 2L
        opened[["square"]] <- opened[["square"]] + 1L
        if (depth[["square"]] > max_depth[["square"]]) {
          max_depth[["square"]] <- depth[["square"]]
          max_line[["square"]] <- line
        }
      },
      "'['" = {
        depth[["square"]] <- depth[["square"]] + 1L
        opened[["square"]] <- opened[["square"]] + 1L
        if (depth[["square"]] > max_depth[["square"]]) {
          max_depth[["square"]] <- depth[["square"]]
          max_line[["square"]] <- line
        }
      },
      "']'" = {
        depth[["square"]] <- depth[["square"]] - 1L
      }
    )
  }

  return(data.frame(
    kind = kinds, opened = unname(opened), max_depth = unname(max_depth),
    max_line = unname(max_line), stringsAsFactors = FALSE
  ))
}

#' Group the names in R code by role
#'
#' Separates assigned names, formal arguments, argument names in calls,
#' called functions, package qualifiers and other symbols.
#' @param text A character vector of R source lines.
#' @return A list of sorted, unique names with elements `assigned`,
#'   `formals`, `arguments`, `called`, `packages` and `used`.
#' @export
symbol_roles <- function(text) {
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  nxt <- c(pd$token[-1L], "")
  prv <- c("", pd$token[-nrow(pd)])
  clean <- function(x) sort(unique(gsub("^`|`$", "", x)))

  # Names bound by `<-`, `=` or `->`
  is_assigned <- pd$token == "SYMBOL" &
    (nxt %in% c("LEFT_ASSIGN", "EQ_ASSIGN") | prv == "RIGHT_ASSIGN")
  assigned <- clean(pd$text[is_assigned])

  # Names of formal arguments
  is_formal <- pd$token == "SYMBOL_FORMALS"
  formals <- clean(pd$text[is_formal])

  # Names of arguments given in calls
  is_argument <- pd$token == "SYMBOL_SUB"
  arguments <- clean(pd$text[is_argument])

  # Names of called functions
  is_called <- pd$token == "SYMBOL_FUNCTION_CALL"
  called <- clean(pd$text[is_called])

  # Names of packages used with `::` or `:::`
  is_package <- pd$token == "SYMBOL_PACKAGE"
  packages <- clean(pd$text[is_package])

  # Every other symbol is a use of a variable
  is_used <- pd$token == "SYMBOL" & !is_assigned
  used <- clean(pd$text[is_used])

  return(list(
    assigned = assigned, formals = formals, arguments = arguments,
    called = called, packages = packages, used = used
  ))
}

#' Extract the values of one kind of token
#'
#' Pulls literal values, comment text or names out of R code.
#' @param text A character vector of R source lines.
#' @param type The kind of token to extract.
#' @param unique Drop repeated values, keeping the first occurrence?
#' @return A vector named by source line: character for `strings`,
#'   `comments`, `calls` and `symbols`, numeric for `numbers`.
#' @export
extract_tokens <- function(text, type = c("strings", "numbers", "comments", "calls", "symbols"),
                           unique = FALSE) {
  type <- match.arg(type)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]

  values <- switch(type,
    strings = {
      rows <- pd[pd$token == "STR_CONST", , drop = FALSE]
      out <- vapply(rows$text, str2lang, character(1), USE.NAMES = FALSE)
      stats::setNames(out, rows$line1)
    },
    numbers = {
      rows <- pd[pd$token == "NUM_CONST" & grepl("^[0-9.]", pd$text), , drop = FALSE]
      out <- vapply(rows$text, function(s) as.numeric(str2lang(s)), numeric(1),
        USE.NAMES = FALSE)
      stats::setNames(out, rows$line1)
    },
    comments = {
      rows <- pd[pd$token == "COMMENT", , drop = FALSE]
      out <- trimws(sub("^#+'?", "", rows$text))
      stats::setNames(out, rows$line1)
    },
    calls = {
      rows <- pd[pd$token == "SYMBOL_FUNCTION_CALL", , drop = FALSE]
      out <- gsub("^`|`$", "", rows$text)
      stats::setNames(out, rows$line1)
    },
    symbols = {
      rows <- pd[pd$token == "SYMBOL", , drop = FALSE]
      out <- gsub("^`|`$", "", rows$text)
      stats::setNames(out, rows$line1)
    }
  )

  if (unique) {
    values <- values[!duplicated(values)]
  }
  return(values)
}
