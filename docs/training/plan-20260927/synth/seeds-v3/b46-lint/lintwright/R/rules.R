#' Describe a lint rule
#'
#' Looks up the severity, the advice and a minimal example for one rule.
#' @param rule The name of the rule.
#' @return A list with elements `rule`, `severity`, `use`, `avoid`,
#'   `message`, `bad` and `good`.
#' @export
rule_info <- function(rule = c("assign_eq", "true_false", "semicolon", "trailing_space",
                               "single_quotes", "browser_call")) {
  rule <- match.arg(rule)
  switch(rule,
    assign_eq = {
      severity <- "style"
      use <- "`<-`"
      avoid <- "`=`"
      bad <- "x = 1"
      good <- "x <- 1"
    },
    true_false = {
      severity <- "warning"
      use <- "`TRUE` and `FALSE`"
      avoid <- "`T` and `F`"
      bad <- "ok <- T"
      good <- "ok <- TRUE"
    },
    semicolon = {
      severity <- "style"
      use <- "a line break"
      avoid <- "`;`"
      bad <- "x <- 1; y <- 2"
      good <- c("x <- 1", "y <- 2")
    },
    trailing_space = {
      severity <- "info"
      use <- "a clean line end"
      avoid <- "trailing white space"
      bad <- "x <- 1 "
      good <- "x <- 1"
    },
    single_quotes = {
      severity <- "style"
      use <- "double quotes"
      avoid <- "single quotes"
      bad <- "x <- 'a'"
      good <- "x <- \"a\""
    },
    browser_call = {
      severity <- "error"
      use <- "a breakpoint set in the IDE"
      avoid <- "`browser()`"
      bad <- "browser()"
      good <- "NULL"
    }
  )

  message <- sprintf("Use %s instead of %s.", use, avoid)
  list(
    rule = rule, severity = severity, use = use, avoid = avoid,
    message = message, bad = bad, good = good
  )
}

#' Apply one lint rule to R code
#'
#' Finds every place where the code breaks the rule.
#' @param text A character vector of R source lines.
#' @param rule The name of the rule.
#' @return A data frame with columns `line`, `column`, `rule`, `severity` and
#'   `message`, one row per hit.
#' @export
lint_rule <- function(text, rule = c("assign_eq", "true_false", "semicolon", "trailing_space",
                                     "single_quotes", "browser_call")) {
  rule <- match.arg(rule)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]

  switch(rule,
    assign_eq = {
      hit <- pd$token == "EQ_ASSIGN"
      line <- pd$line1[hit]
      column <- pd$col1[hit]
    },
    true_false = {
      hit <- pd$token == "SYMBOL" & pd$text %in% c("T", "F")
      line <- pd$line1[hit]
      column <- pd$col1[hit]
    },
    semicolon = {
      hit <- pd$token == "';'"
      line <- pd$line1[hit]
      column <- pd$col1[hit]
    },
    trailing_space = {
      pos <- regexpr("[ ]+$", text)
      line <- which(pos > 0L)
      column <- as.integer(pos[line])
    },
    single_quotes = {
      hit <- pd$token == "STR_CONST" & startsWith(pd$text, "'")
      line <- pd$line1[hit]
      column <- pd$col1[hit]
    },
    browser_call = {
      hit <- pd$token == "SYMBOL_FUNCTION_CALL" & pd$text == "browser"
      line <- pd$line1[hit]
      column <- pd$col1[hit]
    }
  )

  info <- rule_info(rule)
  n <- length(line)
  data.frame(
    line = line, column = column, rule = rep(rule, n),
    severity = rep(info$severity, n), message = rep(info$message, n),
    stringsAsFactors = FALSE
  )
}

#' Lint R code
#'
#' Applies several rules and honours `# nolint` comments.
#' @param text A character vector of R source lines.
#' @param rules The names of the rules to apply.
#' @param nolint Skip lines marked with `# nolint` or `# nolint: rule, ...`?
#' @return A data frame like the one from [lint_rule()], sorted by position.
#' @export
lint_code <- function(text, rules = c("assign_eq", "true_false", "semicolon", "trailing_space",
                                      "single_quotes", "browser_call"),
                      nolint = TRUE) {
  rules <- match.arg(rules, several.ok = TRUE)
  lints <- do.call(rbind, lapply(rules, function(r) lint_rule(text, r)))

  if (nolint) {
    # A bare `# nolint` silences every rule on its line
    all_off <- grepl("# nolint\\s*$", text)
    lints <- lints[!all_off[lints$line], , drop = FALSE]
    # `# nolint: a, b` silences only the listed rules
    listed <- regmatches(text, regexec("# nolint: ?([a-z_, ]+)$", text))
    off <- lapply(listed, function(m) {
      if (length(m) == 0L) character() else trimws(strsplit(m[2], ",")[[1]])
    })
    silenced <- vapply(seq_len(nrow(lints)), function(i) {
      lints$rule[i] %in% off[[lints$line[i]]]
    }, logical(1))
    lints <- lints[!silenced, , drop = FALSE]
  }

  lints <- lints[order(lints$line, lints$column), , drop = FALSE]
  rownames(lints) <- NULL
  lints
}

#' Lint the names bound in R code
#'
#' Checks assigned names and formal arguments against a naming style and
#' suggests a replacement in that style.
#' @param text A character vector of R source lines.
#' @param style The naming style the code should follow.
#' @return A data frame with columns `line`, `name` and `suggestion`, one row
#'   per name that does not follow `style`.
#' @export
lint_names <- function(text, style = c("snake_case", "camelCase", "dotted.case",
                                       "UPPER_CASE", "lowercase")) {
  style <- match.arg(style)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  nxt <- c(pd$token[-1L], "")
  bound <- (pd$token == "SYMBOL" & nxt %in% c("LEFT_ASSIGN", "EQ_ASSIGN")) |
    pd$token == "SYMBOL_FORMALS"
  name <- pd$text[bound]
  words <- strsplit(tolower(gsub("([a-z0-9])([A-Z])", "\\1_\\2", name)), "[._]+")

  switch(style,
    snake_case = {
      pattern <- "^[a-z][a-z0-9]*(_[a-z0-9]+)*$"
      suggestion <- vapply(words, paste, character(1), collapse = "_")
    },
    camelCase = {
      pattern <- "^[a-z][a-z0-9]*([A-Z][a-z0-9]*)*$"
      suggestion <- vapply(words, function(w) {
        paste0(w[1], paste0(toupper(substr(w[-1], 1, 1)), substring(w[-1], 2), collapse = ""))
      }, character(1))
    },
    dotted.case = {
      pattern <- "^[a-z][a-z0-9]*(\\.[a-z0-9]+)*$"
      suggestion <- vapply(words, paste, character(1), collapse = ".")
    },
    UPPER_CASE = {
      pattern <- "^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$"
      suggestion <- toupper(vapply(words, paste, character(1), collapse = "_"))
    },
    lowercase = {
      pattern <- "^[a-z][a-z0-9]*$"
      suggestion <- vapply(words, paste, character(1), collapse = "")
    }
  )

  bad <- !grepl(pattern, name)
  data.frame(
    line = pd$line1[bound][bad], name = name[bad], suggestion = suggestion[bad],
    stringsAsFactors = FALSE
  )
}

#' Lint the spacing around operators
#'
#' Checks the space before and after each operator of one class.
#' @param text A character vector of R source lines.
#' @param operator The class of operators to check.
#' @return A data frame with columns `line`, `column`, `operator` and
#'   `message`, one row per missing or unexpected space.
#' @export
lint_spacing <- function(text, operator = c("assign", "comparison", "arithmetic", "comma",
                                             "pipe")) {
  operator <- match.arg(operator)
  pd <- utils::getParseData(parse(text = text, keep.source = TRUE))
  pd <- pd[pd$terminal, , drop = FALSE]
  pd <- pd[order(pd$line1, pd$col1), , drop = FALSE]
  n <- nrow(pd)

  switch(operator,
    assign = {
      tokens <- c("LEFT_ASSIGN", "RIGHT_ASSIGN", "EQ_ASSIGN")
      space_before <- TRUE
      space_after <- TRUE
    },
    comparison = {
      tokens <- c("EQ", "NE", "LT", "GT", "LE", "GE")
      space_before <- TRUE
      space_after <- TRUE
    },
    arithmetic = {
      tokens <- c("'+'", "'*'", "'/'")
      space_before <- TRUE
      space_after <- TRUE
    },
    comma = {
      tokens <- "','"
      space_before <- FALSE
      space_after <- TRUE
    },
    pipe = {
      tokens <- c("PIPE", "SPECIAL")
      space_before <- TRUE
      space_after <- TRUE
    }
  )

  hit <- pd$token %in% tokens
  prev_line <- c(-1L, pd$line1[-n])
  prev_end <- c(-1L, pd$col2[-n])
  next_line <- c(pd$line1[-1L], -1L)
  next_start <- c(pd$col1[-1L], -1L)

  # Space before the operator
  same_line <- hit & prev_line == pd$line1
  if (space_before) {
    bad_before <- same_line & prev_end == pd$col1 - 1L
  } else {
    bad_before <- same_line & prev_end < pd$col1 - 1L
  }
  # Space after the operator
  same_line <- hit & next_line == pd$line2
  if (space_after) {
    bad_after <- same_line & next_start == pd$col2 + 1L
  } else {
    bad_after <- same_line & next_start > pd$col2 + 1L
  }

  before <- if (space_before) "Missing" else "Unexpected"
  after <- if (space_after) "Missing" else "Unexpected"
  rows <- c(which(bad_before), which(bad_after))
  out <- data.frame(
    line = pd$line1[rows], column = pd$col1[rows], operator = pd$text[rows],
    message = c(sprintf("%s space before `%s`.", before, pd$text[bad_before]),
      sprintf("%s space after `%s`.", after, pd$text[bad_after])),
    stringsAsFactors = FALSE
  )
  out <- out[order(out$line, out$column), , drop = FALSE]
  rownames(out) <- NULL
  out
}
