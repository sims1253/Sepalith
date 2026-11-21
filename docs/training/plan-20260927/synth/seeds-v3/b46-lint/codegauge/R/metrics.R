#' Measure an R source file
#'
#' @description
#' Computes the line and token counts most code reviews start from.
#' @param text A character vector of R source lines.
#' @return A named list with elements `lines`, `blank`, `comment`, `code`,
#'   `maxWidth`, `meanWidth`, `functions` and `assignments`.
#' @family metric functions
#' @export
fileMetrics = function(text) {
  pd = utils::getParseData(parse(text = text, keep.source = TRUE))
  tokens = pd[pd$terminal, , drop = FALSE]
  width = nchar(text)
  filled = nzchar(trimws(text))

  # Physical lines
  lines = length(text)
  # Blank lines
  blank = sum(!filled)
  # Lines holding nothing but a comment
  commentLines = unique(tokens$line1[tokens$token == "COMMENT"])
  codeLines = unique(tokens$line1[tokens$token != "COMMENT"])
  comment = length(setdiff(commentLines, codeLines))
  # Lines where a code token starts
  code = length(codeLines)
  # Width of the widest line
  maxWidth = if (lines > 0L) max(width) else 0L
  # Mean width of the lines that are not blank
  meanWidth = if (any(filled)) mean(width[filled]) else 0
  # Function definitions
  functions = sum(tokens$token == "FUNCTION")
  # Assignments in any direction
  assignments = sum(tokens$token %in% c("LEFT_ASSIGN", "RIGHT_ASSIGN", "EQ_ASSIGN"))

  list(
    lines = lines, blank = blank, comment = comment, code = code,
    maxWidth = maxWidth, meanWidth = meanWidth, functions = functions,
    assignments = assignments
  )
}

#' Compute one code metric
#'
#' @description
#' Computes a single metric from the parse data of R code. For the Halstead
#' volume every terminal token that is not an operand counts as an operator.
#' @param text A character vector of R source lines.
#' @param metric The metric to compute.
#' @return A single number.
#' @family metric functions
#' @export
metricValue = function(text, metric = c("sloc", "commentRatio", "cyclomatic", "maxNesting",
                                        "halsteadVolume", "uniqueCalls")) {
  metric = match.arg(metric)
  pd = utils::getParseData(parse(text = text, keep.source = TRUE))
  tokens = pd[pd$terminal, , drop = FALSE]
  tokens = tokens[order(tokens$line1, tokens$col1), , drop = FALSE]

  switch(metric,
    sloc = {
      codeLines = unique(tokens$line1[tokens$token != "COMMENT"])
      length(codeLines)
    },
    commentRatio = {
      nComment = sum(tokens$token == "COMMENT")
      nCode = length(unique(tokens$line1[tokens$token != "COMMENT"]))
      if (nComment + nCode == 0L) 0 else nComment / (nComment + nCode)
    },
    cyclomatic = {
      branches = c("IF", "FOR", "WHILE", "REPEAT", "AND2", "OR2")
      1L + sum(tokens$token %in% branches)
    },
    maxNesting = {
      depth = cumsum(tokens$token == "'{'") - cumsum(tokens$token == "'}'")
      max(0L, depth)
    },
    halsteadVolume = {
      isOperand = tokens$token %in% c("SYMBOL", "NUM_CONST", "STR_CONST", "NULL_CONST")
      isOperator = !isOperand & tokens$token != "COMMENT"
      vocabulary = length(unique(tokens$text[isOperand])) +
        length(unique(tokens$text[isOperator]))
      size = sum(isOperand | isOperator)
      if (vocabulary == 0L) 0 else size * log2(vocabulary)
    },
    uniqueCalls = {
      calls = tokens$text[tokens$token == "SYMBOL_FUNCTION_CALL"]
      length(unique(calls))
    }
  )
}

#' Classify the lines of an R source file
#'
#' @description
#' Gives every line exactly one class; earlier classes in the list below take
#' precedence over later ones.
#' @param text A character vector of R source lines.
#' @return A character vector with one class per line: `blank`, `continued`
#'   (inside a multi-line token), `mixed` (code and a comment), `code`,
#'   `roxygen` or `comment`.
#' @family metric functions
#' @export
lineClasses = function(text) {
  pd = utils::getParseData(parse(text = text, keep.source = TRUE))
  tokens = pd[pd$terminal, , drop = FALSE]
  codeTokens = tokens[tokens$token != "COMMENT", , drop = FALSE]
  commentTokens = tokens[tokens$token == "COMMENT", , drop = FALSE]
  line = seq_along(text)
  cls = character(length(text))

  # Blank lines
  isBlank = !nzchar(trimws(text))
  cls[isBlank & cls == ""] = "blank"
  # Lines inside a multi-line token, such as a long string
  inner = unlist(Map(function(from, to) seq_len(to - from) + from,
    codeTokens$line1, codeTokens$line2))
  isContinued = line %in% inner & !(line %in% codeTokens$line1)
  cls[isContinued & cls == ""] = "continued"
  # Lines with code and a trailing comment
  isMixed = line %in% codeTokens$line1 & line %in% commentTokens$line1
  cls[isMixed & cls == ""] = "mixed"
  # Lines with code only
  isCode = line %in% codeTokens$line1
  cls[isCode & cls == ""] = "code"
  # Roxygen comment lines
  isRoxygen = line %in% commentTokens$line1[startsWith(commentTokens$text, "#'")]
  cls[isRoxygen & cls == ""] = "roxygen"
  # Plain comment lines
  isComment = line %in% commentTokens$line1
  cls[isComment & cls == ""] = "comment"

  cls
}

#' Compare code metrics before and after a change
#'
#' @description
#' Computes the same metrics for two versions of the code and their change.
#' @param before,after Character vectors of R source lines.
#' @param metrics The metrics to compare, as in [metricValue()].
#' @return A data frame with columns `metric`, `before`, `after`, `change`
#'   and `relative` (change relative to `before`, `NA` when that is zero).
#' @family metric functions
#' @export
compareMetrics = function(before, after, metrics = c("sloc", "commentRatio", "cyclomatic",
                                                     "maxNesting", "halsteadVolume",
                                                     "uniqueCalls")) {
  metrics = match.arg(metrics, several.ok = TRUE)
  old = numeric(length(metrics))
  new = numeric(length(metrics))

  # Measure the code before the change
  for (i in seq_along(metrics)) {
    old[i] = metricValue(before, metrics[i])
  }
  # Measure the code after the change
  for (i in seq_along(metrics)) {
    new[i] = metricValue(after, metrics[i])
  }

  change = new - old
  relative = ifelse(old == 0, NA_real_, change / old)
  data.frame(
    metric = metrics, before = old, after = new, change = change,
    relative = relative, stringsAsFactors = FALSE
  )
}

#' Check code metrics against limits
#'
#' @description
#' `sloc`, `cyclomatic` and `maxNesting` are upper limits; `commentRatio`
#' is a lower limit.
#' @param text A character vector of R source lines.
#' @param limits A named numeric vector with elements `sloc`, `cyclomatic`,
#'   `maxNesting` and `commentRatio`.
#' @return A data frame with columns `metric`, `value`, `limit`, `ok` and
#'   `message`.
#' @family metric functions
#' @export
checkThresholds = function(text, limits = c(sloc = 400, cyclomatic = 30, maxNesting = 5,
                                            commentRatio = 0.1)) {
  metrics = c("sloc", "cyclomatic", "maxNesting", "commentRatio")
  value = stats::setNames(numeric(4), metrics)
  ok = stats::setNames(logical(4), metrics)
  message = stats::setNames(character(4), metrics)

  # Check `sloc` against its upper limit
  value[["sloc"]] = metricValue(text, "sloc")
  ok[["sloc"]] = value[["sloc"]] <= limits[["sloc"]]
  message[["sloc"]] = sprintf("%s is %s, above the limit of %s.", "sloc",
    format(value[["sloc"]]), format(limits[["sloc"]]))
  # Check `cyclomatic` against its upper limit
  value[["cyclomatic"]] = metricValue(text, "cyclomatic")
  ok[["cyclomatic"]] = value[["cyclomatic"]] <= limits[["cyclomatic"]]
  message[["cyclomatic"]] = sprintf("%s is %s, above the limit of %s.", "cyclomatic",
    format(value[["cyclomatic"]]), format(limits[["cyclomatic"]]))
  # Check `maxNesting` against its upper limit
  value[["maxNesting"]] = metricValue(text, "maxNesting")
  ok[["maxNesting"]] = value[["maxNesting"]] <= limits[["maxNesting"]]
  message[["maxNesting"]] = sprintf("%s is %s, above the limit of %s.", "maxNesting",
    format(value[["maxNesting"]]), format(limits[["maxNesting"]]))
  # Check `commentRatio` against its lower limit
  value[["commentRatio"]] = metricValue(text, "commentRatio")
  ok[["commentRatio"]] = value[["commentRatio"]] >= limits[["commentRatio"]]
  message[["commentRatio"]] = sprintf("%s is %s, below the limit of %s.", "commentRatio",
    format(value[["commentRatio"]]), format(limits[["commentRatio"]]))

  message[ok] = ""
  data.frame(
    metric = metrics, value = unname(value), limit = unname(limits[metrics]),
    ok = unname(ok), message = unname(message), stringsAsFactors = FALSE
  )
}
