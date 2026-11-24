#' Measure the functions defined in R code
#'
#' @description
#' Measures every top-level `name <- function(...)` definition.
#' @param text A character vector of R source lines.
#' @return A data frame with one row per function and columns `name`,
#'   `line`, `args`, `lines`, `statements`, `complexity`, `assignments` and
#'   `returns`.
#' @family function metrics
#' @export
functionMetrics <- function(text) {
  exprs <- parse(text = text, keep.source = TRUE)
  isDef <- vapply(exprs, function(e) {
    is.call(e) && (identical(e[[1L]], as.name("<-")) || identical(e[[1L]], as.name("="))) &&
      is.call(e[[3L]]) && identical(e[[3L]][[1L]], as.name("function"))
  }, logical(1))
  defs <- exprs[isDef]
  refs <- attr(exprs, "srcref")[isDef]
  countNodes <- function(node, heads) {
    if (!is.call(node)) return(0L)
    own <- as.integer(is.symbol(node[[1L]]) && as.character(node[[1L]]) %in% heads)
    own + sum(vapply(as.list(node)[-1L], countNodes, integer(1), heads = heads))
  }

  n <- length(defs)
  name <- character(n)
  line <- args <- lines <- statements <- complexity <- assignments <- returns <- integer(n)
  for (i in seq_len(n)) {
    fun <- defs[[i]][[3L]]
    body <- fun[[3L]]
    # Name and first line
    name[i] <- as.character(defs[[i]][[2L]])
    line[i] <- refs[[i]][1L]
    # Formal arguments
    args[i] <- length(fun[[2L]])
    # Lines spanned
    lines[i] <- refs[[i]][3L] - refs[[i]][1L] + 1L
    # Top-level statements of the body
    isBlock <- is.call(body) && identical(body[[1L]], as.name("{"))
    statements[i] <- if (isBlock) length(body) - 1L else 1L
    # Cyclomatic complexity
    complexity[i] <- 1L + countNodes(body, c("if", "for", "while", "repeat", "&&", "||"))
    # Assignments
    assignments[i] <- countNodes(body, c("<-", "<<-", "=", "->", "->>"))
    # Explicit returns
    returns[i] <- countNodes(body, "return")
  }

  return(data.frame(
    name = name, line = line, args = args, lines = lines, statements = statements,
    complexity = complexity, assignments = assignments, returns = returns,
    stringsAsFactors = FALSE
  ))
}

#' Rank functions by one metric
#'
#' @description
#' Orders the output of [functionMetrics()] by a metric, breaking ties with
#' a second metric and then by name.
#' @param metrics A data frame as returned by [functionMetrics()].
#' @param by The metric to rank by.
#' @param n The number of functions to keep.
#' @return The top `n` rows of `metrics` with columns `rank`, `name`, `line`
#'   and the `by` column.
#' @family function metrics
#' @export
rankFunctions <- function(metrics, by = c("lines", "args", "complexity", "statements",
                                          "returns"), n = 5L) {
  by <- match.arg(by)
  switch(by,
    lines = {
      score <- metrics$lines
      tie <- metrics$statements
    },
    args = {
      score <- metrics$args
      tie <- metrics$lines
    },
    complexity = {
      score <- metrics$complexity
      tie <- metrics$lines
    },
    statements = {
      score <- metrics$statements
      tie <- metrics$lines
    },
    returns = {
      score <- metrics$returns
      tie <- metrics$complexity
    }
  )

  ord <- order(-score, -tie, metrics$name)
  top <- metrics[utils::head(ord, n), c("name", "line", by), drop = FALSE]
  top <- cbind(rank = seq_len(nrow(top)), top)
  rownames(top) <- NULL
  return(top)
}

#' Flag functions that exceed limits
#'
#' @description
#' Compares each function with an upper limit per metric.
#' @param metrics A data frame as returned by [functionMetrics()].
#' @param limits A named numeric vector with elements `lines`, `args`,
#'   `complexity` and `statements`.
#' @return A data frame with columns `name`, `line` and `flags` for the
#'   functions that exceed at least one limit.
#' @family function metrics
#' @export
flagFunctions <- function(metrics, limits = c(lines = 50, args = 5, complexity = 10,
                                              statements = 25)) {
  flags <- character(nrow(metrics))

  # Flag functions with too many lines
  over <- metrics$lines > limits[["lines"]]
  flags[over] <- paste0(flags[over], sprintf("%s %d > %g; ", "lines",
    metrics$lines[over], limits[["lines"]]))
  # Flag functions with too many arguments
  over <- metrics$args > limits[["args"]]
  flags[over] <- paste0(flags[over], sprintf("%s %d > %g; ", "args",
    metrics$args[over], limits[["args"]]))
  # Flag functions with too much complexity
  over <- metrics$complexity > limits[["complexity"]]
  flags[over] <- paste0(flags[over], sprintf("%s %d > %g; ", "complexity",
    metrics$complexity[over], limits[["complexity"]]))
  # Flag functions with too many statements
  over <- metrics$statements > limits[["statements"]]
  flags[over] <- paste0(flags[over], sprintf("%s %d > %g; ", "statements",
    metrics$statements[over], limits[["statements"]]))

  flagged <- nzchar(flags)
  return(data.frame(
    name = metrics$name[flagged], line = metrics$line[flagged],
    flags = sub("; $", "", flags[flagged]), stringsAsFactors = FALSE
  ))
}

#' Profile the arguments of a function
#'
#' @description
#' Groups the formal arguments of a function by the kind of default they have.
#' @param fun A function.
#' @return A list of argument names with elements `required`, `optional`,
#'   `dots`, `flags` (logical defaults), `nullable` (`NULL` defaults) and
#'   `computed` (defaults that are calls).
#' @family function metrics
#' @export
argumentProfile <- function(fun) {
  if (!is.function(fun)) {
    stop("`fun` must be a function.", call. = FALSE)
  }
  fmls <- as.list(formals(fun))
  argNames <- as.character(names(fmls))
  isMissing <- vapply(seq_along(fmls), function(i) identical(fmls[[i]], quote(expr = )),
    logical(1))

  # Arguments without a default
  isRequired <- isMissing & argNames != "..."
  required <- argNames[isRequired]
  # Arguments with a default
  isOptional <- !isMissing
  optional <- argNames[isOptional]
  # The dots
  isDots <- argNames == "..."
  dots <- argNames[isDots]
  # Arguments that default to `TRUE` or `FALSE`
  isFlag <- vapply(seq_along(fmls), function(i) {
    !isMissing[i] && is.logical(fmls[[i]]) && length(fmls[[i]]) == 1L && !is.na(fmls[[i]])
  }, logical(1))
  flags <- argNames[isFlag]
  # Arguments that default to `NULL`
  isNullable <- vapply(seq_along(fmls), function(i) !isMissing[i] && is.null(fmls[[i]]),
    logical(1))
  nullable <- argNames[isNullable]
  # Arguments whose default is computed by a call
  isComputed <- vapply(seq_along(fmls), function(i) !isMissing[i] && is.call(fmls[[i]]),
    logical(1))
  computed <- argNames[isComputed]

  return(list(
    required = required, optional = optional, dots = dots, flags = flags,
    nullable = nullable, computed = computed
  ))
}

#' Format the signature of a function
#'
#' @description
#' Renders the formal arguments of a function in one of several styles.
#' @param fun A function.
#' @param name The name to show for `fun`.
#' @param style The output style: a usage line, a call passing every
#'   argument by name, roxygen `@param` stubs or a Markdown table.
#' @return A character vector: one line for `usage` and `call`, one line per
#'   argument for `roxygen`, and a header plus one line per argument for
#'   `markdown`.
#' @family function metrics
#' @export
functionSignature <- function(fun, name, style = c("usage", "call", "roxygen", "markdown")) {
  style <- match.arg(style)
  fmls <- as.list(formals(fun))
  argNames <- as.character(names(fmls))
  defaults <- vapply(seq_along(fmls), function(i) {
    if (identical(fmls[[i]], quote(expr = ))) "" else paste(deparse(fmls[[i]]), collapse = " ")
  }, character(1))
  hasDefault <- nzchar(defaults)

  out <- switch(style,
    usage = {
      parts <- ifelse(hasDefault, paste0(argNames, " = ", defaults), argNames)
      paste0(name, "(", paste(parts, collapse = ", "), ")")
    },
    call = {
      parts <- ifelse(argNames == "...", "...", paste0(argNames, " = ", argNames))
      paste0(name, "(", paste(parts, collapse = ", "), ")")
    },
    roxygen = {
      parts <- ifelse(hasDefault, sprintf(" Defaults to `%s`.", defaults), "")
      sprintf("#' @param %s%s", argNames, parts)
    },
    markdown = {
      parts <- ifelse(hasDefault, sprintf("`%s`", defaults), "")
      c("| Argument | Default |", "|---|---|", sprintf("| `%s` | %s |", argNames, parts))
    }
  )
  return(out)
}
