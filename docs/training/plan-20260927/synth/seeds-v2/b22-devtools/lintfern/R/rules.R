#' Create a lint rule
#'
#' A rule bundles a check function with metadata. Parse rules receive the
#' parse data of a file; line rules receive its raw lines.
#'
#' @param name A short identifier for the rule, e.g. `"line_length"`.
#' @param check A function taking parse data or lines and returning lint hits.
#' @param input Whether `check` consumes parse data or raw lines.
#' @param severity How serious a hit from this rule is.
#' @return An object of class `lintRule`.
#' @export
newLintRule <- function(name, check, input = c("parse", "lines"),
                        severity = c("warning", "style", "error")) {
  input <- match.arg(input)
  severity <- match.arg(severity)

  rule <- structure(
    list(name = name, check = check, input = input, severity = severity),
    class = "lintRule"
  )
  validateLintRule(rule)
}

#' Validate a lint rule
#'
#' @description Rule names double as identifiers in reports and config files,
#'   so they are restricted to lower snake case.
#' @param rule An object of class `lintRule`.
#' @keywords internal
validateLintRule <- function(rule) {
  if (!is.character(rule$name) || length(rule$name) != 1L || !nzchar(rule$name)) {
    stop("A lint rule needs a single non-empty name.", call. = FALSE)
  }
  if (!grepl("^[a-z][a-z0-9_]*$", rule$name)) {
    stop("Rule name '", rule$name, "' must be lower snake case.", call. = FALSE)
  }

  if (!is.function(rule$check)) {
    stop("Rule '", rule$name, "' must have a function as `check`.", call. = FALSE)
  }
  # Primitives have no formals() but still take arguments.
  if (!is.primitive(rule$check) && length(formals(rule$check)) == 0L) {
    stop("The check of rule '", rule$name, "' must accept an argument.", call. = FALSE)
  }
  rule
}

#' Format a lint rule
#'
#' @param x An object of class `lintRule`.
#' @param ... Ignored.
#' @return A single string.
#' @export
format.lintRule <- function(x, ...) {
  sprintf("<lintRule %s> [%s, %s]", x$name, x$input, x$severity)
}

#' Print a lint rule
#'
#' @param x An object of class `lintRule`.
#' @param ... Passed to [format.lintRule()].
#' @return `x`, invisibly.
#' @export
print.lintRule <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}

#' Rule: unused function arguments
#'
#' @return An object of class `lintRule`.
#' @export
ruleUnusedArguments <- function() {
  newLintRule(
    "unused_arguments",
    check = function(pd) checkUnusedArguments(pd),
    input = "parse",
    severity = "warning"
  )
}

#' Rule: overly long functions
#'
#' @description Rule constructors capture their settings in a closure, so the
#'   check itself only ever receives the file input.
#' @param maxLines Maximum number of code lines allowed per function.
#' @return An object of class `lintRule`.
#' @export
ruleLongFunctions <- function(maxLines = 50L) {
  # Force now so a later change to the caller's variable cannot leak in.
  force(maxLines)
  newLintRule(
    "long_functions",
    check = function(pd) checkLongFunctions(pd, maxLines = maxLines),
    input = "parse",
    severity = "style"
  )
}

#' Rule: TODO comments
#'
#' @param tags Comment tags to look for, matched case-insensitively.
#' @return An object of class `lintRule`.
#' @export
ruleTodoComments <- function(tags = c("TODO", "FIXME", "XXX")) {
  force(tags)
  newLintRule(
    "todo_comments",
    check = function(pd) checkTodoComments(pd, tags = tags),
    input = "parse",
    severity = "style"
  )
}

#' Rule: line width
#'
#' @param maxWidth Maximum number of characters allowed per line.
#' @return An object of class `lintRule`.
#' @export
ruleLineLength <- function(maxWidth = 80L) {
  force(maxWidth)
  newLintRule(
    "line_length",
    check = function(lines) checkLineLength(lines, maxWidth = maxWidth),
    input = "lines",
    severity = "style"
  )
}

#' The default rule set
#'
#' @return A named list of `lintRule` objects.
#' @export
defaultRules <- function() {
  rules <- list(
    ruleUnusedArguments(),
    ruleLongFunctions(),
    ruleTodoComments(),
    ruleLineLength()
  )

  names(rules) <- vapply(rules, `[[`, character(1), "name")
  rules
}
