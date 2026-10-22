#' Run one rule against a file context
#'
#' @param rule An object of class `lintRule`.
#' @param context A list with elements `file`, `lines` and `pd`.
#' @return A data frame of lint hits with `file`, `rule` and `severity` added.
#' @keywords internal
applyRule <- function(rule, context) {
  input <- switch(rule$input,
    parse = context$pd,
    lines = context$lines
  )
  # A file that failed to parse has no parse data; skip its parse rules.
  if (is.null(input)) {
    return(NULL)
  }

  # One broken rule should not abort the whole run.
  hits <- tryCatch(
    rule$check(input),
    error = function(e) {
      warning(
        "Rule '", rule$name, "' failed on '", context$file, "': ",
        conditionMessage(e),
        call. = FALSE
      )
      lintHits()
    }
  )
  if (is.null(hits) || nrow(hits) == 0L) {
    return(NULL)
  }

  hits$file <- context$file
  hits$rule <- rule$name
  hits$severity <- rule$severity
  hits
}

#' Lint a single R source file
#'
#' @param path Path to an R source file.
#' @param rules A list of `lintRule` objects, defaulting to [defaultRules()].
#' @return An object of class `lintResult`.
#' @export
lintFile <- function(path, rules = defaultRules()) {
  context <- list(
    file = path,
    lines = readLines(path, warn = FALSE),
    pd = readSourceTokens(path)
  )

  hits <- lapply(rules, applyRule, context = context)
  hits <- Filter(Negate(is.null), hits)
  newLintResult(hits, files = path)
}

#' Lint every R source file below a directory
#'
#' @param path Path to a directory, usually a package root.
#' @param rules A list of `lintRule` objects, defaulting to [defaultRules()].
#' @param pattern Regular expression selecting the files to lint.
#' @return An object of class `lintResult`.
#' @export
lintDir <- function(path = ".", rules = defaultRules(), pattern = "\\.[Rr]$") {
  files <- list.files(path, pattern = pattern, recursive = TRUE, full.names = TRUE)
  # Vendored libraries and reverse-dependency checks are not our code.
  files <- files[!grepl("(^|/)(renv|packrat|revdep)/", files)]

  hits <- list()
  for (file in files) {
    context <- list(
      file = file,
      lines = readLines(file, warn = FALSE),
      pd = readSourceTokens(file)
    )

    for (rule in rules) {
      found <- applyRule(rule, context)
      if (!is.null(found)) {
        hits[[length(hits) + 1L]] <- found
      }
    }
  }

  newLintResult(hits, files = files)
}

#' Construct a lint result
#'
#' @param hits A list of data frames of lint hits.
#' @param files The files that were linted.
#' @return An object of class `lintResult`: a data frame with an extra
#'   `files` attribute.
#' @noRd
newLintResult <- function(hits, files) {
  cols <- c("file", "line", "column", "rule", "severity", "message")
  if (length(hits) == 0L) {
    df <- data.frame(
      file = character(), line = integer(), column = integer(),
      rule = character(), severity = character(), message = character(),
      stringsAsFactors = FALSE
    )
  } else {
    df <- do.call(rbind, hits)[, cols]
    df <- df[order(df$file, df$line, df$column), , drop = FALSE]
  }

  rownames(df) <- NULL
  structure(df, files = files, class = c("lintResult", "data.frame"))
}

#' Format lint results as compiler-style messages
#'
#' @param x An object of class `lintResult`.
#' @param ... Ignored.
#' @return A character vector, one element per hit.
#' @export
format.lintResult <- function(x, ...) {
  if (nrow(x) == 0L) {
    return(character())
  }
  sprintf(
    "%s:%d:%d: %s: %s [%s]",
    x$file, x$line, x$column, x$severity, x$message, x$rule
  )
}

#' Print lint results
#'
#' @param x An object of class `lintResult`.
#' @param ... Passed to [format.lintResult()].
#' @return `x`, invisibly.
#' @export
print.lintResult <- function(x, ...) {
  nFiles <- length(attr(x, "files"))
  if (nrow(x) == 0L) {
    cat("No lints found in ", nFiles, " file(s).\n", sep = "")
  } else {
    cat(format(x, ...), sep = "\n")
    cat("\n", nrow(x), " lint(s) in ", nFiles, " file(s).\n", sep = "")
  }
  invisible(x)
}

#' Summarise lint results by rule and severity
#'
#' @param object An object of class `lintResult`.
#' @param ... Ignored.
#' @return A data frame with one row per rule, counting hits and affected files.
#' @export
summary.lintResult <- function(object, ...) {
  rules <- sort(unique(object$rule))

  out <- data.frame(
    rule = rules,
    severity = character(length(rules)),
    hits = integer(length(rules)),
    files = integer(length(rules)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(rules)) {
    # Severity is fixed per rule, so the first row is representative.
    rows <- object[object$rule == rules[[i]], , drop = FALSE]
    out$severity[[i]] <- rows$severity[[1L]]
    out$hits[[i]] <- nrow(rows)
    out$files[[i]] <- length(unique(rows$file))
  }

  out[order(-out$hits, out$rule), , drop = FALSE]
}
