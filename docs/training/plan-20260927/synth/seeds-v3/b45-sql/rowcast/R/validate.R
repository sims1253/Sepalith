#' Remove literals and comments from SQL text
#'
#' Blanks out quoted strings and comments so rule checks only see SQL syntax.
#'
#' @param sql A single SQL string.
#' @param keep.case Whether to keep the original letter case.
#' @return The cleaned SQL string, upper-cased unless `keep.case` and with whitespace squeezed.
#' @keywords internal
strip.sql.noise <- function(sql, keep.case = FALSE) {
  # Line comments, then block comments
  sql <- gsub("--[^\n]*", " ", sql)
  sql <- gsub("/\\*.*?\\*/", " ", sql, perl = TRUE)
  # Single-quoted literals, then double-quoted identifiers
  sql <- gsub("'([^']|'')*'", "''", sql)
  sql <- gsub("\"([^\"]|\"\")*\"", "\"\"", sql)
  sql <- gsub("\\s+", " ", sql)
  if (keep.case) trimws(sql) else toupper(trimws(sql))
}

#' Validate a SQL query against house rules
#'
#' Runs each requested rule on the query text after literals and comments are removed.
#'
#' @param sql A single SQL string.
#' @param rules Any of `"no_select_star"`, `"delete_needs_where"`, `"update_needs_where"`,
#'   `"select_needs_limit"`, `"single_statement"` and `"balanced_parens"`.
#' @return A data frame with columns `rule`, `passed` and `message`, one row per rule.
#' @export
validate.query <- function(sql,
                           rules = c("no_select_star", "delete_needs_where",
                                     "update_needs_where", "select_needs_limit",
                                     "single_statement", "balanced_parens")) {
  rules <- match.arg(rules, several.ok = TRUE)
  if (!is.character(sql) || length(sql) != 1L) {
    stop("`sql` must be a single string.", call. = FALSE)
  }
  text <- strip.sql.noise(sql)
  passed <- setNames(rep(TRUE, length(rules)), rules)
  # Rule `no_select_star`
  if ("no_select_star" %in% rules) {
    passed[["no_select_star"]] <- !grepl("\\bSELECT\\s+(DISTINCT\\s+)?\\*", text)
  }
  # Rule `delete_needs_where`
  if ("delete_needs_where" %in% rules) {
    is.delete <- grepl("^DELETE\\b", text)
    passed[["delete_needs_where"]] <- !is.delete || grepl("\\bWHERE\\b", text)
  }
  # Rule `update_needs_where`
  if ("update_needs_where" %in% rules) {
    is.update <- grepl("^UPDATE\\b", text)
    passed[["update_needs_where"]] <- !is.update || grepl("\\bWHERE\\b", text)
  }
  # Rule `select_needs_limit`
  if ("select_needs_limit" %in% rules) {
    is.select <- grepl("^SELECT\\b", text)
    passed[["select_needs_limit"]] <- !is.select || grepl("\\b(LIMIT|FETCH)\\b", text)
  }
  # Rule `single_statement`
  if ("single_statement" %in% rules) {
    body <- sub(";\\s*$", "", text)
    passed[["single_statement"]] <- !grepl(";", body, fixed = TRUE)
  }
  # Rule `balanced_parens`
  if ("balanced_parens" %in% rules) {
    chars <- strsplit(text, "")[[1]]
    depth <- cumsum((chars == "(") - (chars == ")"))
    passed[["balanced_parens"]] <- all(depth >= 0L) && sum(chars == "(") == sum(chars == ")")
  }
  message <- ifelse(passed, "", sprintf("Query violates rule `%s`.", rules))
  data.frame(rule = rules, passed = unname(passed), message = unname(message))
}

#' Check identifiers against naming rules
#'
#' Applies each naming rule to every identifier and reports the failures.
#'
#' @param names A character vector of identifiers.
#' @param dialect One of `"postgres"`, `"mysql"` or `"sqlite"`; sets the length limit.
#' @param reserved Upper-case reserved words that must not be used bare.
#' @return A data frame with columns `name` and `rule`, one row per failure.
#' @export
check.identifiers <- function(names, dialect = c("postgres", "mysql", "sqlite"),
                              reserved = c("SELECT", "FROM", "WHERE", "GROUP", "ORDER",
                                           "USER", "TABLE", "INDEX", "KEY")) {
  dialect <- match.arg(dialect)
  max.length <- c(postgres = 63L, mysql = 64L, sqlite = 1024L)[[dialect]]
  failures <- data.frame(name = character(), rule = character())
  # Rule `max_length`
  bad <- nchar(names) > max.length
  if (any(bad)) {
    failures <- rbind(failures, data.frame(name = names[bad], rule = "max_length"))
  }
  # Rule `leading_letter`
  bad <- !grepl("^[A-Za-z_]", names)
  if (any(bad)) {
    failures <- rbind(failures, data.frame(name = names[bad], rule = "leading_letter"))
  }
  # Rule `plain_characters`
  bad <- grepl("[^A-Za-z0-9_]", names)
  if (any(bad)) {
    failures <- rbind(failures, data.frame(name = names[bad], rule = "plain_characters"))
  }
  # Rule `lower_case`
  bad <- names != tolower(names)
  if (any(bad)) {
    failures <- rbind(failures, data.frame(name = names[bad], rule = "lower_case"))
  }
  # Rule `not_reserved`
  bad <- toupper(names) %in% reserved
  if (any(bad)) {
    failures <- rbind(failures, data.frame(name = names[bad], rule = "not_reserved"))
  }
  rownames(failures) <- NULL
  failures
}

#' Check bind parameters against a query
#'
#' Counts the placeholders of the given style and matches them to `params`.
#'
#' @param sql A single SQL string.
#' @param params A list of parameter values; named for the `colon` and `at` styles.
#' @param style One of `"qmark"`, `"dollar"`, `"colon"` or `"at"`.
#' @return `params`, invisibly, reordered to match the placeholders.
#' @export
check.params <- function(sql, params, style = c("qmark", "dollar", "colon", "at")) {
  style <- match.arg(style)
  text <- strip.sql.noise(sql)
  # Named placeholders keep their case, so match them on a case-preserving clean copy
  named <- strip.sql.noise(sql, keep.case = TRUE)
  mismatch <- "Query has %d placeholder(s) but %d parameter(s)."
  switch(style,
    qmark = {
      tokens <- regmatches(text, gregexpr("\\?", text))[[1]]
      found <- length(tokens)
      if (found != length(params)) {
        stop(sprintf(mismatch, found, length(params)), call. = FALSE)
      }
    },
    dollar = {
      tokens <- regmatches(text, gregexpr("\\$[0-9]+", text))[[1]]
      found <- if (length(tokens) == 0L) 0L else max(as.integer(sub("^\\$", "", tokens)))
      if (found != length(params)) {
        stop(sprintf(mismatch, found, length(params)), call. = FALSE)
      }
    },
    colon = {
      pattern <- "(?<!:):[A-Za-z_][A-Za-z0-9_]*"
      tokens <- regmatches(named, gregexpr(pattern, named, perl = TRUE))[[1]]
      keys <- unique(sub("^:", "", tokens))
      missing <- setdiff(keys, names(params))
      if (length(missing) > 0) {
        stop(sprintf("Missing parameter(s): %s.", toString(missing)), call. = FALSE)
      }
      params <- params[keys]
    },
    at = {
      pattern <- "(?<!@)@[A-Za-z_][A-Za-z0-9_]*"
      tokens <- regmatches(named, gregexpr(pattern, named, perl = TRUE))[[1]]
      keys <- unique(sub("^@", "", tokens))
      missing <- setdiff(keys, names(params))
      if (length(missing) > 0) {
        stop(sprintf("Missing parameter(s): %s.", toString(missing)), call. = FALSE)
      }
      params <- params[keys]
    }
  )
  invisible(params)
}

#' Validate a column specification table
#'
#' Checks the fields `name`, `type`, `nullable` and `default` row by row.
#'
#' @param spec A data frame with columns `name`, `type`, `nullable` and `default`.
#' @param types The allowed values of `spec$type`.
#' @return `spec`, invisibly, if every field is valid.
#' @export
validate.column.spec <- function(spec, types = c("integer", "numeric", "text", "boolean",
                                                 "date", "timestamp")) {
  fields <- c("name", "type", "nullable", "default")
  absent <- setdiff(fields, names(spec))
  if (length(absent) > 0) {
    stop(sprintf("`spec` is missing field(s): %s.", toString(absent)), call. = FALSE)
  }
  report <- function(field, bad) {
    if (any(bad)) {
      rows <- paste(which(bad), collapse = ", ")
      stop(sprintf("`spec$%s` is invalid in row(s) %s.", field, rows), call. = FALSE)
    }
  }
  # Field `name`
  bad <- is.na(spec$name) | !grepl("^[a-z_][a-z0-9_]*$", spec$name) | duplicated(spec$name)
  report("name", bad)
  # Field `type`
  bad <- is.na(spec$type) | !(spec$type %in% types)
  report("type", bad)
  # Field `nullable`
  bad <- is.na(spec$nullable) | !is.logical(spec$nullable)
  report("nullable", bad)
  # Field `default`
  bad <- !spec$nullable & !is.na(spec$default) & !nzchar(spec$default)
  report("default", bad)
  invisible(spec)
}
