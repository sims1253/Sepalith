#' Validate a lint configuration
#'
#' Checks every field and reports all problems at once.
#' @param config A list with any of the fields `rules`, `exclude`,
#'   `max_line_length`, `indent`, `naming` and `severity`.
#' @return `config`, invisibly.
#' @export
validate_lint_config <- function(config) {
  fields <- c("rules", "exclude", "max_line_length", "indent", "naming", "severity")
  expected <- c(
    rules = "a character vector of rule names",
    exclude = "a character vector of file patterns",
    max_line_length = "a single whole number of at least 40",
    indent = "a single whole number between 1 and 8",
    naming = "one of the supported naming styles",
    severity = "one of the severity levels"
  )
  styles <- c("snake_case", "camelCase", "dotted.case", "UPPER_CASE", "lowercase")
  levels <- c("info", "style", "warning", "error")
  is_count <- function(x) is.numeric(x) && length(x) == 1L && !is.na(x) && x == round(x)
  is_choice <- function(x, choices) is.character(x) && length(x) == 1L && x %in% choices

  unknown <- setdiff(names(config), fields)
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown config fields: %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }
  problems <- character()

  # Validate `rules`
  value <- config[["rules"]]
  if (!is.null(value) && !is.character(value)) {
    problems <- c(problems, sprintf("`%s` must be %s.", "rules", expected[["rules"]]))
  }
  # Validate `exclude`
  value <- config[["exclude"]]
  if (!is.null(value) && !is.character(value)) {
    problems <- c(problems, sprintf("`%s` must be %s.", "exclude", expected[["exclude"]]))
  }
  # Validate `max_line_length`
  value <- config[["max_line_length"]]
  if (!is.null(value) && !(is_count(value) && value >= 40)) {
    problems <- c(problems, sprintf("`%s` must be %s.", "max_line_length",
      expected[["max_line_length"]]))
  }
  # Validate `indent`
  value <- config[["indent"]]
  if (!is.null(value) && !(is_count(value) && value >= 1 && value <= 8)) {
    problems <- c(problems, sprintf("`%s` must be %s.", "indent", expected[["indent"]]))
  }
  # Validate `naming`
  value <- config[["naming"]]
  if (!is.null(value) && !is_choice(value, styles)) {
    problems <- c(problems, sprintf("`%s` must be %s.", "naming", expected[["naming"]]))
  }
  # Validate `severity`
  value <- config[["severity"]]
  if (!is.null(value) && !is_choice(value, levels)) {
    problems <- c(problems, sprintf("`%s` must be %s.", "severity", expected[["severity"]]))
  }

  if (length(problems) > 0L) {
    stop(paste(problems, collapse = "\n"), call. = FALSE)
  }
  return(invisible(config))
}

#' Read a lint configuration file
#'
#' Reads a DCF file such as `.lintwright` and converts each field.
#' @param path Path to the configuration file.
#' @return A validated configuration list with the fields found in the file.
#' @export
read_lint_config <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Config file `%s` does not exist.", path), call. = FALSE)
  }
  raw <- as.list(read.dcf(path)[1L, ])
  split_list <- function(x) trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  config <- list()

  # Read `rules`
  if (!is.null(raw[["rules"]])) {
    config$rules <- split_list(raw[["rules"]])
  }
  # Read `exclude`
  if (!is.null(raw[["exclude"]])) {
    config$exclude <- split_list(raw[["exclude"]])
  }
  # Read `max_line_length`
  if (!is.null(raw[["max_line_length"]])) {
    config$max_line_length <- as.integer(raw[["max_line_length"]])
  }
  # Read `indent`
  if (!is.null(raw[["indent"]])) {
    config$indent <- as.integer(raw[["indent"]])
  }
  # Read `naming`
  if (!is.null(raw[["naming"]])) {
    config$naming <- trimws(raw[["naming"]])
  }
  # Read `severity`
  if (!is.null(raw[["severity"]])) {
    config$severity <- trimws(raw[["severity"]])
  }

  validate_lint_config(config)
  return(config)
}

#' Write a lint configuration file
#'
#' The inverse of [read_lint_config()]: each field is written as one DCF
#' line.
#' @param config A list with any of the fields `rules`, `exclude`,
#'   `max_line_length`, `indent`, `naming` and `severity`.
#' @param path Path to the configuration file.
#' @return `path`, invisibly.
#' @export
write_lint_config <- function(config, path) {
  validate_lint_config(config)
  out <- character()

  # Write `rules`
  if (!is.null(config$rules)) {
    out <- c(out, rules = paste(config$rules, collapse = ", "))
  }
  # Write `exclude`
  if (!is.null(config$exclude)) {
    out <- c(out, exclude = paste(config$exclude, collapse = ", "))
  }
  # Write `max_line_length`
  if (!is.null(config$max_line_length)) {
    out <- c(out, max_line_length = as.character(config$max_line_length))
  }
  # Write `indent`
  if (!is.null(config$indent)) {
    out <- c(out, indent = as.character(config$indent))
  }
  # Write `naming`
  if (!is.null(config$naming)) {
    out <- c(out, naming = config$naming)
  }
  # Write `severity`
  if (!is.null(config$severity)) {
    out <- c(out, severity = config$severity)
  }

  write.dcf(matrix(out, nrow = 1L, dimnames = list(NULL, names(out))), file = path)
  return(invisible(path))
}

#' Merge two lint configurations
#'
#' Layers a project configuration over a base one, field by field.
#' @param base,override Configuration lists; fields missing from `override`
#'   keep their value from `base`.
#' @return The merged, validated configuration list.
#' @export
merge_lint_config <- function(base, override) {
  validate_lint_config(base)
  validate_lint_config(override)
  levels <- c("info", "style", "warning", "error")
  merged <- base

  # Merge `rules`: entries starting with "-" drop a rule from `base`
  if (!is.null(override$rules)) {
    drop <- sub("^-", "", override$rules[startsWith(override$rules, "-")])
    add <- override$rules[!startsWith(override$rules, "-")]
    merged$rules <- union(setdiff(base$rules, drop), add)
  }
  # Merge `exclude`: patterns from both are kept
  if (!is.null(override$exclude)) {
    merged$exclude <- union(base$exclude, override$exclude)
  }
  # Merge `max_line_length`: the override wins
  if (!is.null(override$max_line_length)) {
    merged$max_line_length <- override$max_line_length
  }
  # Merge `indent`: the override wins
  if (!is.null(override$indent)) {
    merged$indent <- override$indent
  }
  # Merge `naming`: the override wins
  if (!is.null(override$naming)) {
    merged$naming <- override$naming
  }
  # Merge `severity`: the stricter level wins
  if (!is.null(override$severity)) {
    both <- c(base$severity, override$severity)
    merged$severity <- levels[max(match(both, levels))]
  }

  return(validate_lint_config(merged))
}

#' Format a lint configuration for printing
#'
#' Shows one aligned line per field, marking fields left at their default.
#' @param config A list with any of the fields `rules`, `exclude`,
#'   `max_line_length`, `indent`, `naming` and `severity`.
#' @return A character vector with one line per field.
#' @export
format_lint_config <- function(config) {
  fields <- c("rules", "exclude", "max_line_length", "indent", "naming", "severity")
  width <- max(nchar(fields))
  label <- function(field) formatC(field, width = -width)
  show <- function(value) if (is.null(value)) "<default>" else paste(value, collapse = ", ")
  lines <- character()

  # Show `rules`
  value <- config[["rules"]]
  lines <- c(lines, paste0(label("rules"), " : ", show(value)))
  # Show `exclude`
  value <- config[["exclude"]]
  lines <- c(lines, paste0(label("exclude"), " : ", show(value)))
  # Show `max_line_length`
  value <- config[["max_line_length"]]
  lines <- c(lines, paste0(label("max_line_length"), " : ", show(value)))
  # Show `indent`
  value <- config[["indent"]]
  lines <- c(lines, paste0(label("indent"), " : ", show(value)))
  # Show `naming`
  value <- config[["naming"]]
  lines <- c(lines, paste0(label("naming"), " : ", show(value)))
  # Show `severity`
  value <- config[["severity"]]
  lines <- c(lines, paste0(label("severity"), " : ", show(value)))

  return(lines)
}
