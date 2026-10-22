#' List the placeholders used in a template
#'
#' Finds `{{ name }}` placeholders, ignoring surrounding whitespace.
#'
#' @param template A single template string.
#' @return A character vector of distinct placeholder names, in order of
#'   first appearance.
#' @family template helpers
#' @export
template.vars <- function(template) {
  hits <- regmatches(template, gregexpr("\\{\\{\\s*[A-Za-z_][A-Za-z0-9_.]*\\s*\\}\\}", template))
  hits[[1]] %>%
    gsub("^\\{\\{\\s*|\\s*\\}\\}$", "", .) %>%
    unique()
}

#' Check that all placeholders have values
#'
#' @param template A single template string.
#' @param values A named list or character vector of substitution values.
#' @return `values`, invisibly, or an error naming the missing entries.
#' @family template helpers
#' @export
check.template.vars <- function(template, values) {
  needed <- template.vars(template)
  missing <- setdiff(needed, names(values))
  if (length(missing) > 0L) {
    stop(
      "Template needs values for: ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(values)
}

#' Render a double-brace template
#'
#' Substitutes each `{{ name }}` placeholder with the matching entry of
#' `values`. Values are escaped for the chosen format unless the
#' placeholder is written with triple braces.
#'
#' @param template A single template string.
#' @param values A named list or character vector of substitution values.
#' @param format Output format passed to [escape.for()].
#' @return A single rendered string.
#' @family renderers
#' @export
render.braces <- function(template, values, format = c("html", "markdown", "latex")) {
  format <- match.arg(format)
  check.template.vars(template, values)
  # raw placeholders first so the double-brace pass does not see them
  raw.pat <- "\\{\\{\\{\\s*([A-Za-z_][A-Za-z0-9_.]*)\\s*\\}\\}\\}"
  m <- gregexpr(raw.pat, template, perl = TRUE)
  regmatches(template, m) <- lapply(regmatches(template, m), function(tags) {
    keys <- sub(raw.pat, "\\1", tags, perl = TRUE)
    vapply(keys, function(k) as.character(values[[k]]), character(1))
  })
  esc.pat <- "\\{\\{\\s*([A-Za-z_][A-Za-z0-9_.]*)\\s*\\}\\}"
  m <- gregexpr(esc.pat, template, perl = TRUE)
  regmatches(template, m) <- lapply(regmatches(template, m), function(tags) {
    keys <- sub(esc.pat, "\\1", tags, perl = TRUE)
    vapply(keys, function(k) escape.for(values[[k]], format = format), character(1))
  })
  template
}

#' Render a template with glue
#'
#' Evaluates `{expr}` blocks in a data environment built from `values`,
#' escaping each result for the chosen format.
#'
#' @param template A single template string.
#' @param values A named list or character vector of substitution values.
#' @param format Output format passed to [escape.for()].
#' @return A single rendered string.
#' @family renderers
#' @export
render.glue <- function(template, values, format = c("html", "markdown", "latex")) {
  format <- match.arg(format)
  env <- list2env(as.list(values), parent = baseenv())
  out <- glue::glue(
    template,
    .envir = env,
    .transformer = function(text, envir) {
      escape.for(eval(parse(text = text, keep.source = FALSE), envir), format = format)
    }
  )
  as.character(out)
}

#' Render a template with sprintf
#'
#' Positional rendering for templates that use `%s`, `%d` and friends.
#' Values are escaped before substitution when they are character.
#'
#' @param template A single template string.
#' @param values A list of values in placeholder order.
#' @param format Output format passed to [escape.for()].
#' @return A single rendered string.
#' @family renderers
#' @export
render.sprintf <- function(template, values, format = c("html", "markdown", "latex")) {
  format <- match.arg(format)
  n.slots <- lengths(regmatches(template, gregexpr("%[-+ 0#]*[0-9.]*[sdifeEgGxX]", template)))
  if (n.slots != length(values)) {
    stop(sprintf("Template has %d slots but %d values were given.", n.slots, length(values)),
      call. = FALSE
    )
  }
  values <- lapply(values, function(v) {
    if (is.character(v)) escape.for(v, format = format) else v
  })
  do.call(sprintf, c(list(template), values))
}

#' Render a template once per row of a data frame
#'
#' @param template A single template string.
#' @param data A data frame whose columns supply the substitution values.
#' @param engine Which renderer to use.
#' @param format Output format passed to [escape.for()].
#' @return A character vector with one rendered string per row of `data`.
#' @family renderers
#' @export
render.rows <- function(template, data, engine = c("braces", "glue"), format = "html") {
  engine <- match.arg(engine)
  renderer <- switch(engine,
    braces = render.braces,
    glue = render.glue
  )
  out <- character(nrow(data))
  for (i in seq_len(nrow(data))) {
    row.values <- lapply(data[i, , drop = FALSE], function(col) col[[1]])
    out[i] <- renderer(template, row.values, format = format)
  }
  out
}

#' Read a template file
#'
#' Reads a template from disk, dropping a trailing newline and any lines
#' that start with the comment marker.
#'
#' @param path Path to a UTF-8 template file.
#' @param comment Lines starting with this string are removed.
#' @return A single template string.
#' @export
read.template <- function(path, comment = "##") {
  if (!file.exists(path)) {
    stop("Template file not found: ", path, call. = FALSE)
  }
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  if (!is.null(comment)) {
    lines <- lines[!startsWith(lines, comment)]
  }
  paste(lines, collapse = "\n")
}
