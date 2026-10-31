#' Turn a title into a section identifier
#'
#' @export
#' @param title A single string used as the section title.
#' @return A lower-case slug made of letters, digits and hyphens.
section.id <- function(title) {
  id <- tolower(title)
  id <- gsub("[^a-z0-9]+", "-", id)
  id <- gsub("^-+|-+$", "", id)
  if (!nzchar(id)) "section" else id
}

#' Create a report section
#'
#' @description
#' A section holds a title, a heading level and an ordered list of content
#' blocks added with [add.paragraph()], [add.bullets()] and [add.code()].
#'
#' @export
#' @param title A single string used as the section title.
#' @param level Heading level, from 1 to 4.
#' @param id Identifier used for anchors and labels; derived from `title`
#'   when `NULL`.
#' @param blocks A list of content blocks.
#' @return An object of class `rl_section` with elements `title`, `level`,
#'   `id` and `blocks`.
new.section <- function(title, level = 1L, id = NULL, blocks = list()) {
  # Check `title`
  if (!is.character(title) || length(title) != 1L || is.na(title)) {
    stop("`title` must be a single string.", call. = FALSE)
  }
  # Check `level`
  if (!is.numeric(level) || length(level) != 1L || !level %in% 1:4) {
    stop("`level` must be 1, 2, 3 or 4.", call. = FALSE)
  }
  # Check `id`
  if (is.null(id)) {
    id <- section.id(title)
  }

  if (!is.character(id) || length(id) != 1L || !grepl("^[a-z0-9-]+$", id)) {
    stop("`id` must be a lower-case slug.", call. = FALSE)
  }
  # Check `blocks`
  if (!is.list(blocks)) {
    stop("`blocks` must be a list.", call. = FALSE)
  }

  structure(
    list(title = title, level = as.integer(level), id = id, blocks = blocks),
    class = "rl_section"
  )
}

#' Heading line for a section
#'
#' @export
#' @param title A single string used as the section title.
#' @param level Heading level, from 1 to 4.
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"text"`.
#' @param id Identifier used for anchors and labels, or `NULL` for none.
#' @return A character vector of one line, or two lines for `"text"` where
#'   the title is underlined.
section.heading <- function(title, level = 1L, format = c("markdown", "html", "latex", "text"),
                            id = NULL) {
  format <- match.arg(format)
  switch(format,
    markdown = {
      paste(strrep("#", level), title)
    },
    html = {
      anchor <- if (is.null(id)) "" else sprintf(" id=\"%s\"", id)
      sprintf("<h%d%s>%s</h%d>", level, anchor, title, level)
    },
    latex = {
      command <- c("section", "subsection", "subsubsection", "paragraph")[level]
      label <- if (is.null(id)) "" else sprintf("\\label{%s}", id)
      sprintf("\\%s{%s}%s", command, title, label)
    },
    text = {
      underline <- c("=", "-", "~", ".")[level]
      c(title, strrep(underline, nchar(title)))
    }
  )
}

#' Add content blocks to a section
#'
#' @export
#' @param section An `rl_section` object created by [new.section()].
#' @param text A character vector joined into one paragraph.
#' @return The modified `section`.
#' @rdname add.block
add.paragraph <- function(section, text) {
  stopifnot(inherits(section, "rl_section"))
  block <- list(type = "paragraph", content = paste(text, collapse = " "))
  section$blocks <- c(section$blocks, list(block))
  section
}

#' @export
#' @param items A character vector with one entry per list item.
#' @param ordered Whether to number the items.
#' @rdname add.block
add.bullets <- function(section, items, ordered = FALSE) {
  stopifnot(inherits(section, "rl_section"))
  type <- if (ordered) "ordered" else "bullets"
  block <- list(type = type, content = as.character(items))
  section$blocks <- c(section$blocks, list(block))
  section
}

#' @export
#' @param code A character vector of code lines.
#' @param language Language tag used for Markdown code fences.
#' @rdname add.block
add.code <- function(section, code, language = "r") {
  stopifnot(inherits(section, "rl_section"))
  block <- list(type = "code", content = as.character(code), language = language)
  section$blocks <- c(section$blocks, list(block))
  section
}

#' Build a section from a plain list
#'
#' @description
#' Convenient when sections are read from YAML or JSON. Blocks are added in
#' the order `paragraphs`, `bullets`, `steps`, `code`.
#'
#' @export
#' @param spec A list with element `title` and optional elements `level`,
#'   `paragraphs`, `bullets`, `steps` and `code`.
#' @return An `rl_section` object.
section.from.list <- function(spec) {
  if (!is.list(spec) || is.null(spec$title)) {
    stop("`spec` must be a list with a `title` element.", call. = FALSE)
  }

  known <- c("title", "level", "paragraphs", "bullets", "steps", "code")
  unknown <- setdiff(names(spec), known)
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown element(s) in `spec`: %s.", paste(unknown, collapse = ", ")),
         call. = FALSE)
  }

  level <- if (is.null(spec$level)) 1L else spec$level
  section <- new.section(spec$title, level = level)

  # Add `paragraphs`
  for (text in spec$paragraphs) {
    section <- add.paragraph(section, text)
  }
  # Add `bullets`
  if (!is.null(spec$bullets)) {
    section <- add.bullets(section, spec$bullets)
  }
  # Add `steps`
  if (!is.null(spec$steps)) {
    section <- add.bullets(section, spec$steps, ordered = TRUE)
  }
  # Add `code`
  if (!is.null(spec$code)) {
    section <- add.code(section, spec$code)
  }

  section
}

#' Render a section
#'
#' @description
#' Writes the heading followed by every block, each preceded by a blank
#' line. Block types are `paragraph`, `bullets`, `ordered` and `code`.
#'
#' @export
#' @param section An `rl_section` object created by [new.section()].
#' @param format One of `"markdown"`, `"html"`, `"latex"` or `"text"`.
#' @return A character vector with one element per output line.
render.section <- function(section, format = c("markdown", "html", "latex", "text")) {
  format <- match.arg(format)
  lines <- section.heading(section$title, section$level, format, section$id)

  for (block in section$blocks) {
    content <- block$content
    body <- switch(block$type,
      paragraph = switch(format,
        html = paste0("<p>", content, "</p>"),
        content
      ),
      bullets = switch(format,
        markdown = paste("-", content),
        html = c("<ul>", paste0("  <li>", content, "</li>"), "</ul>"),
        latex = c("\\begin{itemize}", paste("  \\item", content), "\\end{itemize}"),
        text = paste("*", content)
      ),
      ordered = switch(format,
        markdown = paste0(seq_along(content), ". ", content),
        html = c("<ol>", paste0("  <li>", content, "</li>"), "</ol>"),
        latex = c("\\begin{enumerate}", paste("  \\item", content), "\\end{enumerate}"),
        text = paste0(seq_along(content), ". ", content)
      ),
      code = switch(format,
        markdown = c(paste0("```", block$language), content, "```"),
        html = c("<pre><code>", content, "</code></pre>"),
        latex = c("\\begin{verbatim}", content, "\\end{verbatim}"),
        text = paste0("    ", content)
      ),
      stop(sprintf("Unknown block type `%s`.", block$type), call. = FALSE)
    )
    lines <- c(lines, "", body)
  }

  lines
}
