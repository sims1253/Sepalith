#' Wrap text in Markdown emphasis
#'
#' `md.italic()` uses single asterisks and `md.bold()` double asterisks.
#' The text itself is escaped so it cannot close the emphasis early.
#'
#' @param x A character vector of plain text.
#' @return A character vector of Markdown fragments.
#' @rdname md.emphasis
#' @export
md.italic <- function(x) {
  paste0("*", escape.markdown(x), "*")
}

#' @rdname md.emphasis
#' @export
md.bold <- function(x) {
  paste0("**", escape.markdown(x), "**")
}

#' Wrap text in Markdown inline code
#'
#' Uses a backtick fence one longer than the longest run of backticks in
#' the text, so code containing backticks is still delimited correctly.
#'
#' @param x A character vector of plain text.
#' @return A character vector of Markdown fragments.
#' @export
md.code <- function(x) {
  runs <- regmatches(x, gregexpr("`+", x))
  fence <- vapply(runs, function(r) {
    strrep("`", max(c(0L, nchar(r))) + 1L)
  }, character(1))

  # a space keeps a leading or trailing backtick apart from the fence
  pad <- ifelse(startsWith(x, "`") | endsWith(x, "`"), " ", "")
  paste0(fence, pad, x, pad, fence)
}

#' Build a Markdown link
#'
#' @param text A character vector of link text.
#' @param url A character vector of link targets.
#' @return A character vector of Markdown fragments.
#' @export
md.link <- function(text, url) {
  url <- gsub(" ", "%20", url, fixed = TRUE)
  url <- gsub(")", "%29", url, fixed = TRUE)
  paste0("[", escape.markdown(text), "](", url, ")")
}

#' Turn a heading into a URL-safe anchor
#'
#' @param x A character vector of heading text.
#' @return A character vector of lowercase, hyphen-separated slugs.
#' @export
slugify <- function(x) {
  x %>%
    stringi::stri_trans_general("Latin-ASCII") %>%
    tolower() %>%
    gsub("[^a-z0-9]+", "-", .) %>%
    gsub("^-+|-+$", "", .)
}

#' Build a Markdown list
#'
#' @param items A character vector of list items.
#' @param ordered Logical; produce a numbered list?
#' @param indent Number of spaces to indent each item.
#' @return A single string with one item per line.
#' @export
md.list <- function(items, ordered = FALSE, indent = 0L) {
  if (length(items) == 0L) {
    return("")
  }

  markers <- if (ordered) paste0(seq_along(items), ".") else rep("-", length(items))
  lines <- paste0(strrep(" ", indent), markers, " ", items)
  paste(lines, collapse = "\n")
}

#' Convert a small Markdown subset to HTML
#'
#' Supports ATX headings, unordered lists, paragraphs, bold, italic,
#' inline code and links. Anything else is passed through as escaped text.
#'
#' @param md A single string of Markdown.
#' @return A single string of HTML.
#' @export
md.to.html <- function(md) {
  # escape first, then turn the markup back into tags
  inline <- function(s) {
    s <- escape.html(s)
    s <- gsub("`([^`]+)`", "<code>\\1</code>", s)
    s <- gsub("\\*\\*([^*]+)\\*\\*", "<strong>\\1</strong>", s)
    s <- gsub("\\*([^*]+)\\*", "<em>\\1</em>", s)
    gsub("\\[([^]]+)\\]\\(([^)]+)\\)", "<a href=\"\\2\">\\1</a>", s)
  }

  lines <- strsplit(md, "\n", fixed = TRUE)[[1]]
  html <- character()
  in.list <- FALSE
  para <- character()
  flush.para <- function() {
    if (length(para) > 0L) {
      html <<- c(html, paste0("<p>", inline(paste(para, collapse = " ")), "</p>"))
      para <<- character()
    }
  }

  for (line in lines) {
    is.item <- grepl("^\\s*[-*] ", line)
    if (in.list && !is.item) {
      html <- c(html, "</ul>")
      in.list <- FALSE
    }

    if (grepl("^#{1,6} ", line)) {
      flush.para()
      level <- attr(regexpr("^#+", line), "match.length")
      text <- sub("^#+\\s+", "", line)
      heading <- sprintf("<h%d id=\"%s\">%s</h%d>", level, slugify(text), inline(text), level)
      html <- c(html, heading)
    } else if (is.item) {
      flush.para()
      if (!in.list) {
        html <- c(html, "<ul>")
        in.list <- TRUE
      }
      html <- c(html, paste0("<li>", inline(sub("^\\s*[-*] ", "", line)), "</li>"))
    } else if (!nzchar(trimws(line))) {
      # a blank line ends the current paragraph
      flush.para()
    } else {
      para <- c(para, trimws(line))
    }
  }

  flush.para()
  if (in.list) {
    html <- c(html, "</ul>")
  }
  paste(html, collapse = "\n")
}
