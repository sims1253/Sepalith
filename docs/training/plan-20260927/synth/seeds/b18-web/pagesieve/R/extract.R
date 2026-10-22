#' Extract text from matching elements
#'
#' Selects elements with a CSS selector and returns their visible text,
#' with whitespace collapsed the way a browser would render it.
#'
#' @param html An `xml_document` or `xml_node` to search.
#' @param css A CSS selector.
#' @param missing Value returned when no element matches.
#' @return A character vector with one entry per matching element.
#' @family extractors
#' @export
extract_text <- function(html, css, missing = NA_character_) {
  nodes <- rvest::html_elements(html, css)
  if (length(nodes) == 0) {
    return(missing)
  }
  text <- rvest::html_text2(nodes)
  text[!nzchar(text)] <- missing
  text
}

#' Extract an attribute from matching elements
#'
#' Selects elements with a CSS selector and returns the value of one of
#' their attributes.
#'
#' @inheritParams extract_text
#' @param attr Name of the attribute to read.
#' @return A character vector with one entry per matching element.
#' @family extractors
#' @export
extract_attr <- function(html, css, attr, missing = NA_character_) {
  nodes <- rvest::html_elements(html, css)
  if (length(nodes) == 0) {
    return(missing)
  }
  values <- rvest::html_attr(nodes, attr)
  values[is.na(values) | !nzchar(values)] <- missing
  values
}

#' Extract numbers from matching elements
#'
#' Selects elements with a CSS selector and parses their text as numbers,
#' dropping currency symbols, thousands separators and units.
#'
#' @inheritParams extract_text
#' @param decimal_mark Character used as the decimal separator.
#' @return A numeric vector with one entry per matching element.
#' @family extractors
#' @export
extract_number <- function(html, css, decimal_mark = ".", missing = NA_real_) {
  nodes <- rvest::html_elements(html, css)
  if (length(nodes) == 0) {
    return(missing)
  }
  text <- rvest::html_text2(nodes)
  if (decimal_mark != ".") {
    text <- gsub(".", "", text, fixed = TRUE)
    text <- sub(decimal_mark, ".", text, fixed = TRUE)
  }
  text <- gsub("[^0-9.eE+-]", "", text)
  values <- suppressWarnings(as.numeric(text))
  values[is.na(values)] <- missing
  values
}

#' Extract absolute links from a page
#'
#' Collects the `href` of every anchor matching a selector, resolves them
#' against a base URL and drops fragments, `mailto:` and `javascript:` links.
#'
#' @param html An `xml_document` or `xml_node` to search.
#' @param base_url URL used to resolve relative links.
#' @param css A CSS selector.
#' @param same_host If `TRUE`, keep only links on the host of `base_url`.
#' @return A character vector of unique absolute URLs.
#' @export
extract_links <- function(html, base_url, css = "a[href]", same_host = FALSE) {
  hrefs <- rvest::html_attr(rvest::html_elements(html, css), "href")
  hrefs <- hrefs[!is.na(hrefs)]
  hrefs <- hrefs[!grepl("^(mailto|javascript|tel):", hrefs, ignore.case = TRUE)]
  hrefs <- sub("#.*$", "", hrefs)
  hrefs <- hrefs[nzchar(hrefs)]
  links <- unique(xml2::url_absolute(hrefs, base_url))
  if (same_host) {
    host <- xml2::url_parse(base_url)$server
    link_hosts <- xml2::url_parse(links)$server
    links <- links[link_hosts == host]
  }
  links
}

#' Extract page metadata
#'
#' Reads the document title and the `<meta>` tags that carry a `name` or
#' `property` attribute into a named list.
#'
#' @param html An `xml_document` holding the parsed page.
#' @return A named list with `title` and one element per meta tag.
#' @export
extract_meta <- function(html) {
  meta <- list(title = extract_text(html, "title")[1])
  nodes <- rvest::html_elements(html, "meta")
  for (node in nodes) {
    key <- rvest::html_attr(node, "name")
    if (is.na(key)) {
      key <- rvest::html_attr(node, "property")
    }
    content <- rvest::html_attr(node, "content")
    if (is.na(key) || is.na(content)) {
      next
    }
    key <- tolower(key)
    if (is.null(meta[[key]])) {
      meta[[key]] <- content
    } else {
      meta[[key]] <- c(meta[[key]], content)
    }
  }
  meta
}

#' Parse a field specification
#'
#' Splits a spec like `"a.title@href"` into its selector and attribute.
#' A spec without `@` refers to the element text.
#'
#' @param spec A single field specification string.
#' @return A list with elements `css` and `attr` (`NULL` for text).
#' @keywords internal
parse_field_spec <- function(spec) {
  if (!grepl("@", spec, fixed = TRUE)) {
    return(list(css = trimws(spec), attr = NULL))
  }
  parts <- strsplit(spec, "@", fixed = TRUE)[[1]]
  if (length(parts) != 2 || !nzchar(parts[2])) {
    stop("Invalid field spec: ", spec, call. = FALSE)
  }
  css <- trimws(parts[1])
  list(css = if (nzchar(css)) css else NULL, attr = trimws(parts[2]))
}

#' Extract a table of records from repeated elements
#'
#' Treats every element matched by `item_css` as one record and fills one
#' column per field specification, using the first match inside the item.
#'
#' @param html An `xml_document` or `xml_node` to search.
#' @param item_css A CSS selector matching one element per record.
#' @param fields Named character vector of field specifications, see
#'   [parse_field_spec()].
#' @return A data frame with one row per item and one column per field.
#' @export
extract_records <- function(html, item_css, fields) {
  if (is.null(names(fields)) || any(!nzchar(names(fields)))) {
    stop("`fields` must be a fully named character vector.", call. = FALSE)
  }
  items <- rvest::html_elements(html, item_css)
  specs <- lapply(fields, parse_field_spec)
  out <- matrix(
    NA_character_,
    nrow = length(items),
    ncol = length(specs),
    dimnames = list(NULL, names(fields))
  )
  for (i in seq_along(items)) {
    for (j in seq_along(specs)) {
      spec <- specs[[j]]
      node <- if (is.null(spec$css)) {
        items[[i]]
      } else {
        rvest::html_element(items[[i]], spec$css)
      }
      if (inherits(node, "xml_missing")) {
        next
      }
      value <- if (is.null(spec$attr)) {
        rvest::html_text2(node)
      } else {
        rvest::html_attr(node, spec$attr)
      }
      if (!is.na(value) && nzchar(value)) {
        out[i, j] <- value
      }
    }
  }
  as.data.frame(out, stringsAsFactors = FALSE)
}
