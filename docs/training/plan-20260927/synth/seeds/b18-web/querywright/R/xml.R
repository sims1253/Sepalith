#' Read the text of a child node
#'
#' Finds the first node matching `xpath` below `node` and returns its
#' trimmed text, or `NA` when there is no such node.
#'
#' @param node An `xml_node` to search from.
#' @param xpath An XPath expression relative to `node`.
#' @param ns A namespace map as returned by [xml2::xml_ns()].
#' @return A single string.
#' @keywords internal
child_text <- function(node, xpath, ns = character()) {
  child <- xml2::xml_find_first(node, xpath, ns)
  if (inherits(child, "xml_missing")) {
    return(NA_character_)
  }
  child %>% xml2::xml_text() %>% trimws()
}

#' Read an attribute of a child node
#'
#' Finds the first node matching `xpath` below `node` and returns the
#' value of one attribute, or `NA` when there is no such node.
#'
#' @inheritParams child_text
#' @param attr Name of the attribute to read.
#' @return A single string.
#' @keywords internal
child_attr <- function(node, xpath, attr, ns = character()) {
  child <- xml2::xml_find_first(node, xpath, ns)
  if (inherits(child, "xml_missing")) {
    return(NA_character_)
  }
  child %>% xml2::xml_attr(attr) %>% trimws()
}

#' Read a sitemap
#'
#' Parses a `urlset` sitemap into one row per URL. Namespaces are stripped
#' so that sitemaps with and without the standard namespace both work.
#'
#' @param x A path, URL or literal XML string.
#' @return A data frame with columns `loc`, `lastmod`, `changefreq` and
#'   `priority`.
#' @family xml readers
#' @export
read_sitemap <- function(x) {
  doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
  nodes <- xml2::xml_find_all(doc, "/urlset/url")
  out <- data.frame(
    loc = character(length(nodes)),
    lastmod = as.Date(rep(NA_character_, length(nodes))),
    changefreq = character(length(nodes)),
    priority = numeric(length(nodes)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(nodes)) {
    out$loc[i] <- child_text(nodes[[i]], "./loc")
    out$lastmod[i] <- as.Date(substr(child_text(nodes[[i]], "./lastmod"), 1, 10))
    out$changefreq[i] <- child_text(nodes[[i]], "./changefreq")
    out$priority[i] <- as.numeric(child_text(nodes[[i]], "./priority"))
  }
  out$priority[is.na(out$priority)] <- 0.5
  out
}

#' Read a sitemap index
#'
#' Parses a `sitemapindex` document into one row per child sitemap.
#'
#' @param x A path, URL or literal XML string.
#' @return A data frame with columns `loc` and `lastmod`.
#' @family xml readers
#' @export
read_sitemap_index <- function(x) {
  doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
  nodes <- xml2::xml_find_all(doc, "/sitemapindex/sitemap")
  out <- data.frame(
    loc = character(length(nodes)),
    lastmod = as.Date(rep(NA_character_, length(nodes))),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(nodes)) {
    out$loc[i] <- child_text(nodes[[i]], "./loc")
    out$lastmod[i] <- as.Date(substr(child_text(nodes[[i]], "./lastmod"), 1, 10))
  }
  out
}

#' Parse the items of an RSS feed
#'
#' Reads one row per `<item>`, falling back to the link when an item has
#' no `<guid>`.
#'
#' @param doc An `xml_document` holding an RSS 2.0 feed with namespaces
#'   stripped.
#' @return A data frame with columns `title`, `link`, `published` and `id`.
#' @family feed parsers
#' @keywords internal
parse_rss_items <- function(doc) {
  items <- xml2::xml_find_all(doc, "/rss/channel/item")
  out <- data.frame(
    title = character(length(items)),
    link = character(length(items)),
    published = as.POSIXct(rep(NA_real_, length(items)), tz = "UTC"),
    id = character(length(items)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(items)) {
    out$title[i] <- child_text(items[[i]], "./title")
    out$link[i] <- child_text(items[[i]], "./link")
    out$published[i] <- as.POSIXct(
      child_text(items[[i]], "./pubDate"),
      format = "%a, %d %b %Y %H:%M:%S",
      tz = "UTC"
    )
    out$id[i] <- child_text(items[[i]], "./guid")
  }
  out$id[is.na(out$id)] <- out$link[is.na(out$id)]
  out
}

#' Parse the entries of an Atom feed
#'
#' Reads one row per `<entry>`, falling back to the link when an entry has
#' no `<id>`.
#'
#' @param doc An `xml_document` holding an Atom feed with namespaces
#'   stripped.
#' @return A data frame with columns `title`, `link`, `published` and `id`.
#' @family feed parsers
#' @keywords internal
parse_atom_entries <- function(doc) {
  entries <- xml2::xml_find_all(doc, "/feed/entry")
  out <- data.frame(
    title = character(length(entries)),
    link = character(length(entries)),
    published = as.POSIXct(rep(NA_real_, length(entries)), tz = "UTC"),
    id = character(length(entries)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(entries)) {
    out$title[i] <- child_text(entries[[i]], "./title")
    out$link[i] <- child_attr(entries[[i]], "./link[not(@rel) or @rel='alternate']", "href")
    out$published[i] <- as.POSIXct(
      substr(child_text(entries[[i]], "./updated"), 1, 19),
      format = "%Y-%m-%dT%H:%M:%S",
      tz = "UTC"
    )
    out$id[i] <- child_text(entries[[i]], "./id")
  }
  out$id[is.na(out$id)] <- out$link[is.na(out$id)]
  out
}

#' Read an RSS or Atom feed
#'
#' Detects the feed flavour from the root element and dispatches to the
#' matching parser, so callers always get the same columns back.
#'
#' @param x A path, URL or literal XML string.
#' @param since Optional `POSIXct`; older items are dropped.
#' @return A data frame with columns `title`, `link`, `published` and `id`,
#'   newest first.
#' @export
read_feed <- function(x, since = NULL) {
  doc <- xml2::read_xml(x) %>% xml2::xml_ns_strip()
  root <- doc %>% xml2::xml_root() %>% xml2::xml_name()
  items <- switch(root,
    rss = parse_rss_items(doc),
    feed = parse_atom_entries(doc),
    stop("Unsupported feed root element: <", root, ">", call. = FALSE)
  )
  if (!is.null(since)) {
    items <- items[!is.na(items$published) & items$published >= since, , drop = FALSE]
  }
  items <- items[order(items$published, decreasing = TRUE), , drop = FALSE]
  rownames(items) <- NULL
  items
}
