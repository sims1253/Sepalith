#' Scrape all tables from a page
#'
#' Parses every element matched by `css` with [rvest::html_table()] and
#' drops tables that end up with no rows.
#'
#' @param html An `xml_document` holding the parsed page.
#' @param css A CSS selector matching `<table>` elements.
#' @param header Use the first row as column names.
#' @return A list of data frames.
#' @export
scrape_tables <- function(html, css = "table", header = TRUE) {
  nodes <- rvest::html_elements(html, css)
  tables <- vector("list", length(nodes))
  for (i in seq_along(nodes)) {
    tbl <- tryCatch(
      rvest::html_table(nodes[[i]], header = header, fill = TRUE),
      error = function(e) NULL
    )
    if (!is.null(tbl) && nrow(tbl) > 0) {
      tables[[i]] <- as.data.frame(tbl, stringsAsFactors = FALSE)
    }
  }
  Filter(Negate(is.null), tables)
}

#' Pick the table that contains the expected columns
#'
#' Returns the first table whose cleaned column names include every name in
#' `must_have`.
#'
#' @param tables A list of data frames, as returned by [scrape_tables()].
#' @param must_have Character vector of required column names.
#' @return A data frame, or `NULL` when no table qualifies.
#' @export
pick_table <- function(tables, must_have) {
  must_have <- clean_names(must_have)
  for (tbl in tables) {
    cols <- clean_names(names(tbl))
    if (all(must_have %in% cols)) {
      names(tbl) <- cols
      return(tbl)
    }
  }
  NULL
}

#' Clean column names
#'
#' Lower-cases names, replaces runs of non-alphanumeric characters with an
#' underscore and makes the result unique.
#'
#' @param x Character vector of names.
#' @return A character vector of the same length.
#' @keywords internal
clean_names <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x[!nzchar(x)] <- "col"
  make.unique(x, sep = "_")
}

#' Parse a scraped numeric column
#'
#' Strips thousands separators, currency symbols and footnote markers
#' before converting to numbers.
#'
#' @param x Character vector of cell values.
#' @param na Strings to treat as missing.
#' @return A numeric vector.
#' @family column parsers
#' @keywords internal
parse_number_col <- function(x, na = c("", "-", "n/a", "NA")) {
  x <- trimws(as.character(x))
  x[tolower(x) %in% tolower(na)] <- NA
  x <- gsub("\\[[^]]*\\]", "", x)
  x <- gsub("[^0-9.eE-]", "", x)
  suppressWarnings(as.numeric(x))
}

#' Parse a scraped percentage column
#'
#' Removes the percent sign and footnote markers and divides by 100.
#'
#' @param x Character vector of cell values.
#' @param na Strings to treat as missing.
#' @return A numeric vector of proportions.
#' @family column parsers
#' @keywords internal
parse_percent_col <- function(x, na = c("", "-", "n/a", "NA")) {
  x <- trimws(as.character(x))
  x[tolower(x) %in% tolower(na)] <- NA
  x <- gsub("\\[[^]]*\\]", "", x)
  x <- gsub("[^0-9.-]", "", x)
  suppressWarnings(as.numeric(x)) / 100
}

#' Parse a scraped date column
#'
#' Tries each format in turn and keeps the first successful parse for
#' every cell.
#'
#' @param x Character vector of cell values.
#' @param na Strings to treat as missing.
#' @param formats Character vector of [strptime()] formats to try.
#' @return A `Date` vector.
#' @family column parsers
#' @keywords internal
parse_date_col <- function(x, na = c("", "-", "n/a", "NA"),
                           formats = c("%Y-%m-%d", "%d.%m.%Y", "%B %d, %Y")) {
  x <- trimws(as.character(x))
  x[tolower(x) %in% tolower(na)] <- NA
  x <- gsub("\\[[^]]*\\]", "", x)
  out <- as.Date(rep(NA_character_, length(x)))
  for (fmt in formats) {
    todo <- is.na(out) & !is.na(x)
    if (!any(todo)) {
      break
    }
    out[todo] <- as.Date(x[todo], format = fmt)
  }
  out
}

#' Coerce table columns to declared types
#'
#' Applies one of the column parsers to every column listed in `types`;
#' unlisted columns are left as they are.
#'
#' @param df A data frame of scraped cells.
#' @param types Named character vector mapping column names to one of
#'   `"number"`, `"percent"`, `"date"`, `"text"` or `"logical"`.
#' @return The data frame with converted columns.
#' @export
coerce_columns <- function(df, types) {
  unknown <- setdiff(names(types), names(df))
  if (length(unknown) > 0) {
    stop("Unknown columns: ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  for (col in names(types)) {
    df[[col]] <- switch(types[[col]],
      number = parse_number_col(df[[col]]),
      percent = parse_percent_col(df[[col]]),
      date = parse_date_col(df[[col]]),
      text = trimws(as.character(df[[col]])),
      logical = tolower(trimws(df[[col]])) %in% c("yes", "true", "x", "1"),
      stop("Unsupported type for column ", col, ": ", types[[col]], call. = FALSE)
    )
  }
  df
}

#' Combine tables scraped from several pages
#'
#' Aligns columns by name, fills missing columns with `NA` and adds a
#' source column identifying where every row came from.
#'
#' @param tables A list of data frames, as returned by [scrape_tables()].
#' @param id Name of the column that records the source of each row.
#' @return A single data frame.
#' @export
combine_tables <- function(tables, id = "source") {
  if (length(tables) == 0) {
    return(data.frame())
  }
  sources <- names(tables) %||% as.character(seq_along(tables))
  all_cols <- unique(unlist(lapply(tables, names), use.names = FALSE))
  pieces <- vector("list", length(tables))
  for (i in seq_along(tables)) {
    tbl <- tables[[i]]
    for (col in setdiff(all_cols, names(tbl))) {
      tbl[[col]] <- rep(NA, nrow(tbl))
    }
    tbl <- tbl[all_cols]
    tbl[[id]] <- rep(sources[[i]], nrow(tbl))
    pieces[[i]] <- tbl
  }
  out <- do.call(rbind, pieces)
  rownames(out) <- NULL
  out[c(id, all_cols)]
}
