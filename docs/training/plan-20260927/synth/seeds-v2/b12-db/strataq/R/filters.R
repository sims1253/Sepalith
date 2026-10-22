#' Filter a lazy table to a date window
#'
#' Keeps rows whose `day` column falls in `[from, to)`. Either bound may be
#' `NULL` to leave that side open.
#'
#' @param lazy A `tbl_lazy` object.
#' @param from Inclusive lower bound, or `NULL`.
#' @param to Exclusive upper bound, or `NULL`.
#' @return A `tbl_lazy` object.
#' @importFrom dplyr filter
#' @importFrom magrittr %>%
#' @export
filterWindow <- function(lazy, from = NULL, to = NULL) {
  # convert in R so the database receives a typed date literal
  if (!is.null(from)) {
    fromDate <- as.Date(from)
    lazy <- lazy %>% filter(day >= !!fromDate)
  }
  if (!is.null(to)) {
    toDate <- as.Date(to)
    lazy <- lazy %>% filter(day < !!toDate)
  }

  lazy
}

#' Filter a lazy table to some regions or segments
#'
#' `filterRegions()` keeps rows whose `region` column is one of `regions`;
#' `filterSegments()` keeps rows whose `segment` column is one of
#' `segments`. An empty or `NULL` vector keeps every row.
#'
#' @param lazy A `tbl_lazy` object.
#' @param regions A character vector of region codes, or `NULL`.
#' @return A `tbl_lazy` object.
#' @importFrom dplyr filter
#' @export
filterRegions <- function(lazy, regions = NULL) {
  if (length(regions) == 0) {
    return(lazy)
  }
  # region codes are stored upper-case
  regions <- unique(toupper(regions))
  lazy %>% filter(region %in% !!regions)
}

#' @rdname filterRegions
#' @param segments A character vector of segment names, or `NULL`.
#' @export
filterSegments <- function(lazy, segments = NULL) {
  if (length(segments) == 0) {
    return(lazy)
  }
  # segment names, unlike region codes, are stored lower-case
  segments <- unique(tolower(segments))
  lazy %>% filter(segment %in% !!segments)
}

#' Filter a lazy table to a value range
#'
#' Keeps rows where `column` lies between `lower` and `upper`, inclusive.
#' Either bound may be `NULL`.
#'
#' @param lazy A `tbl_lazy` object.
#' @param column Name of the numeric column, as a string.
#' @param lower Inclusive lower bound, or `NULL`.
#' @param upper Inclusive upper bound, or `NULL`.
#' @return A `tbl_lazy` object.
#' @importFrom dplyr filter
#' @importFrom rlang sym
#' @export
filterBetween <- function(lazy, column, lower = NULL, upper = NULL) {
  col <- sym(column)
  if (!is.null(lower) && !is.null(upper) && lower > upper) {
    stop("`lower` must not exceed `upper`.", call. = FALSE)
  }

  if (!is.null(lower)) {
    lazy <- lazy %>% filter(!!col >= !!lower)
  }
  if (!is.null(upper)) {
    lazy <- lazy %>% filter(!!col <= !!upper)
  }

  lazy
}

#' Apply a list of filters to a lazy table
#'
#' Applies the window, region and segment filters named in `spec`, in a
#' fixed order. Unknown names in `spec` raise an error.
#'
#' @param lazy A `tbl_lazy` object.
#' @param spec A named list with any of `from`, `to`, `regions`, `segments`.
#' @return A `tbl_lazy` object.
#' @export
applyFilterSpec <- function(lazy, spec = list()) {
  known <- c("from", "to", "regions", "segments")
  unknown <- setdiff(names(spec), known)
  if (length(unknown) > 0) {
    stop("Unknown filter(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }

  lazy %>%
    filterWindow(from = spec$from, to = spec$to) %>%
    filterRegions(spec$regions) %>%
    filterSegments(spec$segments)
}

#' Keep the latest row per group
#'
#' Uses a window function to keep, for every value of `by`, the row with the
#' greatest `orderCol`. Runs entirely on the server.
#'
#' @param lazy A `tbl_lazy` object.
#' @param by Name of the grouping column, as a string.
#' @param orderCol Name of the ordering column, as a string.
#' @return A `tbl_lazy` object.
#' @importFrom dplyr group_by filter ungroup row_number desc
#' @importFrom dbplyr window_order
#' @importFrom rlang sym
#' @export
latestPerGroup <- function(lazy, by, orderCol) {
  byCol <- sym(by)
  ordCol <- sym(orderCol)

  # dbplyr turns this into ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)
  lazy %>%
    group_by(!!byCol) %>%
    window_order(desc(!!ordCol)) %>%
    filter(row_number() == 1L) %>%
    ungroup()
}
