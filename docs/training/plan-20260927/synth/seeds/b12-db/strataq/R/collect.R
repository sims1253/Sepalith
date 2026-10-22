#' Collect a lazy table with a row limit
#'
#' Collects at most `maxRows` rows and warns when the result was cut off,
#' so that an accidental full-table pull is noticed.
#'
#' @param lazy A `tbl_lazy` object.
#' @param maxRows Maximum number of rows to fetch.
#' @return A tibble.
#' @family collectors
#' @importFrom dplyr collect
#' @importFrom utils head
#' @importFrom magrittr %>%
#' @export
collectCapped <- function(lazy, maxRows = 1e5) {
  out <- lazy %>%
    head(maxRows + 1) %>%
    collect()
  if (nrow(out) > maxRows) {
    warning("Result truncated to ", format(maxRows, big.mark = ","), " rows.", call. = FALSE)
    out <- out[seq_len(maxRows), ]
  }
  out
}

#' Collect a lazy table and time the query
#'
#' Collects `lazy` and attaches the elapsed wall time, in seconds, as the
#' `"elapsed"` attribute of the result.
#'
#' @param lazy A `tbl_lazy` object.
#' @return A tibble with an `"elapsed"` attribute.
#' @family collectors
#' @importFrom dplyr collect
#' @export
collectTimed <- function(lazy) {
  started <- proc.time()[["elapsed"]]
  out <- collect(lazy)
  attr(out, "elapsed") <- proc.time()[["elapsed"]] - started
  out
}

#' Collect a lazy table in chunks
#'
#' Pages through `lazy` ordered by `orderCol`, fetching `chunkSize` rows at
#' a time, and calls `callback` on each chunk.
#'
#' @param lazy A `tbl_lazy` object.
#' @param orderCol Name of the ordering column, as a string.
#' @param chunkSize Number of rows per chunk.
#' @param callback A function called with each chunk and its index.
#' @return The total number of rows processed, invisibly.
#' @family collectors
#' @importFrom dplyr collect arrange
#' @importFrom rlang sym
#' @importFrom magrittr %>%
#' @export
collectChunked <- function(lazy, orderCol, chunkSize = 10000L, callback) {
  callback <- match.fun(callback)
  ordered <- lazy %>% arrange(!!sym(orderCol))
  total <- 0L
  chunk <- 0L
  repeat {
    sql <- paste(
      dbplyr::sql_render(ordered),
      "LIMIT", chunkSize, "OFFSET", total
    )
    page <- DBI::dbGetQuery(dbplyr::remote_con(lazy), sql)
    if (nrow(page) == 0) {
      break
    }
    chunk <- chunk + 1L
    callback(page, chunk)
    total <- total + nrow(page)
    if (nrow(page) < chunkSize) {
      break
    }
  }
  invisible(total)
}

#' Count rows of a lazy table on the server
#'
#' Runs `COUNT(*)` remotely and returns a plain integer instead of a
#' one-row tibble.
#'
#' @param lazy A `tbl_lazy` object.
#' @return A single number.
#' @family summaries
#' @importFrom dplyr summarise collect n
#' @importFrom magrittr %>%
#' @export
countRemote <- function(lazy) {
  res <- lazy %>%
    summarise(n = n()) %>%
    collect()
  as.numeric(res$n[1])
}

#' Count distinct values of a column on the server
#'
#' Runs `COUNT(DISTINCT col)` remotely and returns a plain number.
#'
#' @param lazy A `tbl_lazy` object.
#' @param column Name of the column, as a string.
#' @return A single number.
#' @family summaries
#' @importFrom dplyr summarise collect n_distinct
#' @importFrom rlang sym
#' @importFrom magrittr %>%
#' @export
countDistinctRemote <- function(lazy, column) {
  col <- sym(column)
  res <- lazy %>%
    summarise(n = n_distinct(!!col)) %>%
    collect()
  as.numeric(res$n[1])
}

#' Summarise revenue by day on the server
#'
#' Groups the orders table by `day` and `region`, computes order counts and
#' revenue remotely, and collects the small result.
#'
#' @param con A `DBIConnection` object.
#' @param spec A named list with any of `from`, `to`, `regions`, `segments`.
#' @param schema Name of the schema, or `NULL` for the default schema.
#' @return A tibble with columns `day`, `region`, `orders` and `revenue`.
#' @importFrom dplyr group_by summarise arrange collect n
#' @importFrom magrittr %>%
#' @export
dailyRevenue <- function(con, spec = list(), schema = "sales") {
  ordersWithCustomers(con, schema) %>%
    applyFilterSpec(spec) %>%
    group_by(day, region) %>%
    summarise(
      orders = n(),
      revenue = sum(amount, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(day, region) %>%
    collect()
}
