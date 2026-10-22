#' Collect a lazy table with a row limit
#'
#' Collects at most `maxRows` rows and warns when the result was cut off,
#' so that an accidental full-table pull is noticed.
#'
#' @param lazy A `tbl_lazy` object.
#' @param maxRows Maximum number of rows to fetch.
#' @return A tibble.
#' @importFrom dplyr collect
#' @importFrom utils head
#' @importFrom magrittr %>%
#' @export
collectCapped <- function(lazy, maxRows = 1e5) {
  # fetch one extra row so we can tell a full result from a cut-off one
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
#' @importFrom dplyr collect arrange
#' @importFrom rlang sym
#' @export
collectChunked <- function(lazy, orderCol, chunkSize = 10000L, callback) {
  callback <- match.fun(callback)
  # paste() would render 1e5 as "1e+05", which no database accepts
  chunkSize <- as.integer(chunkSize)
  if (is.na(chunkSize) || chunkSize < 1L) {
    stop("`chunkSize` must be a positive whole number.", call. = FALSE)
  }

  # a stable order is what makes LIMIT/OFFSET paging correct
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

#' Count rows or distinct values of a lazy table on the server
#'
#' `countRemote()` runs `COUNT(*)` and `countDistinctRemote()` runs
#' `COUNT(DISTINCT col)` remotely; both return a plain number instead of a
#' one-row tibble.
#'
#' @param lazy A `tbl_lazy` object.
#' @return A single number.
#' @importFrom dplyr summarise collect n
#' @export
countRemote <- function(lazy) {
  res <- lazy %>%
    summarise(n = n()) %>%
    collect()
  as.numeric(res$n[1])
}

#' @rdname countRemote
#' @param column Name of the column, as a string.
#' @importFrom dplyr n_distinct
#' @importFrom rlang sym
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
#' @export
dailyRevenue <- function(con, spec = list(), schema = "sales") {
  # everything up to collect() runs on the server
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
