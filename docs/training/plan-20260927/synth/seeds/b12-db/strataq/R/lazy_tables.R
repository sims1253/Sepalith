#' Open a lazy table reference
#'
#' Returns a [dplyr::tbl()] for `table`, optionally inside `schema`. Nothing
#' is fetched until the result is collected.
#'
#' @param con A `DBIConnection` object.
#' @param table Name of the table.
#' @param schema Name of the schema, or `NULL` for the default schema.
#' @return A `tbl_lazy` object.
#' @importFrom dplyr tbl
#' @importFrom dbplyr in_schema
#' @export
lazyTable <- function(con, table, schema = NULL) {
  if (!DBI::dbIsValid(con)) {
    stop("Connection is closed.", call. = FALSE)
  }
  if (is.null(schema)) {
    tbl(con, table)
  } else {
    tbl(con, in_schema(schema, table))
  }
}

#' Open the orders table
#'
#' Returns the `orders` fact table with the order date renamed to `day`
#' and cancelled orders removed.
#'
#' @param con A `DBIConnection` object.
#' @param schema Name of the schema, or `NULL` for the default schema.
#' @return A `tbl_lazy` object.
#' @family source tables
#' @importFrom dplyr filter rename
#' @importFrom magrittr %>%
#' @export
ordersTable <- function(con, schema = "sales") {
  lazyTable(con, "orders", schema) %>%
    filter(status != "cancelled") %>%
    rename(day = order_date)
}

#' Open the customers table
#'
#' Returns the `customers` dimension table restricted to active customers,
#' with the signup date renamed to `day`.
#'
#' @param con A `DBIConnection` object.
#' @param schema Name of the schema, or `NULL` for the default schema.
#' @return A `tbl_lazy` object.
#' @family source tables
#' @importFrom dplyr filter rename
#' @importFrom magrittr %>%
#' @export
customersTable <- function(con, schema = "sales") {
  lazyTable(con, "customers", schema) %>%
    filter(is_active == TRUE) %>%
    rename(day = signup_date)
}

#' Open the shipments table
#'
#' Returns the `shipments` table restricted to dispatched parcels, with the
#' dispatch date renamed to `day`.
#'
#' @param con A `DBIConnection` object.
#' @param schema Name of the schema, or `NULL` for the default schema.
#' @return A `tbl_lazy` object.
#' @family source tables
#' @importFrom dplyr filter rename
#' @importFrom magrittr %>%
#' @export
shipmentsTable <- function(con, schema = "logistics") {
  lazyTable(con, "shipments", schema) %>%
    filter(!is.na(dispatched_at)) %>%
    rename(day = dispatched_at)
}

#' Join orders to customers
#'
#' Left-joins the orders table to the customers table on `customer_id`,
#' keeping the customer region and segment.
#'
#' @param con A `DBIConnection` object.
#' @param schema Name of the schema, or `NULL` for the default schema.
#' @return A `tbl_lazy` object.
#' @importFrom dplyr left_join select
#' @importFrom magrittr %>%
#' @export
ordersWithCustomers <- function(con, schema = "sales") {
  customers <- customersTable(con, schema) %>%
    select(customer_id, region, segment)
  ordersTable(con, schema) %>%
    left_join(customers, by = "customer_id")
}

#' Show the SQL behind a lazy table
#'
#' Renders the query that would be sent for `lazy`, either printing it as
#' [dplyr::show_query()] does or returning it as a string.
#'
#' @param lazy A `tbl_lazy` object.
#' @param print Whether to print the SQL instead of returning it.
#' @return The SQL as a character string, invisibly when printed.
#' @importFrom dbplyr remote_query
#' @export
lazySql <- function(lazy, print = TRUE) {
  if (!inherits(lazy, "tbl_lazy")) {
    stop("`lazy` must be a lazy table, not ", class(lazy)[1], ".", call. = FALSE)
  }
  sql <- as.character(remote_query(lazy))
  if (print) {
    cat(sql, "\n", sep = "")
    return(invisible(sql))
  }
  sql
}

#' Compute a lazy table into a temporary table
#'
#' Materialises `lazy` on the server with [dplyr::compute()] and adds
#' indexes on the given columns, so that later joins are cheap.
#'
#' @param lazy A `tbl_lazy` object.
#' @param name Name of the temporary table.
#' @param indexes A character vector of columns to index, or `NULL`.
#' @return A `tbl_lazy` pointing at the temporary table.
#' @importFrom dplyr compute
#' @export
materialise <- function(lazy, name, indexes = NULL) {
  idx <- if (is.null(indexes)) list() else as.list(indexes)
  compute(lazy, name = name, temporary = TRUE, indexes = idx)
}
