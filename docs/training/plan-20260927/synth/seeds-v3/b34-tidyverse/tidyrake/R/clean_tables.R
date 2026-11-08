#' Specification of a raw table
#'
#' @param table One of `"orders"`, `"customers"`, `"products"` or
#'   `"returns"`.
#' @return A list with elements `key` (the primary key column), `required`
#'   (columns that must be present) and `date_col` (the column used to
#'   order rows, or `NULL`).
#' @keywords internal
table_spec <- function(table = c("orders", "customers", "products", "returns")) {
  table <- match.arg(table)
  switch(table,
    orders = list(
      key = "order_id",
      required = c("order_id", "customer_id", "order_date", "amount", "status"),
      date_col = "order_date"
    ),
    customers = list(
      key = "customer_id",
      required = c("customer_id", "name", "email", "signup_date", "region"),
      date_col = "signup_date"
    ),
    products = list(
      key = "product_id",
      required = c("product_id", "title", "category", "price", "active"),
      date_col = NULL
    ),
    returns = list(
      key = "return_id",
      required = c("return_id", "order_id", "return_date", "reason", "refund"),
      date_col = "return_date"
    )
  )
}

#' Clean the raw orders table
#'
#' @param orders A data frame of raw orders.
#' @param min_date Optional earliest date to keep.
#' @return A tibble with one row per `order_id`, sorted by `order_date`.
#' @importFrom dplyr rename_with mutate filter distinct arrange
#' @importFrom magrittr %>%
#' @export
clean_orders <- function(orders, min_date = NULL) {
  spec <- table_spec("orders")
  orders <- dplyr::rename_with(orders, standardise_names)
  check_known_cols(orders, spec$required, "orders")

  # Normalise types and text, then keep one row per key
  out <- orders %>%
    dplyr::mutate(
      order_id = trimws(as.character(order_id)),
      customer_id = trimws(as.character(customer_id)),
      order_date = as.Date(order_date),
      amount = as.numeric(amount),
      status = tolower(trimws(status))
    ) %>%
    dplyr::filter(!is.na(order_id), order_id != "") %>%
    dplyr::distinct(order_id, .keep_all = TRUE)
  if (!is.null(min_date)) {
    out <- dplyr::filter(out, order_date >= as.Date(min_date))
  }
  out <- dplyr::arrange(out, order_date, order_id)
  attr(out, "rows_dropped") <- nrow(orders) - nrow(out)
  out
}

#' Clean the raw customers table
#'
#' @param customers A data frame of raw customers.
#' @param min_date Optional earliest date to keep.
#' @return A tibble with one row per `customer_id`, sorted by `signup_date`.
#' @export
clean_customers <- function(customers, min_date = NULL) {
  spec <- table_spec("customers")
  customers <- dplyr::rename_with(customers, standardise_names)
  check_known_cols(customers, spec$required, "customers")

  # Normalise types and text, then keep one row per key
  out <- customers %>%
    dplyr::mutate(
      customer_id = trimws(as.character(customer_id)),
      name = trimws(as.character(name)),
      email = tolower(trimws(email)),
      signup_date = as.Date(signup_date),
      region = tolower(trimws(region))
    ) %>%
    dplyr::filter(!is.na(customer_id), customer_id != "") %>%
    dplyr::distinct(customer_id, .keep_all = TRUE)
  if (!is.null(min_date)) {
    out <- dplyr::filter(out, signup_date >= as.Date(min_date))
  }
  out <- dplyr::arrange(out, signup_date, customer_id)
  attr(out, "rows_dropped") <- nrow(customers) - nrow(out)
  out
}

#' Clean the raw products table
#'
#' @param products A data frame of raw products.
#' @param active_only Keep only products flagged as active?
#' @return A tibble with one row per `product_id`, sorted by `product_id`.
#' @export
clean_products <- function(products, active_only = FALSE) {
  spec <- table_spec("products")
  products <- dplyr::rename_with(products, standardise_names)
  check_known_cols(products, spec$required, "products")

  # Normalise types and text, then keep one row per key
  out <- products %>%
    dplyr::mutate(
      product_id = trimws(as.character(product_id)),
      title = trimws(as.character(title)),
      category = tolower(trimws(category)),
      price = as.numeric(price),
      active = as.logical(active)
    ) %>%
    dplyr::filter(!is.na(product_id), product_id != "") %>%
    dplyr::distinct(product_id, .keep_all = TRUE)
  if (active_only) {
    out <- dplyr::filter(out, active %in% TRUE)
  }
  out <- dplyr::arrange(out, product_id)
  attr(out, "rows_dropped") <- nrow(products) - nrow(out)
  out
}

#' Clean the raw returns table
#'
#' @param returns A data frame of raw returns.
#' @param min_date Optional earliest date to keep.
#' @return A tibble with one row per `return_id`, sorted by `return_date`.
#' @export
clean_returns <- function(returns, min_date = NULL) {
  spec <- table_spec("returns")
  returns <- dplyr::rename_with(returns, standardise_names)
  check_known_cols(returns, spec$required, "returns")

  # Normalise types and text, then keep one row per key
  out <- returns %>%
    dplyr::mutate(
      return_id = trimws(as.character(return_id)),
      order_id = trimws(as.character(order_id)),
      return_date = as.Date(return_date),
      reason = tolower(trimws(reason)),
      refund = as.numeric(refund)
    ) %>%
    dplyr::filter(!is.na(return_id), return_id != "") %>%
    dplyr::distinct(return_id, .keep_all = TRUE)
  if (!is.null(min_date)) {
    out <- dplyr::filter(out, return_date >= as.Date(min_date))
  }
  out <- dplyr::arrange(out, return_date, return_id)
  attr(out, "rows_dropped") <- nrow(returns) - nrow(out)
  out
}

#' Clean one raw table by name
#'
#' @param data A data frame holding the raw table.
#' @param table Which cleaner to apply.
#' @param ... Passed to the table-specific cleaner.
#' @return The cleaned table.
#' @export
clean_table <- function(data,
                        table = c("orders", "customers", "products", "returns"),
                        ...) {
  table <- match.arg(table)
  switch(table,
    orders = clean_orders(data, ...),
    customers = clean_customers(data, ...),
    products = clean_products(data, ...),
    returns = clean_returns(data, ...)
  )
}

#' Summarise the rows removed by cleaning
#'
#' @param before A named list of raw tables.
#' @param after A named list of cleaned tables with the same names.
#' @param total Append a row named `"total"` with the column sums?
#' @return A data frame with columns `table`, `rows_before`, `rows_after`,
#'   `dropped` and `pct_dropped`, one row per table.
#' @export
summarise_cleaning <- function(before, after, total = TRUE) {
  if (!identical(sort(names(before)), sort(names(after)))) {
    stop("`before` and `after` must name the same tables.", call. = FALSE)
  }
  tables <- names(before)

  rows_before <- vapply(tables, function(nm) nrow(before[[nm]]), integer(1))
  rows_after <- vapply(tables, function(nm) nrow(after[[nm]]), integer(1))
  dropped <- rows_before - rows_after
  pct_dropped <- ifelse(rows_before > 0L, round(100 * dropped / rows_before, 1), 0)

  out <- data.frame(
    table = tables,
    rows_before = unname(rows_before),
    rows_after = unname(rows_after),
    dropped = unname(dropped),
    pct_dropped = unname(pct_dropped),
    stringsAsFactors = FALSE
  )

  # Append a total row computed from the column sums
  if (total) {
    out <- rbind(out, data.frame(
      table = "total",
      rows_before = sum(out$rows_before),
      rows_after = sum(out$rows_after),
      dropped = sum(out$dropped),
      pct_dropped = if (sum(out$rows_before) > 0L) {
        round(100 * sum(out$dropped) / sum(out$rows_before), 1)
      } else {
        0
      },
      stringsAsFactors = FALSE
    ))
  }
  out
}
