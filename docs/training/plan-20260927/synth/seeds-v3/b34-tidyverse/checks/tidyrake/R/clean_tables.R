# skipped: needs dplyr (clean_orders, clean_customers, clean_products, clean_returns,
# clean_table)

stopifnot(identical(table_spec("orders")$key, "order_id"))
stopifnot(identical(table_spec("customers")$date_col, "signup_date"))
stopifnot(is.null(table_spec("products")$date_col), "price" %in% table_spec("products")$required)
stopifnot(identical(table_spec("returns")$required[5], "refund"))
stopifnot(inherits(try(table_spec("stores"), silent = TRUE), "try-error"))

b <- list(orders = data.frame(x = 1:4), returns = data.frame(x = integer()))
a <- list(returns = data.frame(x = integer()), orders = data.frame(x = 1:3))
s <- summarise_cleaning(b, a, total = FALSE)
stopifnot(identical(s$table, c("orders", "returns")), identical(s$rows_after, c(3L, 0L)))
stopifnot(identical(s$dropped, c(1L, 0L)), identical(s$pct_dropped, c(25, 0)))
stopifnot(inherits(try(summarise_cleaning(b, a["orders"]), silent = TRUE), "try-error"))
st <- summarise_cleaning(b, list(orders = data.frame(x = 1:2), returns = b$returns))
stopifnot(identical(st$table, c("orders", "returns", "total")), identical(st$dropped[3], 2L))
stopifnot(identical(st$pct_dropped[3], 50), identical(st$rows_before[3], 4L))
stopifnot(identical(summarise_cleaning(b["returns"], a["returns"])$pct_dropped, c(0, 0)))
