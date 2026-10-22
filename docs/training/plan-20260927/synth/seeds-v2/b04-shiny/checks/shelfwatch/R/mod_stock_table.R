# skipped: mod_stock_filter_ui, mod_stock_filter_server, mod_stock_table_ui,
#   mod_stock_table_server (need shiny / DT); stock_summary_boxes (needs bslib)

# stock_status
stopifnot(identical(stock_status(c(100, 50, 10, 0, -2), 50), c("ok", "low", "low", "out", "out")))
stopifnot(identical(stock_status(c(5, 5), c(1, 10)), c("ok", "low")))
stopifnot(identical(stock_status(numeric(), 1), character()))

# format_stock_table puts out-of-stock items first
stock <- data.frame(
  sku = c("B-1", "A-1", "C-1"), name = c("b", "a", "c"), warehouse = "n",
  on_hand = c(5, 100, 0), unit_cost = c(1.5, 2, 0.25), status = c("low", "ok", "out")
)
tab <- format_stock_table(stock)
stopifnot(identical(tab$SKU, c("C-1", "B-1", "A-1")), identical(tab$`Unit cost`[1], "0.25"))
stopifnot(identical(names(tab)[4], "On hand"), nrow(format_stock_table(stock[0, ])) == 0L)
