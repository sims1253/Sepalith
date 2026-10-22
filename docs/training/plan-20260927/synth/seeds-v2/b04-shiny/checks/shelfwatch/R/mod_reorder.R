# skipped: mod_reorder_ui, mod_reorder_server (need shiny / bslib / DT);
#   plot_order_totals (needs ggplot2)

# clean_supplier() lives in utils_validate.R; a minimal stand-in keeps this
# check independent of the working directory
if (!exists("clean_supplier")) clean_supplier <- function(x) trimws(x)

stock <- data.frame(sku = c("BOL-0042", "NUT-0108"), unit_cost = c(0.35, 0.12))

# empty_order_table and build_order_row
empty <- empty_order_table()
stopifnot(nrow(empty) == 0L)
stopifnot(identical(names(empty), c("sku", "supplier", "qty", "unit_cost", "line_total")))
row <- build_order_row(stock, "BOL-0042", 100, " acme ")
stopifnot(identical(row$qty, 100L), identical(row$supplier, "acme"), all.equal(row$line_total, 35))
stopifnot(identical(names(rbind(empty, row)), names(empty)))
stopifnot(inherits(try(build_order_row(stock, "ZZZ-9999", 1, "x"), silent = TRUE), "try-error"))

# reorder_quantity
stopifnot(identical(reorder_quantity(120, 50, 200), 0L))
stopifnot(identical(reorder_quantity(12, 50, 200, case_size = 25L), 200L))
stopifnot(identical(reorder_quantity(50, 50, 40), 0L))

# order_totals_by_supplier
orders <- rbind(row, build_order_row(stock, "NUT-0108", 500, "Boltworks"))
tot <- order_totals_by_supplier(orders)
stopifnot(identical(tot$supplier, c("Boltworks", "acme")), all.equal(tot$line_total, c(60, 35)))
