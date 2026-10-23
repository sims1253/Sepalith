# skipped: needs shiny (render_box, valueboxes_ui, valueboxes_server)
orders <- data.frame(
  order_id = c(1, 2, 2, 3, NA),
  customer_id = c("a", "b", "b", "a", "c"),
  amount = c(10, 20, 5, 15, NA),
  sessions = c(10, 5, 0, 5, 0),
  duration = c(30, 60, 90, 120, NA),
  date = as.Date(c("2024-01-01", "2024-01-02", "2024-01-08", "2024-01-09", "2024-01-10"))
)

k <- kpi_values(orders)
stopifnot(identical(names(k), c("revenue", "orders", "customers", "conversion",
                                "order_value", "duration")))
stopifnot(k$revenue == 50, k$orders == 3L, k$customers == 3L)
stopifnot(all.equal(k$conversion, 0.15), all.equal(k$order_value, 50 / 3), k$duration == 75)
empty <- kpi_values(orders[0, ])
stopifnot(is.na(empty$conversion), is.na(empty$order_value), is.na(empty$duration))

per <- kpi_periods(orders, "2024-01-08", "2024-01-14")
stopifnot(identical(names(per), c("current", "previous", "year_ago")))
stopifnot(nrow(per$year_ago) == 0L)
ya <- kpi_periods(orders, "2025-01-01", "2025-01-02")
stopifnot(identical(ya$year_ago$order_id, c(1, 2)), nrow(ya$current) == 0L)
stopifnot(nrow(per$current) == 3L, nrow(per$previous) == 2L)
stopifnot(inherits(try(kpi_periods(orders, "2024-01-08", "2024-01-01"), silent = TRUE),
                   "try-error"))

st <- kpi_status(k, list(revenue = 40, orders = 3, customers = 10))
stopifnot(identical(names(st), c("revenue", "orders", "customers", "conversion", "order_value")))
stopifnot(identical(unname(st), c("above", "on", "below", "none", "none")))
stopifnot(identical(kpi_status(k, list(revenue = 51), tolerance = 0.1)[["revenue"]], "on"))

tg <- kpi_targets(revenue = 100, orders = NA, customers = 5, conversion = 0.1, order_value = 20)
stopifnot(identical(names(tg), c("revenue", "customers", "conversion", "order_value")))
stopifnot(identical(kpi_targets(), list()))
for (arg in c("revenue", "orders", "customers", "conversion", "order_value")) {
  msg <- tryCatch(do.call(kpi_targets, stats::setNames(list(-1), arg)),
                  error = conditionMessage)
  stopifnot(identical(msg, sprintf("The `%s` target must not be negative.", arg)))
}
