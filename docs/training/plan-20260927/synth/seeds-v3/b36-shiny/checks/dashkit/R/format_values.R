stopifnot(identical(format_kpi(c(1234.5, -20, NA), "currency"), c("$1,234.5", "-$20.0", "-")))
stopifnot(identical(format_kpi(0.1234, "percent"), "12.3%"))
stopifnot(identical(format_kpi(c(12345, 7), "count"), c("12,345", "7")))
stopifnot(identical(format_kpi(c(125, 59), "duration"), c("2m 05s", "0m 59s")))
stopifnot(identical(format_kpi(1.25, "ratio", digits = 2), "1.25x"))

up <- format_change(110, 100)
stopifnot(identical(up, list(text = "+10.0%", direction = "up", icon = "arrow-up")))
down <- format_change(75, 100)
stopifnot(identical(down$text, "-25.0%"), identical(down$icon, "arrow-down"))
stopifnot(identical(format_change(5, 5)$direction, "flat"))
stopifnot(identical(format_change(5, 0)$text, "n/a"))

stopifnot(identical(format_compact(c(950, 1500, 2500000, -3.2e9, NA)),
                    c("950", "1.5K", "2.5M", "-3.2B", "-")))
stopifnot(identical(format_compact(12345, digits = 0), "12K"))

d <- as.Date("2024-03-06")
stopifnot(identical(format_period_label(d, "week"), "Week of 2024-03-04"))
stopifnot(identical(format_period_label(d, "month"), "March 2024"))
stopifnot(identical(format_period_label(d, "quarter"), "Q1 2024"))
stopifnot(identical(format_period_label(d, "year"), "2024"))
stopifnot(grepl("2024$", format_period_label(d, "day")))

tab <- data.frame(rev = c(10, 2000.5), share = c(0.5, NA), n = c(3, 4000),
                  day = as.Date(c("2024-01-02", NA)))
ft <- format_table_columns(tab, currency = "rev", percent = "share", count = "n", date = "day")
stopifnot(identical(ft$rev, c("$10.00", "$2,000.50")), identical(ft$share, c("50.0%", "-")))
stopifnot(identical(ft$n, c("3", "4,000")), identical(ft$day, c("02 Jan 2024", "-")))
stopifnot(inherits(try(format_table_columns(tab, count = "zzz"), silent = TRUE), "try-error"))

spec <- value_box_spec(0.25, 0.2, "percent", "Conversion")
stopifnot(identical(names(spec), c("title", "value", "change", "direction", "icon", "color")))
stopifnot(identical(spec$value, "25.0%"), identical(spec$color, "green"))
stopifnot(identical(value_box_spec(5, 10, "count", "Orders")$color, "red"))
stopifnot(identical(value_box_spec(5, NA, "count", "Orders")$color, "gray"))

cur <- list(revenue = 1200, orders = 30, conversion = 0.05, order_value = 40)
prev <- list(revenue = 1000, orders = 40, conversion = 0.05, order_value = NA)
dt <- format_delta_table(cur, prev)
stopifnot(identical(names(dt), c("kpi", "current", "previous", "change")))
stopifnot(identical(dt$kpi, c("revenue", "orders", "conversion", "order_value")))
stopifnot(identical(dt$current, c("$1,200.0", "30", "5.0%", "$40.0")))
stopifnot(identical(dt$previous[4], "-"),
          identical(dt$change, c("+20.0%", "-25.0%", "0.0%", "n/a")))
