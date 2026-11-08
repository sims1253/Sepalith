# skipped: needs shiny (filters_ui, filters_server)
sales <- data.frame(
  region = c("South", "North", "North", "East"),
  channel = c("Web", "Store", "Web", "Web"),
  segment = c("B2C", "B2B", "B2C", "B2C"),
  product = c("Core", "Plus", "Core", "Core"),
  date = as.Date(c("2024-01-05", "2024-02-10", "2024-03-15", "2024-04-20")),
  amount = c(10, 20, 30, 40)
)

ch <- filter_choices(sales)
stopifnot(identical(names(ch), c("region", "channel", "segment", "product", "dates")))
stopifnot(identical(ch$region, c("East", "North", "South")),
          identical(ch$product, c("Core", "Plus")))
stopifnot(identical(ch$dates, as.Date(c("2024-01-05", "2024-04-20"))))
stopifnot(inherits(try(filter_choices(sales[, 1:3]), silent = TRUE), "try-error"))

stopifnot(nrow(apply_filters(sales)) == 4L)
stopifnot(identical(apply_filters(sales, region = "North")$amount, c(20, 30)))
stopifnot(identical(apply_filters(sales, channel = "Web", segment = "B2C")$amount, c(10, 30, 40)))
stopifnot(identical(apply_filters(sales, product = "Plus")$amount, 20))
stopifnot(identical(apply_filters(sales, dates = c("2024-02-01", "2024-03-31"))$amount, c(20, 30)))

stopifnot(identical(describe_filters(), "All data"))
stopifnot(identical(describe_filters(region = "North", product = c("Core", "Plus")),
                    "region: North; product: Core, Plus"))
stopifnot(identical(describe_filters(channel = c("a", "b", "c"), segment = "B2B"),
                    "channel: a, b +1; segment: B2B"))

fc <- filter_counts(sales, region = "North", product = c("Core", "Plus"))
stopifnot(identical(names(fc), c("filter", "selected", "kept", "share")))
stopifnot(identical(fc$filter, c("region", "channel", "segment", "product")))
stopifnot(identical(fc$kept, c(2L, 4L, 4L, 4L)), identical(fc$selected, c(1L, 0L, 0L, 2L)))
fc2 <- filter_counts(sales, channel = "Web", segment = "B2B")
stopifnot(identical(fc2$kept[2:3], c(3L, 1L)), identical(fc2$share[3], 0.25))
