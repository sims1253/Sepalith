e0 <- emptyIssues()
stopifnot(nrow(e0) == 0L, identical(names(e0), c("row", "field", "value", "problem")))

a1 <- addIssues(e0, "f", c("x", "y", "z"), c(FALSE, TRUE, NA), "bad")
stopifnot(identical(a1$row, 2L), identical(a1$value, "y"))
stopifnot(identical(addIssues(e0, "f", "x", FALSE, "bad"), e0))

orders <- data.frame(
  orderId = c("ORD-000001", "ORD-000001", "ord-2", "ORD-000003"),
  customerId = c("C00001", "C00001", "C00002", "C2"),
  sku = c("ABC-1234", "abc-1", "XYZ-0001", "XYZ-0002"),
  quantity = c(2, 0, 5, 2000),
  unitPrice = c(10, 5, -1, 1),
  discount = c(0, NA, 0.1, 1.5),
  currency = c("EUR", "EUR", "CHF", NA),
  orderDate = c("2024-01-01", "2024-01-02", "2099-01-01", NA),
  shipDate = c("2024-01-03", "2023-12-31", NA, "2024-01-05"),
  deliveryDate = c("2024-01-02", NA, "2099-02-01", "2024-01-06"),
  stringsAsFactors = FALSE
)

iss <- validateOrderLines(orders)
stopifnot(identical(iss$row, c(2L, 2L, 3L, 3L, 3L, 4L, 4L, 4L, 4L)))
stopifnot(identical(iss$field, c("sku", "quantity", "orderId", "unitPrice", "currency",
                                 "customerId", "quantity", "discount", "currency")))
stopifnot(identical(iss$problem[6], "not of the form Cnnnnn"))
stopifnot(identical(iss$problem[1], "not of the form AAA-nnnn"))
stopifnot(sum(validateOrderLines(orders, currencies = c("EUR", "CHF"))$field ==
                "currency") == 1L)
stopifnot(!"2000" %in% validateOrderLines(orders, maxQuantity = 5000)$value)

dt <- validateOrderDates(orders, today = as.Date("2024-06-01"))
stopifnot(identical(dt$row, c(1L, 2L, 3L, 3L, 4L)))
stopifnot(identical(dt$field, c("deliveryDate", "shipDate", "orderDate", "deliveryDate",
                                "orderDate")))
stopifnot(identical(dt$problem[1], "before shipDate"))

totals <- data.frame(orderId = c("ORD-000001", "ord-2", "ORD-000003"),
                     total = c(20, -5, 2000.5), stringsAsFactors = FALSE)
tt <- validateOrderTotals(orders, totals)
stopifnot(identical(names(tt), c("orderId", "expected", "reported", "difference")))
stopifnot(identical(tt$orderId, "ORD-000003"), isTRUE(all.equal(tt$difference, 0.5)))
stopifnot(nrow(validateOrderTotals(orders, totals, tolerance = 1)) == 0L)
stopifnot(identical(validateOrderTotals(orders, totals[1, ])$orderId, c("ORD-000003", "ord-2")))

ok <- list(street = "Main St", houseNumber = "12a", postcode = "10115", city = "Berlin",
           country = "DE")
stopifnot(identical(validateAddress(ok), character()))
stopifnot(identical(validateAddress(list(street = "x", city = " ")),
                    c("`houseNumber` is missing.", "`postcode` is missing.",
                      "`city` is missing.", "`country` is missing.")))
bad <- modifyList(ok, list(houseNumber = "x1", country = "Germany", city = "B3rlin"))
stopifnot(identical(validateAddress(bad), c("`houseNumber` is malformed.",
                                            "`city` is malformed.", "`country` is malformed.")))
bad2 <- modifyList(ok, list(street = "123", postcode = "1"))
stopifnot(identical(validateAddress(bad2), c("`street` is malformed.",
                                             "`postcode` is malformed.")))

ship <- data.frame(
  trackingNo = c("T1", "T1", NA),
  carrier = c("DHL", "FedEx", "UPS"),
  weightKg = c(2, 40, 1),
  lengthCm = c(30, 30, 0),
  widthCm = c(20, 200, 20),
  heightCm = c(10, 10, NA),
  insuredValue = c(NA, 100, -1),
  stringsAsFactors = FALSE
)
si <- validateShipments(ship)
stopifnot(identical(si$row, c(2L, 2L, 2L, 2L, 3L, 3L, 3L, 3L)))
stopifnot(identical(si$field, c("trackingNo", "carrier", "weightKg", "widthCm",
                                "trackingNo", "lengthCm", "heightCm", "insuredValue")))
stopifnot(identical(si$problem[2], "not an allowed code"))
stopifnot(nrow(validateShipments(ship[1, ])) == 0L)
stopifnot(!"40" %in% validateShipments(ship, carriers = "FedEx", maxKg = 50)$value)
stopifnot(inherits(try(validateOrderTotals(orders, totals["orderId"]), silent = TRUE),
                   "try-error"))
