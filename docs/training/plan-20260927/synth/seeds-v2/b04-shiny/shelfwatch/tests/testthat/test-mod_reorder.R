fake_stock <- function() {
  data.frame(
    sku = c("BOL-0042", "NUT-0108", "WSH-0007"),
    name = c("Hex bolt M8", "Lock nut M8", "Flat washer M8"),
    warehouse = c("north", "north", "south"),
    category = c("fasteners", "fasteners", "fasteners"),
    on_hand = c(12, 340, 0),
    reorder_point = c(50, 100, 80),
    target_level = c(200, 400, 300),
    unit_cost = c(0.35, 0.12, 0.04),
    stringsAsFactors = FALSE
  )
}

test_that("reorder_quantity returns zero above the reorder point", {
  expect_identical(reorder_quantity(120, 50, 200), 0L)
  expect_identical(reorder_quantity(51, 50, 200), 0L)
})

test_that("reorder_quantity rounds the shortfall up to whole cases", {
  expect_identical(reorder_quantity(12, 50, 200), 188L)
  expect_identical(reorder_quantity(12, 50, 200, case_size = 25L), 200L)
  expect_identical(reorder_quantity(0, 80, 300, case_size = 100L), 300L)
})

test_that("a valid line is added to the pending orders", {
  shiny::testServer(mod_reorder_server, args = list(stock = shiny::reactive(fake_stock())), {
    session$setInputs(sku = "BOL-0042", qty = 150, supplier = "  acme  fasteners ", add = 1)
    pending <- session$returned()

    expect_equal(nrow(pending), 1)
    # supplier names are tidied before they are stored
    expect_equal(pending$supplier, "Acme Fasteners")
    expect_equal(pending$line_total, 150 * 0.35)
    expect_match(output$feedback, "Added 150 x BOL-0042")
  })
})

test_that("an invalid quantity is reported and nothing is added", {
  shiny::testServer(mod_reorder_server, args = list(stock = shiny::reactive(fake_stock())), {
    session$setInputs(sku = "BOL-0042", qty = 2.5, supplier = "Acme", add = 1)
    expect_equal(nrow(session$returned()), 0)
    expect_equal(output$feedback, "Quantity must be a whole number.")
  })
})

test_that("a missing supplier is reported and nothing is added", {
  shiny::testServer(mod_reorder_server, args = list(stock = shiny::reactive(fake_stock())), {
    session$setInputs(sku = "NUT-0108", qty = 60, supplier = "   ", add = 1)
    expect_equal(nrow(session$returned()), 0)
    expect_equal(output$feedback, "Supplier cannot be blank.")
  })
})

test_that("an unknown SKU is reported and nothing is added", {
  shiny::testServer(mod_reorder_server, args = list(stock = shiny::reactive(fake_stock())), {
    session$setInputs(sku = "ZZZ-9999", qty = 10, supplier = "Acme", add = 1)
    expect_equal(nrow(session$returned()), 0)
    expect_equal(output$feedback, "SKU ZZZ-9999 is not in stock data.")
  })
})

test_that("lines accumulate and the clear button empties them", {
  shiny::testServer(mod_reorder_server, args = list(stock = shiny::reactive(fake_stock())), {
    session$setInputs(sku = "BOL-0042", qty = 100, supplier = "Acme", add = 1)
    session$setInputs(sku = "WSH-0007", qty = 300, supplier = "Acme", add = 2)
    expect_equal(nrow(session$returned()), 2)

    session$setInputs(clear = 1)
    expect_equal(nrow(session$returned()), 0)
    expect_equal(output$feedback, "Cleared all pending lines.")
  })
})

test_that("order_totals_by_supplier sorts suppliers by spend", {
  orders <- rbind(
    build_order_row(fake_stock(), "BOL-0042", 100, "Acme"),
    build_order_row(fake_stock(), "NUT-0108", 500, "Boltworks"),
    build_order_row(fake_stock(), "WSH-0007", 300, "Acme")
  )

  totals <- order_totals_by_supplier(orders)
  expect_equal(totals$supplier, c("Boltworks", "Acme"))
  expect_equal(totals$line_total, c(60, 47))
  expect_equal(totals$qty, c(500, 400))
})

test_that("build_order_row rejects unknown SKUs", {
  expect_error(
    build_order_row(fake_stock(), "ZZZ-9999", 10, "Acme"),
    "Unknown SKU: ZZZ-9999"
  )
})
