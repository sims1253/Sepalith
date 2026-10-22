sample_ledger <- function() {
  ledger(
    date = c("2026-01-02", "2026-01-05", "2026-02-10", "2026-03-01", "2026-03-15"),
    type = c("deposit", "buy", "buy", "dividend", "sell"),
    symbol = c(NA, "ACME", "GLOBX", "ACME", "ACME"),
    quantity = c(NA, 10, 5, NA, 4),
    price = c(NA, 100, 40, NA, 120),
    amount = c(5000, NA, NA, 12.5, NA),
    currency = "eur"
  )
}

test_that("money() rounds to minor units and validates the currency", {
  expect_equal(vctrs::vec_data(money(1.239, "USD")), 1.24)
  expect_equal(vctrs::vec_data(money(1.6, "JPY")), 2)
  expect_error(money(1, "EURO"), "three-letter")
  expect_error(money(Inf, "USD"), "finite")
})

test_that("money arithmetic respects currencies", {
  a <- money(10, "EUR")
  b <- money(2.5, "EUR")
  expect_equal(vctrs::vec_data(a + b), 12.5)
  expect_equal(currency(a * 3), "EUR")
  expect_equal(a / b, 4)
  expect_error(a + money(1, "USD"), "Can't combine")
  expect_error(a * b)
})

test_that("ledger() derives trade amounts and signs cash outflows", {
  led <- sample_ledger()
  e <- led$entries
  expect_s3_class(led, "foliokit_ledger")
  expect_equal(led$currency, "EUR")
  expect_equal(e$amount[e$type == "buy"], c(-1000, -200))
  expect_equal(e$amount[e$type == "sell"], 480)
  expect_equal(ledger()$currency, "USD")
  expect_equal(nrow(ledger()$entries), 0L)
})

test_that("validate_ledger() rejects malformed input", {
  expect_error(ledger(date = "2026-01-01", type = "gift", amount = 10), "Unknown transaction")
  expect_error(
    ledger(date = "2026-01-01", type = "buy", symbol = "ACME", quantity = 0, price = 5),
    "positive quantity"
  )
})

test_that("ledger_add() keeps entries in date order", {
  led <- ledger_add(sample_ledger(), date = "2026-01-03", type = "fee", amount = 7)
  expect_equal(nrow(led$entries), 6L)
  expect_false(is.unsorted(led$entries$date))
  expect_equal(led$entries$amount[[2]], -7)
  expect_equal(led$currency, "EUR")
})

test_that("summary() totals cash flows by type", {
  s <- summary(sample_ledger())
  expect_s3_class(s, "summary_foliokit_ledger")
  expect_equal(s$n, 5L)
  expect_equal(s$totals[["dividend"]], 12.5)
  expect_equal(s$cash, 5000 - 1000 - 200 + 12.5 + 480)
  expect_output(print(s), "Ledger summary \\(EUR\\)")
})

test_that("ledger_positions() replays trades with average cost", {
  pos <- ledger_positions(sample_ledger(), as_of = "2026-12-31")
  expect_s3_class(pos, "foliokit_positions")
  expect_equal(pos$symbol, c("ACME", "GLOBX"))
  expect_equal(pos$quantity, c(6, 5))
  expect_equal(pos$cost, c(600, 200))
})

test_that("value_positions() and as.data.frame() add market columns", {
  pos <- ledger_positions(sample_ledger(), as_of = "2026-12-31")
  valued <- value_positions(pos, c(ACME = 110, GLOBX = 50, OTHER = 1))
  expect_equal(valued$value, c(660, 250))
  expect_equal(sum(valued$weight), 1)
  df <- as.data.frame(valued)
  expect_false(inherits(df, "foliokit_positions"))
  expect_equal(unique(df$currency), "EUR")
  expect_error(value_positions(pos, c(ACME = 1)), "No price for: GLOBX")
})
