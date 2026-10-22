sampleTransactions <- function() {
  data.frame(
    date = as.Date(c(
      "2026-09-02", "2026-09-03", "2026-09-15", "2026-09-20", "2026-09-28", "2026-10-01"
    )),
    category = c("groceries", "rent", "groceries", "fun", "salary", "rent"),
    amount = c(-120, -900, -80, -45, 2600, -900),
    stringsAsFactors = FALSE
  )
}

sampleLimits <- function() {
  c(groceries = 250, rent = 900, fun = 40, transport = 100)
}

test_that("spendByCategory ignores income and other months", {
  spent <- spendByCategory(sampleTransactions(), "2026-09")
  expect_equal(spent[["groceries"]], 200)
  expect_equal(spent[["rent"]], 900)
  expect_equal(spent[["fun"]], 45)
  expect_false("salary" %in% names(spent))
})

test_that("spendByCategory returns an empty vector for a quiet month", {
  spent <- spendByCategory(sampleTransactions(), "2026-07")
  expect_length(spent, 0)
  expect_type(spent, "double")
})

test_that("budgetVariance treats unspent categories as zero", {
  variance <- budgetVariance(spendByCategory(sampleTransactions(), "2026-09"), sampleLimits())
  expect_equal(variance$category, names(sampleLimits()))
  expect_equal(variance$spent, c(200, 900, 45, 0))
  expect_equal(variance$remaining, c(50, 0, -5, 100))
})

test_that("varianceStatus distinguishes under, near and over", {
  expect_equal(varianceStatus(c(10, 95, 100, 101), 100), c("under", "near", "near", "over"))
  expect_equal(varianceStatus(80, 100, warnAt = 0.75), "near")
})

test_that("formatMoney adds separators and keeps the sign", {
  expect_equal(formatMoney(c(1234.5, -0.25, 0)), c("$1,234.50", "-$0.25", "$0.00"))
  expect_equal(formatMoney(12, symbol = "EUR "), "EUR 12.00")
})

test_that("the module reports status for the chosen month", {
  shiny::testServer(
    categoryBudgetServer,
    args = list(transactions = shiny::reactive(sampleTransactions()), limits = sampleLimits()),
    {
      session$setInputs(month = "2026-09", category = "groceries")
      expect_equal(variance()$status, c("under", "near", "over", "under"))
      session$setInputs(month = "2026-10")
      expect_equal(variance()$spent, c(0, 900, 0, 0))
    }
  )
})

test_that("saving a limit updates the returned limits", {
  shiny::testServer(
    categoryBudgetServer,
    args = list(transactions = shiny::reactive(sampleTransactions()), limits = sampleLimits()),
    {
      session$setInputs(month = "2026-09", category = "fun", limit = 60, setLimit = 1)
      expect_equal(session$returned()[["fun"]], 60)
      expect_equal(output$message, "Limit for fun set to $60.00.")
      expect_equal(variance()$status[3], "under")
    }
  )
})

test_that("a negative limit is rejected", {
  shiny::testServer(
    categoryBudgetServer,
    args = list(transactions = shiny::reactive(sampleTransactions()), limits = sampleLimits()),
    {
      session$setInputs(month = "2026-09", category = "rent", limit = -5, setLimit = 1)
      expect_equal(session$returned()[["rent"]], 900)
      expect_equal(output$message, "Enter a limit of zero or more.")
    }
  )
})
