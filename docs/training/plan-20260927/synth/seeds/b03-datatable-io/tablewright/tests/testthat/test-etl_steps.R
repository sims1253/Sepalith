make_orders <- function() {
  data.table::data.table(
    order_id = c("A1", "A2", "A3", "A4"),
    customer = c("c01", "c02", "c01", "c03"),
    amount = c("10.5", "3", "oops", "7.25"),
    placed = c("2026-01-03", "2026-01-04", "2026-01-04", "2026-01-09")
  )
}

test_that("check_has_cols reports every missing column", {
  dt <- make_orders()
  expect_invisible(check_has_cols(dt, c("order_id", "amount")))
  expect_error(
    check_has_cols(dt, c("order_id", "region", "channel"), "load"),
    "\\[load\\] missing column\\(s\\): region, channel"
  )
})

test_that("check_unique_key detects duplicated keys", {
  dt <- make_orders()
  expect_invisible(check_unique_key(dt, "order_id"))
  expect_error(check_unique_key(dt, "customer", "dedupe"), "1 duplicated key")
})

test_that("check_no_missing counts NA values per column", {
  dt <- make_orders()
  dt[2, customer := NA]
  expect_invisible(check_no_missing(dt, "order_id"))
  expect_error(check_no_missing(dt, c("order_id", "customer")), "customer \\(1\\)")
})

test_that("step_rename renames by reference and logs the step", {
  dt <- make_orders()
  step_rename(dt, c(id = "order_id", cust = "customer"))
  expect_named(dt, c("id", "cust", "amount", "placed"))
  expect_equal(attr(dt, "tw_log")$step, "rename")
  expect_equal(attr(dt, "tw_log")$n, 2L)
})

test_that("step_cast converts types and warns about lost values", {
  dt <- make_orders()
  expect_warning(
    step_cast(dt, c(amount = "numeric", placed = "Date")),
    "1 value\\(s\\) in 'amount' became NA"
  )
  expect_type(dt$amount, "double")
  expect_s3_class(dt$placed, "Date")
  expect_true(is.na(dt$amount[3]))
})

test_that("step_cast rejects unknown target types", {
  dt <- make_orders()
  expect_error(step_cast(dt, c(amount = "decimal")), "Unsupported type\\(s\\): decimal")
})

test_that("step_derive evaluates grouped expressions", {
  dt <- make_orders()
  suppressWarnings(step_cast(dt, c(amount = "numeric")))
  step_derive(dt, list(n_orders = quote(.N)), by = "customer")
  expect_equal(dt$n_orders, c(2L, 1L, 2L, 1L))
  expect_error(step_derive(dt, list(quote(.N))), "must be named")
})

test_that("step_filter drops rows and treats NA as FALSE", {
  dt <- make_orders()
  suppressWarnings(step_cast(dt, c(amount = "numeric")))
  out <- step_filter(dt, quote(amount > 5))
  expect_equal(out$order_id, c("A1", "A4"))
  expect_equal(nrow(dt), 4L)
  expect_equal(utils::tail(attr(out, "tw_log")$n, 1), 2L)
})
