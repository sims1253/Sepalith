test_that("present_value and future_value are inverses", {
  amount <- c(100, 2500, 0.5)
  pv <- present_value(amount, 0.04, 7)
  expect_equal(future_value(pv, 0.04, 7), amount)
  expect_equal(present_value(110, 0.1, 1), 100)
  expect_equal(future_value(100, 0.1, 1), 110)
})

test_that("negative periods are rejected", {
  expect_error(present_value(100, 0.05, -1), "non-negative")
  expect_error(future_value(100, 0.05, -1), "non-negative")
  expect_error(present_value(100, -1.5, 3), "greater than -1")
})

test_that("annuity_pv handles zero rates and annuities due", {
  expect_equal(annuity_pv(100, 0, 12), 1200)
  ordinary <- annuity_pv(100, 0.05, 10)
  due <- annuity_pv(100, 0.05, 10, due = TRUE)
  expect_equal(due, ordinary * 1.05)
  expect_equal(round(ordinary, 2), 772.17)
})

test_that("annuity_fv handles zero rates and annuities due", {
  expect_equal(annuity_fv(100, 0, 12), 1200)
  ordinary <- annuity_fv(100, 0.05, 10)
  due <- annuity_fv(100, 0.05, 10, due = TRUE)
  expect_equal(due, ordinary * 1.05)
  expect_equal(round(ordinary, 2), 1257.79)
})

test_that("annuity_payment repays the principal exactly", {
  pmt <- annuity_payment(10000, 0.01, 24)
  expect_equal(annuity_pv(pmt, 0.01, 24), 10000)
  with_balloon <- annuity_payment(10000, 0.01, 24, balloon = 2000)
  expect_lt(with_balloon, pmt)
  expect_equal(annuity_payment(1200, 0, 12), 100)
})

test_that("net_present_value respects the start period", {
  flows <- c(-1000, 300, 400, 500)
  expect_equal(net_present_value(flows, 0), 200)
  shifted <- net_present_value(flows, 0.1, start = 1)
  expect_equal(shifted, net_present_value(flows, 0.1) / 1.1)
  expect_equal(net_present_value(numeric(0), 0.1), 0)
})

test_that("internal_rate zeroes the net present value", {
  flows <- c(-1000, 300, 400, 500)
  irr <- internal_rate(flows)
  expect_equal(net_present_value(flows, irr), 0, tolerance = 1e-6)
  expect_gt(irr, 0.08)
  expect_lt(irr, 0.09)
})

test_that("internal_rate warns without a sign change", {
  expect_warning(out <- internal_rate(c(100, 200, 300)), "sign change")
  expect_true(is.na(out))
})
