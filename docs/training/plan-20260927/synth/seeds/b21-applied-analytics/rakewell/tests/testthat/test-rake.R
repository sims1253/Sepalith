make_sample <- function() {
  data.frame(
    sex = c("f", "f", "f", "m", "m", "m", "m", "m"),
    age = c("18-34", "35-64", "65+", "18-34", "18-34", "35-64", "35-64", "65+"),
    region = c("n", "s", "n", "s", "n", "s", "n", "s"),
    stringsAsFactors = FALSE
  )
}

test_that("rake_weights reproduces the sex margin", {
  df <- make_sample()
  margins <- list(sex = c(f = 0.52, m = 0.48))
  w <- rake_weights(df, margins)
  share_f <- sum(w[df$sex == "f"]) / sum(w)
  expect_equal(share_f, 0.52, tolerance = 1e-6)
  expect_true(attr(w, "converged"))
  expect_equal(sum(w), nrow(df))
})

test_that("rake_weights reproduces two margins jointly", {
  df <- make_sample()
  margins <- list(
    sex = c(f = 0.52, m = 0.48),
    age = c("18-34" = 0.3, "35-64" = 0.45, "65+" = 0.25)
  )
  w <- rake_weights(df, margins, total = 1000)
  cmp <- compare_margins(df, w, margins)
  expect_equal(nrow(cmp), 5L)
  expect_lt(max(abs(cmp$difference)), 1e-5)
  expect_equal(sum(w), 1000)
})

test_that("rake_weights accepts margins given as counts", {
  df <- make_sample()
  as_shares <- rake_weights(df, list(sex = c(f = 0.5, m = 0.5)))
  as_counts <- rake_weights(df, list(sex = c(f = 5000, m = 5000)))
  expect_equal(as.numeric(as_shares), as.numeric(as_counts))
})

test_that("check_margins rejects unknown variables", {
  df <- make_sample()
  expect_error(
    check_margins(list(income = c(low = 1, high = 1)), df),
    "not found in `data`"
  )
})

test_that("check_margins rejects categories without a target", {
  df <- make_sample()
  expect_error(
    check_margins(list(age = c("18-34" = 1, "35-64" = 1)), df),
    "without a target"
  )
})

test_that("check_margins rejects negative targets", {
  df <- make_sample()
  expect_error(
    check_margins(list(sex = c(f = -1, m = 2)), df),
    "negative or missing"
  )
})

test_that("poststratify matches population cell counts", {
  df <- make_sample()
  pop <- data.frame(
    sex = c("f", "f", "m", "m"),
    region = c("n", "s", "n", "s"),
    n = c(260, 240, 250, 250)
  )
  w <- poststratify(df, pop, strata = c("sex", "region"))
  keys <- paste(df$sex, df$region, sep = ":")
  expect_equal(sum(w[keys == "f:n"]), 260)
  expect_equal(sum(w[keys == "m:s"]), 250)
  expect_equal(sum(w), 1000)
})

test_that("weight diagnostics agree for equal weights", {
  w <- rep(2, 10)
  expect_equal(kish_deff(w), 1)
  expect_equal(effective_n(w), 10)
  expect_equal(weight_cv(w), 0)
  report <- weight_report(w, label = "flat")
  expect_s3_class(report, "weight_report")
  expect_equal(report$ratio, 1)
})

test_that("trimming keeps the total and respects the cap", {
  w <- c(rep(1, 9), 30)
  capped <- trim_weights_cap(w, cap = 5)
  winsorised <- trim_weights_quantile(w, lower = 0, upper = 0.9)
  expect_equal(sum(capped), sum(w))
  expect_equal(sum(winsorised), sum(w))
  expect_lte(max(capped) / stats::median(capped), 5 + 1e-8)
})
