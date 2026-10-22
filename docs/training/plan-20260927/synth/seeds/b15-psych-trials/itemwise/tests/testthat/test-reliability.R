make_responses <- function(n = 200, k = 6, seed = 11) {
  set.seed(seed)
  trait <- rnorm(n)
  out <- sapply(seq_len(k), function(j) {
    raw <- trait + rnorm(n, sd = 0.8)
    as.integer(cut(raw, breaks = c(-Inf, -1, -0.3, 0.3, 1, Inf)))
  })
  colnames(out) <- paste0("q", seq_len(k))
  as.data.frame(out)
}

test_that("reverse_key mirrors responses around the scale midpoint", {
  expect_equal(reverse_key(1:5, min = 1, max = 5), 5:1)
  expect_equal(reverse_key(c(0, 3), min = 0, max = 3), c(3, 0))
  expect_error(reverse_key("a", min = 1, max = 5), "numeric")
})

test_that("score_sum and score_mean respect reverse keys and missingness", {
  resp <- data.frame(a = c(1, 5, NA), b = c(5, 1, 2), c = c(2, 4, 3))
  sums <- score_sum(resp, c("a", "b", "c"), reverse = "b", max_missing = 0)
  means <- score_mean(resp, c("a", "b", "c"), reverse = "b", max_missing = 1)
  expect_equal(sums, c(4, 14, NA))
  expect_equal(means, c(4 / 3, 14 / 3, 3.5))
})

test_that("score_scales returns one column per scale in the key", {
  resp <- make_responses(n = 20, k = 4)
  key <- list(
    first = list(items = c("q1", "q2")),
    second = list(items = c("q3", "q4"), reverse = "q4")
  )
  scored <- score_scales(resp, key, method = "sum", min = 1, max = 5)
  expect_named(scored, c("first", "second"))
  expect_equal(nrow(scored), 20)
  expect_equal(scored$first, resp$q1 + resp$q2)
})

test_that("cronbach_alpha is high for consistent items and errors on one item", {
  resp <- make_responses()
  alpha <- cronbach_alpha(resp, names(resp))
  expect_gt(alpha, 0.8)
  expect_lt(alpha, 1)
  expect_error(cronbach_alpha(resp, "q1"), "at least two items")
})

test_that("alpha_if_deleted returns one value per item", {
  resp <- make_responses()
  resp$noise <- sample(1:5, nrow(resp), replace = TRUE)
  items <- names(resp)
  dropped <- alpha_if_deleted(resp, items)
  expect_named(dropped, items)
  expect_equal(names(which.max(dropped)), "noise")
})

test_that("corrected item-total correlations are below uncorrected ones", {
  resp <- make_responses()
  corrected <- item_total_cor(resp, names(resp), corrected = TRUE)
  uncorrected <- item_total_cor(resp, names(resp), corrected = FALSE)
  expect_true(all(corrected < uncorrected))
  expect_true(all(corrected > 0))
})

test_that("spearman_brown and spearman_brown_factor are inverses", {
  expect_equal(spearman_brown(0.6, factor = 1), 0.6)
  expect_equal(spearman_brown(0.5, factor = 2), 2 / 3)
  needed <- spearman_brown_factor(0.7, target = 0.9)
  expect_equal(spearman_brown(0.7, factor = needed), 0.9)
})

test_that("split_half supports every split rule", {
  resp <- make_responses()
  for (rule in c("odd_even", "first_last", "random")) {
    res <- split_half(resp, names(resp), split = rule, seed = 3)
    expect_equal(res$split, rule)
    expect_gt(res$rel, res$r_half)
  }
})

test_that("reliability_report prints a summary table", {
  resp <- make_responses(n = 80)
  rep <- reliability_report(resp, names(resp), name = "mood")
  expect_s3_class(rep, "itemwise_reliability")
  expect_equal(nrow(rep$items), ncol(resp))
  expect_output(print(rep), "Reliability of 'mood'")
  expect_output(print(rep), "Cronbach's alpha")
})
