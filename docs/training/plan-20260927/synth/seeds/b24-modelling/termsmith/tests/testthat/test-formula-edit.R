test_that("add_terms appends new terms and keeps the environment", {
  env <- new.env()
  f <- stats::as.formula("y ~ a + b", env = env)
  out <- add_terms(f, c("c", "a"))
  expect_equal(attr(stats::terms(out), "term.labels"), c("a", "b", "c"))
  expect_identical(environment(out), env)
  expect_equal(deparse1(out[[2L]]), "y")
})

test_that("drop_terms removes terms and falls back to an intercept", {
  f <- y ~ a + b
  out <- drop_terms(f, "b")
  expect_equal(attr(stats::terms(out), "term.labels"), "a")
  empty <- drop_terms(f, c("a", "b"))
  expect_length(attr(stats::terms(empty), "term.labels"), 0L)
  expect_equal(attr(stats::terms(empty), "intercept"), 1L)
})

test_that("editors return the input unchanged when there is nothing to do", {
  f <- y ~ a + b
  expect_identical(add_terms(f, "a"), f)
  expect_identical(drop_terms(f, "z"), f)
  expect_identical(expand_dot(f, data.frame(y = 1, a = 1, b = 1)), f)
})

test_that("swap_response accepts strings and calls", {
  f <- y ~ a + b
  out_chr <- swap_response(f, "log(z)")
  out_call <- swap_response(f, quote(log(z)))
  expect_equal(deparse1(out_chr), "log(z) ~ a + b")
  expect_equal(out_chr, out_call)
  expect_equal(lhs_vars(out_chr), "z")
})

test_that("lhs_vars and rhs_vars split the variables", {
  f <- cbind(s, f) ~ x + log(w) + x:w
  expect_equal(lhs_vars(f), c("s", "f"))
  expect_equal(rhs_vars(f), c("x", "w"))
  expect_equal(lhs_vars(~ x), character())
  expect_equal(rhs_vars(y ~ .), character())
})

test_that("interact_with moderates every main effect by default", {
  out <- interact_with(y ~ a + b, "g")
  labels <- attr(stats::terms(out), "term.labels")
  expect_true(all(c("g", "a:g", "b:g") %in% labels))
  only_a <- interact_with(y ~ a + b, "g", terms = "a")
  expect_false("b:g" %in% attr(stats::terms(only_a), "term.labels"))
  expect_error(interact_with(y ~ a, "g", terms = "q"), "Unknown term")
})

test_that("formula_signature ignores term and interaction order", {
  sig_1 <- formula_signature(y ~ a + b + a:b)
  sig_2 <- formula_signature(y ~ b:a + b + a)
  expect_identical(sig_1, sig_2)
  expect_match(formula_signature(y ~ a - 1), "- 1$")
  expect_identical(formula_signature(~ 1), "~ 1")
})

test_that("check_formula raises a classed error", {
  expect_error(add_terms("y ~ x", "z"), class = "termsmith_bad_formula")
  expect_error(drop_terms(list(), "z"), class = "termsmith_bad_formula")
  expect_error(swap_response(1, "y"), "must be a formula")
})
