test_that("strSquish() trims and collapses whitespace", {
  expect_identical(strSquish("  too   many\tspaces "), "too many spaces")
  expect_identical(strSquish(c("a", " b ", "")), c("a", "b", ""))
  expect_identical(strSquish(NA_character_), NA_character_)
  expect_error(strSquish(1:3), class = "pathwren_error_type")
})

test_that("strPadLeft() right-aligns text", {
  expect_identical(strPadLeft(c("7", "42", "512"), 3), c("  7", " 42", "512"))
  expect_identical(strPadLeft("7", 3, pad = "0"), "007")
  expect_identical(strPadLeft("toolong", 3), "toolong")
  expect_identical(strPadLeft(c("a", NA), 2), c(" a", NA))
  expect_error(strPadLeft("a", 3, pad = "ab"), class = "pathwren_error_pad")
})

test_that("strPadRight() left-aligns text", {
  expect_identical(strPadRight(c("7", "42", "512"), 3), c("7  ", "42 ", "512"))
  expect_identical(strPadRight("7", 3, pad = "."), "7..")
  expect_identical(strPadRight("toolong", 3), "toolong")
  expect_identical(strPadRight(c("a", NA), 2), c("a ", NA))
  expect_error(strPadRight("a", 3, pad = ""), class = "pathwren_error_pad")
})

test_that("strTruncate() keeps results within the width", {
  x = c("short", "a considerably longer label", NA)
  out = strTruncate(x, 10)
  expect_identical(out, c("short", "a consi...", NA))
  expect_true(all(nchar(out[!is.na(out)]) <= 10))
  expect_identical(strTruncate("abcdef", 4, ellipsis = "~"), "abc~")
  expect_error(strTruncate("abc", 2), class = "pathwren_error_width")
})

test_that("strToSnake() and strToCamel() are inverses for simple names", {
  camel = c("monthlyNetTotal", "id", "taxRate2")
  snake = c("monthly_net_total", "id", "tax_rate2")
  expect_identical(strToSnake(camel), snake)
  expect_identical(strToCamel(snake), camel)
  expect_identical(strToSnake("Net Total.2025"), "net_total_2025")
  expect_identical(strToCamel("net-total 2025"), "netTotal2025")
})

test_that("case converters handle edge cases", {
  expect_identical(strToSnake(""), "")
  expect_identical(strToCamel(""), "")
  expect_identical(strToSnake("__already__snake__"), "already_snake")
  expect_identical(strToCamel(c("a_b", NA)), c("aB", NA))
  expect_error(strToCamel(list("a")), class = "pathwren_error_type")
})

test_that("strCountMatches() counts regex and fixed patterns", {
  x = c("a.b.c", "abc", "", NA)
  expect_identical(strCountMatches(x, ".", fixed = TRUE), c(2L, 0L, 0L, NA))
  expect_identical(strCountMatches(x, "[a-c]"), c(3L, 3L, 0L, NA))
  expect_identical(strCountMatches("2025-01-31", "[0-9]+"), 3L)
})

test_that("type errors name the offending argument", {
  expect_snapshot(strSquish(factor("a")), error = TRUE)
  expect_snapshot(strPadLeft("a", 2, pad = 0), error = TRUE)
})
