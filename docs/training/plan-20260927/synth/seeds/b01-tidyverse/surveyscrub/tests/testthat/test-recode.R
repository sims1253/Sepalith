responses <- tibble::tibble(
  id = 1:4,
  q1 = c("Strongly agree", "agree", "  Disagree ", "No opinion"),
  q2 = c("Never", "ALWAYS", "sometimes", NA),
  q3 = c("Yes", "no", "Don't know", "y")
)

test_that("recode_agree5 maps agreement labels to 1-5", {
  out <- recode_agree5(responses, q1)
  expect_type(out$q1, "integer")
  expect_equal(out$q1, c(5L, 4L, 2L, NA))
})

test_that("recode_freq5 maps frequency labels to 1-5", {
  out <- recode_freq5(responses, q2)
  expect_type(out$q2, "integer")
  expect_equal(out$q2, c(1L, 5L, 3L, NA))
})

test_that("recode_yesno maps yes/no labels to logicals", {
  out <- recode_yesno(responses, q3)
  expect_type(out$q3, "logical")
  expect_equal(out$q3, c(TRUE, FALSE, NA, TRUE))
})

test_that("recoders leave unselected columns alone", {
  out <- recode_agree5(responses, q1)
  expect_identical(out$q2, responses$q2)
  expect_identical(out$q3, responses$q3)
})

test_that("reverse_items flips scores on the requested scale", {
  scores <- tibble::tibble(a = c(1L, 3L, 5L), b = c(1L, 4L, 7L))
  expect_equal(reverse_items(scores, a)$a, c(5L, 3L, 1L))
  expect_equal(reverse_items(scores, b, points = 7L)$b, c(7L, 4L, 1L))
  expect_error(reverse_items(scores, a, points = 1), "at least 2")
})

test_that("recode_missing_codes replaces sentinel values with NA", {
  scores <- tibble::tibble(a = c(1, -99, 3), b = c(999, 2, -98))
  out <- recode_missing_codes(scores, c(a, b))
  expect_equal(out$a, c(1, NA, 3))
  expect_equal(out$b, c(NA, 2, NA))
})

test_that("collapse_rare lumps infrequent levels into Other", {
  pets <- tibble::tibble(pet = c(rep("dog", 6), rep("cat", 3), "ferret"))
  out <- collapse_rare(pets, pet, min_prop = 0.2)
  expect_s3_class(out$pet, "factor")
  expect_equal(levels(out$pet), c("dog", "cat", "Other"))
  expect_equal(sum(out$pet == "Other"), 1L)
})
