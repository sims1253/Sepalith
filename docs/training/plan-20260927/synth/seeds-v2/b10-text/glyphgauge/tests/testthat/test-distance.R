test_that("levenshteinDist matches hand-computed distances", {
  expect_equal(levenshteinDist("kitten", "sitting"), 3L)
  expect_equal(levenshteinDist("flaw", "lawn"), 2L)
  expect_equal(levenshteinDist("", "abc"), 3L)
  expect_equal(levenshteinDist("same", "same"), 0L)

  expect_equal(levenshteinDist("ABC", "abc", ignoreCase = TRUE), 0L)
})

test_that("osaDist counts an adjacent swap as one edit", {
  expect_equal(osaDist("ab", "ba"), 1L)
  expect_equal(levenshteinDist("ab", "ba"), 2L)

  # OSA may not edit the swapped pair again, so this stays at 3
  expect_equal(osaDist("ca", "abc"), 3L)
  expect_equal(osaDist("", ""), 0L)
})

test_that("hammingDist requires equal-length strings", {
  expect_equal(hammingDist("karolin", "kathrin"), 3L)
  expect_equal(hammingDist("ABCD", "abcd", ignoreCase = TRUE), 0L)
  expect_error(hammingDist("abc", "abcd"), "equal length")
})

test_that("lcsDist counts deletions from both strings", {
  expect_equal(lcsDist("abcde", "ace"), 2L)
  expect_equal(lcsDist("abc", "xyz"), 6L)
  expect_equal(lcsDist("abc", "abc"), 0L)
})

test_that("distances propagate missing values", {
  expect_identical(levenshteinDist(NA_character_, "a"), NA_integer_)
  expect_identical(osaDist("a", NA_character_), NA_integer_)
  expect_identical(hammingDist(NA_character_, "a"), NA_integer_)
  expect_identical(lcsDist("a", NA_character_), NA_integer_)
})

test_that("pair arguments are validated", {
  expect_error(levenshteinDist(1, "a"), "`a` must be a single string")
  expect_error(levenshteinDist("a", c("b", "c")), "`b` must be a single string")
  expect_error(osaDist(c("a", "b"), "c"), "`a` must be a single string")
})

test_that("stringDist recycles and dispatches by method", {
  d <- stringDist(c("kitten", "flaw"), c("sitting", "lawn"))
  expect_equal(d, c(3L, 2L))

  expect_equal(stringDist("ab", c("ba", "ab"), method = "osa"), c(1L, 0L))
  expect_error(stringDist("a", "b", method = "cosine"))
})

test_that("stringSim rescales into the unit interval", {
  expect_equal(stringSim("abc", "abc"), 1)
  expect_equal(stringSim("", ""), 1)
  expect_equal(stringSim("abcd", "abcf"), 0.75)
  expect_equal(stringSim("abc", c("abc", "abd")), c(1, 2 / 3))

  # lcs divides by the combined length of both strings
  expect_equal(stringSim("abc", "xyz", method = "lcs"), 0)
})
