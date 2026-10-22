# toChars / checkPair
stopifnot(identical(toChars("abc"), c("a", "b", "c")), length(toChars("")) == 0)
stopifnot(isTRUE(checkPair("a", "b")))
stopifnot(inherits(try(checkPair(1, "b"), silent = TRUE), "try-error"))

# levenshteinDist
stopifnot(levenshteinDist("kitten", "sitting") == 3L, levenshteinDist("", "abc") == 3L)
stopifnot(levenshteinDist("ABC", "abc", ignoreCase = TRUE) == 0L)
stopifnot(identical(levenshteinDist(NA_character_, "a"), NA_integer_))

# osaDist
stopifnot(osaDist("ab", "ba") == 1L, osaDist("ca", "abc") == 3L, osaDist("", "") == 0L)

# hammingDist
stopifnot(hammingDist("karolin", "kathrin") == 3L)
stopifnot(inherits(try(hammingDist("abc", "abcd"), silent = TRUE), "try-error"))

# lcsDist
stopifnot(lcsDist("abcde", "ace") == 2L, lcsDist("abc", "xyz") == 6L)

# stringDist
stopifnot(identical(stringDist(c("kitten", "flaw"), c("sitting", "lawn")), c(3L, 2L)))
stopifnot(identical(stringDist("ab", c("ba", "ab"), method = "osa"), c(1L, 0L)))

# stringSim
stopifnot(stringSim("", "") == 1, all.equal(stringSim("abcd", "abcf"), 0.75))
stopifnot(all.equal(stringSim("abc", c("abc", "abd")), c(1, 2 / 3)))
stopifnot(stringSim("abc", "xyz", method = "lcs") == 0)
