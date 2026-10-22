# normalise_text
stopifnot(identical(normalise_text("  Hello\tWORLD \n"), "hello world"))
stopifnot(identical(normalise_text("a1 b22", strip_digits = TRUE), "a b"))
stopifnot(is.na(normalise_text(NA_character_)))

# tokenise_words
tw <- tokenise_words(c(x = "It's a dog-eat-dog world!", y = NA))
stopifnot(identical(tw$x, c("it's", "a", "dog", "eat", "dog", "world")), length(tw$y) == 0)
stopifnot(identical(tokenise_words("a bb ccc", min_length = 2L)[[1]], c("bb", "ccc")))

# tokenise_sentences
ts <- tokenise_sentences(c("One here. Two there! three? Four.", NA))
stopifnot(identical(ts[[1]], c("One here.", "Two there! three?", "Four.")), length(ts[[2]]) == 0)

# tokenise_shingles
stopifnot(identical(tokenise_shingles("Ab c", k = 3L)[[1]], c("ab_", "b_c")))
stopifnot(length(tokenise_shingles("ab", k = 3L)[[1]]) == 0)
stopifnot(inherits(try(tokenise_shingles("ab", k = 0L), silent = TRUE), "try-error"))

# tokenise
stopifnot(identical(tokenise("A b", "words")[[1]], c("a", "b")))
stopifnot(identical(tokenise("abcd", "shingles", k = 4L)[[1]], "abcd"))

# count_tokens / tokens_to_long
stopifnot(identical(count_tokens(list(a = c("x", "y"), b = character())), c(a = 2L, b = 0L)))
long <- tokens_to_long(list(a = c("x", "y"), b = "z"))
stopifnot(identical(long$doc_id, c("a", "a", "b")), identical(long$position, c(1L, 2L, 1L)))
stopifnot(identical(tokens_to_long(list(character()))$token, character()))
