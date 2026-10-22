toks <- list(d1 = c("a", "b", "c", "a", "b"), d2 = "z")

# ngrams_one / make_ngrams
stopifnot(identical(ngrams_one(c("a", "b", "c"), 2L, "_"), c("a_b", "b_c")))
stopifnot(identical(make_ngrams(toks, 3L)$d1, c("a b c", "b c a", "c a b")))
stopifnot(identical(make_ngrams(toks)$d2, character()))

# make_ngram_range
stopifnot(identical(make_ngram_range(list(c("a", "b")), 1L, 2L)[[1]], c("a", "b", "a b")))
stopifnot(inherits(try(make_ngram_range(toks, 0L, 2L), silent = TRUE), "try-error"))

# make_skipgrams
stopifnot(identical(make_skipgrams(list(c("a", "b", "c")), k = 1L)[[1]],
                    c("a b", "a c", "b c")))
stopifnot(identical(make_skipgrams(list(c("a", "b", "c")), k = 0L)[[1]], c("a b", "b c")))

# count_ngrams
cn <- count_ngrams(make_ngrams(toks))
stopifnot(identical(cn$ngram[1], "a b"), cn$count[1] == 2L)
stopifnot(nrow(count_ngrams(make_ngrams(toks), min_count = 5L)) == 0)
stopifnot(identical(names(count_ngrams(make_ngrams(toks), min_count = 5L)), c("ngram", "count")))

# bigram_pmi
pm <- bigram_pmi(toks, min_count = 2L)
stopifnot(nrow(pm) == 1, pm$first == "a", pm$second == "b")
stopifnot(all.equal(pm$pmi, log2((2 / 4) / ((2 / 6) * (2 / 6)))))
