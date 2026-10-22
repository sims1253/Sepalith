# stopword_list
stopifnot(identical(stopword_list("minimal"), c("a", "an", "and", "of", "the", "to")))
stopifnot(!is.unsorted(stopword_list("de")), "the" %in% stopword_list())

# remove_stopwords / keep_vocabulary
toks <- list(d1 = c("The", "cat", "and", "hat"), d2 = character())
stopifnot(identical(remove_stopwords(toks)$d1, c("cat", "hat")))
stopifnot(identical(remove_stopwords(toks, ignore_case = FALSE)$d1, c("The", "cat", "hat")))
stopifnot(identical(keep_vocabulary(toks, c("CAT"))$d1, "cat"))
stopifnot(length(keep_vocabulary(toks, "x")$d2) == 0)

# corpus_stopwords
corp <- list(c("x", "y"), c("x", "z"), c("x"), c("x", "y"), c("w"))
stopifnot(identical(corpus_stopwords(corp, max_doc_prop = 0.6), "x"))
stopifnot(identical(suppressWarnings(corpus_stopwords(corp[1:2])), character()))

# light_stem
stopifnot(identical(light_stem(c("happiness", "running", "cats", "is", NA)),
                    c("happi", "runn", "cat", "is", NA)))

# map_tokens
stopifnot(identical(map_tokens(list(a = c("x", "y")), toupper), list(a = c("X", "Y"))))
