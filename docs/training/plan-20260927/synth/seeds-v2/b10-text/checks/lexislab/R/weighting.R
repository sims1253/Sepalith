docs <- list(
  a = c("cat", "sat", "on", "the", "mat"),
  b = c("the", "cat", "ate", "the", "rat"),
  c = c("dogs", "chase", "cats")
)
tdm <- build_tdm(docs)

# build_tdm
stopifnot(identical(dim(tdm), c(10L, 3L)), tdm["the", "b"] == 2L, tdm["cat", "c"] == 0L)
stopifnot(identical(unname(build_tdm(docs, c("the", "cat"))["the", ]), c(1L, 2L, 0L)))
stopifnot(identical(colnames(build_tdm(unname(docs))), c("doc1", "doc2", "doc3")))

# weight_tf
stopifnot(all.equal(unname(colSums(weight_tf(tdm, "proportion"))), c(1, 1, 1)))
stopifnot(all.equal(weight_tf(tdm, "log")["the", "b"], log(3)))

# weight_idf
idf <- weight_idf(tdm)
stopifnot(idf[["dogs"]] > idf[["the"]], all.equal(idf[["the"]], log(4 / 3) + 1))
stopifnot(all.equal(weight_idf(tdm, smooth = FALSE)[["dogs"]], log(3)))

# weight_tfidf
w <- weight_tfidf(tdm)
stopifnot(all.equal(unname(sqrt(colSums(w^2))), c(1, 1, 1)))
stopifnot(all.equal(weight_tfidf(tdm, normalise = FALSE)["dogs", "c"], log(2) + 1))

# weight_bm25
bm <- weight_bm25(tdm)
stopifnot(all(bm >= 0), bm["dogs", "a"] == 0)
stopifnot(inherits(try(weight_bm25(tdm, b = 2), silent = TRUE), "try-error"))
stopifnot(all(weight_bm25(tdm[, 0, drop = FALSE]) == 0))

# doc_cosine
sim <- doc_cosine(w)
stopifnot(isSymmetric(sim), all.equal(unname(diag(sim)), c(1, 1, 1)))
stopifnot(sim["a", "b"] > sim["a", "c"])

# top_terms
top <- top_terms(w, n = 2L)
stopifnot(nrow(top) == 6L, identical(top$rank, rep(1:2, 3)))
stopifnot(all(diff(top$weight[top$doc == "b"]) <= 0))
