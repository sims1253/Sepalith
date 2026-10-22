docs <- list(
  a = c("cat", "sat", "on", "the", "mat"),
  b = c("the", "cat", "ate", "the", "rat"),
  c = c("dogs", "chase", "cats")
)

test_that("build_tdm counts terms per document", {
  tdm <- build_tdm(docs)
  expect_equal(dim(tdm), c(10L, 3L))
  expect_equal(colnames(tdm), c("a", "b", "c"))

  expect_equal(tdm["the", "b"], 2L)
  expect_equal(tdm["cat", "c"], 0L)
  expect_equal(colSums(tdm), c(a = 5, b = 5, c = 3))
})

test_that("build_tdm respects a fixed vocabulary", {
  tdm <- build_tdm(docs, vocabulary = c("the", "cat"))
  expect_equal(rownames(tdm), c("the", "cat"))
  expect_equal(unname(tdm["the", ]), c(1L, 2L, 0L))
  expect_equal(unname(tdm["cat", ]), c(1L, 1L, 0L))
})

test_that("weight_tf schemes scale counts as documented", {
  tdm <- build_tdm(docs)
  expect_equal(weight_tf(tdm, "raw"), tdm * 1)
  expect_equal(unname(colSums(weight_tf(tdm, "proportion"))), c(1, 1, 1))
  expect_equal(weight_tf(tdm, "binary")["the", "b"], 1)
  # log scheme is log1p, so two occurrences give log(3)
  expect_equal(weight_tf(tdm, "log")["the", "b"], log(3))
  expect_error(weight_tf(tdm, "sqrt"))
})

test_that("weight_idf gives rare terms higher weight", {
  tdm <- build_tdm(docs)
  idf <- weight_idf(tdm)
  expect_named(idf, rownames(tdm))
  expect_gt(idf[["dogs"]], idf[["the"]])

  expect_equal(idf[["the"]], log(4 / 3) + 1)
  expect_equal(weight_idf(tdm, smooth = FALSE)[["dogs"]], log(3))
})

test_that("weight_tfidf normalises documents to unit length", {
  tdm <- build_tdm(docs)
  w <- weight_tfidf(tdm)
  expect_equal(unname(sqrt(colSums(w^2))), c(1, 1, 1))

  # "dogs" occurs once, in one of three documents
  raw <- weight_tfidf(tdm, normalise = FALSE)
  expect_equal(raw["dogs", "c"], log(2) + 1)
})

test_that("weight_bm25 rejects b outside the unit interval", {
  tdm <- build_tdm(docs)
  expect_error(weight_bm25(tdm, b = 1.5), "between 0 and 1")
  expect_error(weight_bm25(tdm, b = -0.1), "between 0 and 1")

  w <- weight_bm25(tdm)
  expect_true(all(w >= 0))
  expect_equal(w["dogs", "a"], 0)
})

test_that("doc_cosine is symmetric with a unit diagonal", {
  sim <- doc_cosine(weight_tfidf(build_tdm(docs)))
  expect_equal(sim, t(sim))
  expect_equal(unname(diag(sim)), c(1, 1, 1))
  expect_gt(sim["a", "b"], sim["a", "c"])
})

test_that("top_terms returns ranked rows per document", {
  top <- top_terms(weight_tfidf(build_tdm(docs)), n = 2L)
  expect_s3_class(top, "data.frame")
  expect_named(top, c("doc", "rank", "term", "weight"))
  expect_equal(nrow(top), 6L)

  expect_equal(top$rank, rep(1:2, 3))
  expect_true(all(diff(top$weight[top$doc == "b"]) <= 0))
})
