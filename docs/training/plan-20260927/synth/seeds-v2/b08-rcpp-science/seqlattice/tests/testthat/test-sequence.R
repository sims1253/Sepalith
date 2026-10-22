test_that("gc_content counts G and C over called bases", {
  seqs <- c(a = "ACGT", b = "GGCC", c = "ATAT")
  gc <- gc_content(seqs)
  expect_equal(unname(gc), c(0.5, 1, 0))
  expect_named(gc, c("a", "b", "c"))
})

test_that("gc_content ignores N and handles all-N sequences", {
  expect_equal(gc_content("GCNN"), 1)
  expect_true(is.na(gc_content("NNNN")))
  expect_equal(gc_content("NNNN", na_if_empty = FALSE), 0)
})

test_that("gc_skew is signed and bounded", {
  expect_equal(gc_skew("GGGC"), 0.5)
  expect_equal(gc_skew("CCCG"), -0.5)
  expect_true(is.na(gc_skew("ATAT")))
  expect_equal(gc_skew("ATAT", na_if_empty = FALSE), 0)
})

test_that("reverse_complement round-trips and keeps names", {
  seqs <- c(x = "AACGTN", y = "acgg")
  rc <- reverse_complement(seqs)

  expect_equal(unname(rc), c("NACGTT", "CCGT"))
  expect_named(rc, c("x", "y"))
  expect_equal(unname(reverse_complement(rc)), toupper(unname(seqs)))
})

test_that("kmer_table folds canonical k-mers", {
  tab <- kmer_table("AACGTT", k = 2L, canonical = TRUE)
  expect_s3_class(tab, "data.frame")
  expect_equal(sum(tab$count), 5L)
  expect_true(all(tab$kmer <= reverse_complement(tab$kmer)))
})

test_that("kmer_table returns an empty table for short input", {
  tab <- kmer_table(c("AC", "G"), k = 3L)
  expect_equal(nrow(tab), 0L)
  expect_named(tab, c("kmer", "count"))
})

test_that("gc_profile windows cover the sequence", {
  s <- strrep("GCAT", 25)
  prof <- gc_profile(s, width = 20L, step = 10L)

  expect_equal(prof$start[1:3], c(1L, 11L, 21L))
  expect_true(all(prof$end - prof$start == 19L))
  expect_true(all(abs(prof$gc - 0.5) < 1e-12))
})

test_that("input checks fire before compiled code is called", {
  expect_error(gc_content(1:3), "must be a character vector")
  expect_error(gc_content(c("ACGT", NA)), "must not contain missing values")
  expect_error(reverse_complement("ACGU"), "non-ACGTN characters at position\\(s\\) 1")
  expect_error(kmer_table("ACGT", k = 0), "positive whole number")
  expect_error(gc_profile("ACGT", width = 10L), "must be at most 4")
})
