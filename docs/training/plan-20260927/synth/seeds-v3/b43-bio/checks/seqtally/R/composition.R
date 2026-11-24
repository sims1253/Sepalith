near <- function(a, b) isTRUE(all.equal(unname(a), unname(b)))

# composition_stats: every statistic
cs <- composition_stats("AAGGGCTT")
stopifnot(near(cs$gc, 0.5), near(cs$at, 0.5), near(cs$gc_skew, 0.5))
stopifnot(near(cs$at_skew, 0), near(cs$purine, 5 / 8), near(composition_stats("ACGT")$entropy, 2))
stopifnot(identical(names(composition_stats("ACGT", c("gc", "purine"))), c("gc", "purine")))
stopifnot(is.na(composition_stats("NN", "gc")$gc))
stopifnot(near(cs$keto, 5 / 8), near(cs$amino, 3 / 8))

# gc_content: every method arm
stopifnot(near(gc_content("GCAT"), 0.5), near(gc_content("GCATSS", "strict"), 0.5))
stopifnot(near(gc_content("GCATSW", "ambiguous"), 0.5))
stopifnot(near(gc_content("GN", "weighted"), 0.75), is.na(gc_content("NN")))
stopifnot(identical(names(gc_content(c(a = "GG"))), "a"))

# codon_position_gc: every position
cp <- codon_position_gc("GAAGCAGCG")
stopifnot(near(cp[["gc1"]], 1), near(cp[["gc2"]], 2 / 3), near(cp[["gc3"]], 1 / 3))
stopifnot(near(cp[["gc"]], 2 / 3), all(is.na(codon_position_gc("AC"))))
stopifnot(near(codon_position_gc("TGAA", frame = 2)[["gc1"]], 1))
stopifnot(inherits(try(codon_position_gc("ACG", 4), silent = TRUE), "try-error"))

# sliding_composition: every stat arm
sc <- sliding_composition("GGGGAAAA", width = 4L, step = 4L)
stopifnot(nrow(sc) == 2L, near(sc$value, c(1, 0)))
stopifnot(near(sliding_composition("GGGGAAAA", 4L, 4L, "at")$value, c(0, 1)))
stopifnot(near(sliding_composition("GGCCAATT", 4L, 4L, "purine")$value, c(0.5, 0.5)))
stopifnot(near(sliding_composition("ACGTAAAA", 4L, 4L, "entropy")$value, c(2, 0)))
stopifnot(is.na(sliding_composition("NNNNACGT", 4L, 4L)$value[1]))

# compare_composition: mirrored x/y
cc <- compare_composition("AAAA", "AACC")
stopifnot(near(cc$x_freq[["A"]], 1), near(cc$y_freq[["C"]], 0.5))
stopifnot(near(cc$difference[["A"]], -0.5), near(cc$distance, 0.5))
stopifnot(inherits(try(compare_composition("NN", "AC"), silent = TRUE), "try-error"))
stopifnot(inherits(try(compare_composition("AC", "NN"), silent = TRUE), "try-error"))

# composition_table: every column
ct <- composition_table(c(a = "ACGN", b = "GGGG"))
stopifnot(identical(ct$id, c("a", "b")), ct$length == c(4L, 4L))
stopifnot(near(ct$gc, c(2 / 3, 1)), near(ct$at, c(1 / 3, 0)), near(ct$n_frac, c(0.25, 0)))
stopifnot(near(ct$entropy[2], 0), identical(composition_table("AC")$id, "seq1"))
