# count_bases: every ambiguity arm and both alphabets
cb <- count_bases(c(s1 = "ACGTNR", s2 = "aacc"))
stopifnot(identical(colnames(cb), c("A", "C", "G", "T")), cb["s1", "A"] == 1, cb["s2", "C"] == 2)
cb_other <- count_bases("ACGTNR", ambiguous = "other")
stopifnot(cb_other[1, "other"] == 2)
cb_split <- count_bases("ACGTNR", ambiguous = "split")
stopifnot(abs(cb_split[1, "A"] - 1.75) < 1e-12, abs(sum(cb_split) - 6) < 1e-12)
stopifnot(count_bases("ACGU", alphabet = "rna")[1, "U"] == 1)

# dinucleotide_counts
dn <- dinucleotide_counts("ACGCG")
stopifnot(length(dn) == 16L, dn[["CG"]] == 2, dn[["AC"]] == 1, dn[["GC"]] == 1)
stopifnot(all(dinucleotide_counts("A") == 0))
oe <- dinucleotide_counts("ACGCG", observed_expected = TRUE)
stopifnot(is.na(oe[["TT"]]), oe[["CG"]] > 1)

# run_stats: every element
rs <- run_stats("AAAACGGGTTTTTC")
stopifnot(rs$n_runs == 3L, rs$longest == 5L, rs$longest_base == "T")
stopifnot(abs(rs$mean_length - 4) < 1e-12, abs(rs$covered - 12 / 14) < 1e-12)
rs0 <- run_stats("ACGT")
stopifnot(rs0$n_runs == 0L, is.na(rs0$longest_base), is.na(rs0$mean_length))
stopifnot(inherits(try(run_stats("A", 1), silent = TRUE), "try-error"))

# base_skew
bs <- base_skew("GGGCAAAT", width = 4L)
stopifnot(nrow(bs) == 2L, bs$gc_skew[1] == 0.5, bs$at_skew[2] == 0.5)
stopifnot(is.na(bs$at_skew[1]), is.na(bs$gc_skew[2]))
stopifnot(inherits(try(base_skew("ACG", width = 5L), silent = TRUE), "try-error"))

# positional_counts: every type arm
aln <- c("ACG", "ACT", "AGT", "TCT")
pc <- positional_counts(aln)
stopifnot(pc["A", 1] == 3, pc["T", 3] == 3)
stopifnot(all(abs(colSums(positional_counts(aln, "frequency")) - 1) < 1e-12))
stopifnot(positional_counts(aln, "percent")["A", 1] == 75)
lo <- positional_counts(aln, "log_odds")
stopifnot(lo["A", 1] > 0, lo["G", 1] < 0)

# kmer_counts
kc <- kmer_counts("ACGTACGT", k = 2)
stopifnot(kc[["AC"]] == 2L, kc[["CG"]] == 2L, sum(kc) == 7L)
stopifnot(length(kmer_counts("AC", k = 3)) == 0L, sum(kmer_counts("ACNGT", k = 2)) == 2L)
kcc <- kmer_counts("AAATTT", k = 3, canonical = TRUE)
stopifnot(kcc[["AAA"]] == 2L, !"TTT" %in% names(kcc))
stopifnot(sum(kmer_counts("ACGTACGT", k = 2, step = 2)) == 4L)
