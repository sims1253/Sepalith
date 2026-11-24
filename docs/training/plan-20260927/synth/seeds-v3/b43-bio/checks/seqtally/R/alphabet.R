# alphabet_symbols: every arm
stopifnot(identical(alphabet_symbols("dna")$symbols, c("A", "C", "G", "T")))
stopifnot(identical(alphabet_symbols("rna")$symbols, c("A", "C", "G", "U")))
stopifnot(length(alphabet_symbols("iupac_dna")$symbols) == 15L)
stopifnot("U" %in% alphabet_symbols("iupac_rna")$symbols)
stopifnot(alphabet_symbols("protein")$wildcard == "X", !alphabet_symbols("protein")$nucleic)

# check_sequence_args: every validated argument
stopifnot(isTRUE(check_sequence_args("ACGTACGT", 1, 8, 4, 2)))
err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
stopifnot(grepl("`seq`", err(check_sequence_args(NA_character_, 1, 2, 1, 1))))
stopifnot(grepl("`start` must lie", err(check_sequence_args("ACGT", 0, 2, 1, 1))))
stopifnot(grepl("`end` must lie", err(check_sequence_args("ACGT", 1, 9, 1, 1))))
stopifnot(grepl("greater", err(check_sequence_args("ACGT", 3, 2, 1, 1))))
stopifnot(grepl("`width` must be", err(check_sequence_args("ACGT", 1, 2, "a", 1))))
stopifnot(grepl("`step` must lie", err(check_sequence_args("ACGT", 1, 2, 1, 7))))

# validate_sequence
v <- validate_sequence(c(a = "ACGT", b = "acg-t", c = "ACXT", d = NA))
stopifnot(identical(as.vector(v), c(TRUE, TRUE, FALSE, FALSE)))
stopifnot(identical(attr(v, "invalid")$c, "X"))
stopifnot(!validate_sequence("acgt", allow_lower = FALSE))
stopifnot(!validate_sequence("AC-T", allow_gaps = FALSE))
stopifnot(grepl("allow_gaps", err(validate_sequence("A", allow_gaps = NA))))

# sequence_issues: every column
si <- sequence_issues(c(s1 = "ACgN-Z", s2 = ""))
stopifnot(si$length == c(6L, 0L), si$empty == c(FALSE, TRUE))
stopifnot(si$invalid[1] == 2L, si$gap[1] == 1L, si$lower[1] == 1L, si$wildcard[1] == 1L)
stopifnot(sequence_issues("ACNN", "iupac_dna")$wildcard == 2L)

# clean_sequence: every arm
stopifnot(clean_sequence("acXgt", action = "drop") == "ACGT")
stopifnot(clean_sequence("acXgt", action = "mask") == "ACNGT")
w <- tryCatch(clean_sequence("AXT", action = "warn"), warning = conditionMessage)
stopifnot(grepl("masked", w))
stopifnot(grepl("invalid symbol", err(clean_sequence("AXT", action = "error"))))
stopifnot(clean_sequence("ACGT", action = "error") == "ACGT")

# detect_alphabet
stopifnot(detect_alphabet("ACGT") == "dna", detect_alphabet("ACGU") == "rna")
stopifnot(detect_alphabet("ACGTRY") == "iupac_dna", detect_alphabet("MKLVQE") == "protein")
stopifnot(is.na(suppressMessages(detect_alphabet("AC1"))), is.na(detect_alphabet("")))
