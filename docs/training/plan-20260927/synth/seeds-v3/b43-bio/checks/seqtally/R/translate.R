# genetic_code: every table arm
std <- genetic_code("standard")
stopifnot(
  length(std$code) == 64L,
  std$code[["ATG"]] == "M",
  setequal(std$stops, c("TAA", "TAG", "TGA"))
)
vm <- genetic_code("vertebrate_mito")
stopifnot(vm$code[["AGA"]] == "*", vm$code[["TGA"]] == "W", length(vm$stops) == 4L)
stopifnot(
  genetic_code("yeast_mito")$code[["CTG"]] == "T",
  "ATA" %in% genetic_code("yeast_mito")$starts
)
stopifnot(
  genetic_code("mold_mito")$code[["TGA"]] == "W",
  "TTA" %in% genetic_code("mold_mito")$starts
)
stopifnot(
  identical(genetic_code("bacterial")$code, std$code),
  length(genetic_code("bacterial")$starts) == 7L
)

# translate_sequence
stopifnot(translate_sequence("ATGGCCTAAGGG") == "MA*G")
stopifnot(translate_sequence("ATGGCCTAAGGG", to_stop = TRUE) == "MA")
stopifnot(translate_sequence("AUGNNN") == "MX", translate_sequence("CATGAA", frame = 2) == "M")
stopifnot(
  translate_sequence("ATGTGA", table = "vertebrate_mito") == "MW",
  translate_sequence("AT") == ""
)
stopifnot(inherits(try(translate_sequence("ATG", frame = 4), silent = TRUE), "try-error"))
stopifnot(inherits(try(translate_sequence("ATG", to_stop = NA), silent = TRUE), "try-error"))
stopifnot(inherits(try(translate_sequence("ATG", unknown = "XX"), silent = TRUE), "try-error"))

# six_frame_translate: forward and reverse blocks
sf <- six_frame_translate("ATGAAACCC")
stopifnot(identical(names(sf), c("+1", "+2", "+3", "-1", "-2", "-3")))
stopifnot(sf[["+1"]] == "MKP", sf[["-1"]] == "GFH", sf[["+2"]] == "*N", sf[["-3"]] == "VS")

# codon_usage
cu <- codon_usage(c("GCTGCTGCC", "AAA"))
stopifnot(nrow(cu) == 64L, cu$count[cu$codon == "GCT"] == 2L, cu$count[cu$codon == "AAA"] == 1L)
stopifnot(abs(cu$frequency[cu$codon == "GCT"] - 0.5) < 1e-12)
stopifnot(abs(cu$rscu[cu$codon == "GCT"] - 8 / 3) < 1e-12, is.na(cu$rscu[cu$codon == "TGG"]))
stopifnot(cu$per_thousand[cu$codon == "AAA"] == 250, cu$aa_fraction[cu$codon == "AAG"] == 0)
stopifnot(abs(cu$aa_fraction[cu$codon == "GCC"] - 1 / 3) < 1e-12)
stopifnot(codon_usage("ATG", pseudocount = 1)$rscu[1] == 1)

# find_orfs: both strand blocks
fwd <- "CCATGAAATAGCC"
o <- find_orfs(fwd, min_codons = 2L, strand = "forward")
stopifnot(nrow(o) == 1L, o$start == 3L, o$end == 11L, o$strand == "+", o$codons == 3L)
rc <- paste(rev(strsplit(chartr("ACGT", "TGCA", fwd), "")[[1]]), collapse = "")
r <- find_orfs(rc, min_codons = 2L, strand = "reverse")
stopifnot(nrow(r) == 1L, r$start == 3L, r$end == 11L, r$strand == "-")
stopifnot(nrow(find_orfs(fwd, min_codons = 5L)) == 0L, nrow(find_orfs(fwd, 2L)) == 1L)
