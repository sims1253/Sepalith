# qualityEncoding: every arm
stopifnot(qualityEncoding("sanger")$offset == 33L, qualityEncoding("sanger")$maxScore == 93L)
stopifnot(qualityEncoding("solexa")$minScore == -5L, qualityEncoding("solexa")$scale == "solexa")
stopifnot(
  qualityEncoding("illumina13")$offset == 64L,
  qualityEncoding("illumina15")$minScore == 2L
)
stopifnot(qualityEncoding("illumina18")$maxScore == 41L)

# decodeQuality: both bounds
stopifnot(identical(decodeQuality("!+5I")[[1]], c(0L, 10L, 20L, 40L)))
stopifnot(identical(decodeQuality(c(a = "@h"), "illumina13")$a, c(0L, 40L)))
stopifnot(inherits(try(decodeQuality("@", "illumina15"), silent = TRUE), "try-error"))
stopifnot(inherits(try(decodeQuality("K", "illumina18"), silent = TRUE), "try-error"))
stopifnot(inherits(try(decodeQuality(1), silent = TRUE), "try-error"))

# encodeQuality: both bounds
stopifnot(
  encodeQuality(c(0, 10, 20, 40)) == "!+5I",
  encodeQuality(list(c(0, 40)), "illumina13") == "@h"
)
stopifnot(inherits(try(encodeQuality(-1), silent = TRUE), "try-error"))
stopifnot(inherits(try(encodeQuality(50, "illumina18"), silent = TRUE), "try-error"))

# convertQuality: phred-phred, solexa-phred, phred-solexa, clamping
stopifnot(convertQuality("@h", "illumina13", "sanger") == "!I")
stopifnot(convertQuality(c(x = "K"), "sanger", "illumina18") == c(x = "J"))
stopifnot(decodeQuality(convertQuality(";", "solexa", "sanger"))[[1]] == 1L)
stopifnot(decodeQuality(convertQuality("!", "sanger", "solexa"), "solexa")[[1]] == -5L)
stopifnot(decodeQuality(convertQuality("5", "sanger", "solexa"), "solexa")[[1]] == 20L)

# qualitySummary: every element
qs = qualitySummary(c("!+5?", ""))
stopifnot(qs$meanQ[1] == 15, qs$medianQ[1] == 15, qs$minQ[1] == 0, qs$maxQ[1] == 30)
stopifnot(qs$q20[1] == 0.5, qs$q30[1] == 0.25, abs(qs$expectedErrors[1] - 1.111) < 1e-12)
stopifnot(all(is.na(c(qs$meanQ[2], qs$q30[2], qs$expectedErrors[2]))))

# trimQuality: both sides
tq = trimQuality("ACGTAC", "#5II5#")
stopifnot(tq$sequence == "CGTA", tq$quality == "5II5", tq$left == 1L, tq$right == 1L)
stopifnot(trimQuality("ACGTAC", "#5II5#", side = "left")$right == 0L)
stopifnot(trimQuality("ACGTAC", "#5II5#", side = "right")$sequence == "ACGTA")
stopifnot(
  trimQuality("AC", "##")$sequence == "",
  inherits(try(trimQuality("A", "II"), silent = TRUE), "try-error")
)
