# samFlags: every consistency check
f = samFlags(c(99L, 4L, 16L))
stopifnot(ncol(f) == 12L, f$paired[1], f$properPair[1], f$mateReverse[1], f$first[1])
stopifnot(f$unmapped[2], f$reverse[3], length(attr(f, "problems")) == 0L)
p = attr(samFlags(c(2L, 8L, 32L, 64L, 128L)), "problems")
stopifnot(length(p) == 5L, grepl("properPair", p[1]), grepl("mateUnmapped", p[2]))
stopifnot(grepl("mateReverse", p[3]), grepl("first", p[4]), grepl("second", p[5]))
stopifnot(inherits(try(samFlags(5000L), silent = TRUE), "try-error"))

# cigarLength: every arm
cg = "3H2S10M2I3D5M1S"
stopifnot(
  cigarLength(cg) == 18L,
  cigarLength(cg, "query") == 20L,
  cigarLength(cg, "aligned") == 15L
)
stopifnot(cigarLength(cg, "softClip") == 3L, cigarLength(cg, "hardClip") == 3L)
stopifnot(cigarLength(cg, "insertion") == 2L, cigarLength(cg, "deletion") == 3L)
stopifnot(is.na(cigarLength("*")), inherits(try(cigarLength("10Q"), silent = TRUE), "try-error"))

# parseSamLine: every converted field
ln = paste("r1", "99", "chr1", "100", "60", "10M", "=", "200", "110", "ACGTACGTAC",
           "IIIIIIIIII", "NM:i:0", sep = "\t")
s = parseSamLine(ln)
stopifnot(s$flag == 99L, s$pos == 100L, s$mapq == 60L, s$pnext == 200L, s$tlen == 110L)
stopifnot(s$rnext == "chr1", identical(s$tags, "NM:i:0"))
for (k in c(2, 4, 5, 8, 9)) {
  bad = strsplit(ln, "\t")[[1]]
  bad[k] = "x"
  stopifnot(inherits(try(parseSamLine(paste(bad, collapse = "\t")), silent = TRUE), "try-error"))
}
stopifnot(inherits(try(parseSamLine("a\tb"), silent = TRUE), "try-error"))

# parseSamTags: every type arm
tg = parseSamTags(c("XA:A:x", "NM:i:3", "AS:f:1.5", "RG:Z:grp 1", "XH:H:1AFF", "XB:B:i,1,2",
                    "XF:B:f,0.5"))
stopifnot(tg$XA == "x", tg$NM == 3L, tg$AS == 1.5, tg$RG == "grp 1")
stopifnot(identical(tg$XH, as.raw(c(26, 255))), identical(tg$XB, 1:2), tg$XF == 0.5)
stopifnot(inherits(try(parseSamTags("bad"), silent = TRUE), "try-error"))

# formatSamTags: every type arm, round trip
types = c("A", "i", "f", "Z", "H", "B", "B")
back = formatSamTags(tg, types)
stopifnot(identical(back, c("XA:A:x", "NM:i:3", "AS:f:1.5", "RG:Z:grp 1", "XH:H:1AFF",
                            "XB:B:i,1,2", "XF:B:f,0.5")))
stopifnot(inherits(try(formatSamTags(list(a = 1), "Q"), silent = TRUE), "try-error"))
stopifnot(inherits(try(formatSamTags(list(a = 1), c("i", "i")), silent = TRUE), "try-error"))

# alignmentSpan: leading and trailing clips
sp = alignmentSpan(100, "3H2S10M2I3D5M1S")
stopifnot(sp$refStart == 100L, sp$refEnd == 117L, sp$clipLeft == 5L, sp$clipRight == 1L)
sp2 = alignmentSpan(5, "8M")
stopifnot(sp2$clipLeft == 0L, sp2$clipRight == 0L, sp2$refEnd == 12L)
stopifnot(inherits(try(alignmentSpan(1, "abc"), silent = TRUE), "try-error"))
