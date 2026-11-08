# convert.coords: every from arm and every to arm
stopifnot(identical(convert.coords(0L, 10L, "bed", "gff"), list(start = 1L, end = 10L)))
stopifnot(identical(convert.coords(0L, 10L, "psl", "vcf"), list(start = 1L, end = 10L)))
stopifnot(identical(convert.coords(0L, 9L, "bam", "gff"), list(start = 1L, end = 10L)))
stopifnot(identical(convert.coords(1L, 10L, "gff", "bed"), list(start = 0L, end = 10L)))
stopifnot(identical(convert.coords(1L, 10L, "vcf", "bam"), list(start = 0L, end = 9L)))
stopifnot(identical(convert.coords(5L, 8L, "bed", "psl"), list(start = 5L, end = 8L)))
stopifnot(inherits(try(convert.coords(1:2, 3L, "bed", "gff"), silent = TRUE), "try-error"))

# parse.region: every piece
pr <- parse.region(c("chr1:1,000-2,000", "chrX:5-9(-)"))
stopifnot(identical(pr$chrom, c("chr1", "chrX")), pr$start == c(1000L, 5L), pr$end == c(2000L, 9L))
stopifnot(identical(pr$strand, c("*", "-")))
stopifnot(inherits(try(parse.region("chr1-5"), silent = TRUE), "try-error"))
stopifnot(inherits(try(parse.region("chr1:9-5"), silent = TRUE), "try-error"))

# region.string: every style
iv <- data.frame(chrom = "chr2", start = 1500L, end = 2500L, strand = "-", name = "p1")
stopifnot(region.string(iv) == "chr2:1500-2500", region.string(iv, "ensembl") == "2:1500-2500:-1")
stopifnot(region.string(iv, "igv") == "chr2:1,500-2,500")
stopifnot(region.string(iv, "bed") == "chr2\t1499\t2500")
stopifnot(region.string(iv, "genbank") == "complement(1500..2500)")
stopifnot(
  region.string(iv, "stranded") == "chr2:1500-2500(-)",
  length(region.string(iv[0, ])) == 0L
)

# read.interval.lines: every format arm
bed <- read.interval.lines(c("track name=x", "chr1\t0\t10\tf1\t0\t+"), "bed")
stopifnot(bed$start == 1L, bed$end == 10L, bed$strand == "+", bed$name == "f1")
gff <- read.interval.lines("chr1\tsrc\tgene\t5\t20\t.\t-\t.\tID=g1;Name=x", "gff")
stopifnot(gff$start == 5L, gff$end == 20L, gff$strand == "-", gff$name == "g1")
vcf <- read.interval.lines(c("#CHROM", "chr1\t100\trs1\tACG\tA"), "vcf")
stopifnot(vcf$start == 100L, vcf$end == 102L, vcf$strand == "*", vcf$name == "rs1")
np <- read.interval.lines("chr1\t9\t20\tpk\t0\t.\t5\t-1\t-1\t3", "narrowpeak")
stopifnot(np$start == 10L, np$strand == "*", read.interval.lines("c\t0\t5", "bed")$strand == "*")

# write.interval.lines: every format arm round-trips
for (f in c("bed", "gff", "narrowpeak")) {
  back <- read.interval.lines(write.interval.lines(iv, f), f)
  stopifnot(back$start == iv$start, back$end == iv$end, back$strand == "-", back$name == "p1")
}
stopifnot(inherits(try(write.interval.lines(iv[, 1:3]), silent = TRUE), "try-error"))
stopifnot(
  inherits(try(write.interval.lines(transform(iv, start = 0L)), silent = TRUE), "try-error")
)
stopifnot(inherits(try(write.interval.lines(transform(iv, end = NA)), silent = TRUE), "try-error"))
back <- read.interval.lines(write.interval.lines(iv, "vcf"), "vcf")
stopifnot(back$start == iv$start, back$end == iv$end, back$name == "p1")

# bin.intervals: first and last bin
bi <- bin.intervals(data.frame(chrom = c("a", "b"), start = c(950L, 10L), end = c(2100L, 20L)))
stopifnot(bi$index == c(1L, 1L, 1L, 2L), bi$bin == c(1L, 2L, 3L, 1L))
stopifnot(bi$overlap == c(51L, 1000L, 100L, 11L), bi$bin.end[3] == 3000L)
stopifnot(inherits(try(bin.intervals(iv, 0), silent = TRUE), "try-error"))
