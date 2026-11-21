fa = c(">s1 first one", "ACGT", "AC", ">s2", "GG")
fq = c("@r1 lane1", "ACGT", "+", "IIII", "@r2", "GG", "+", "#5")
gb = c("LOCUS       AB001 6 bp DNA", "DEFINITION  test locus", "ORIGIN",
       "        1 acgtac", "//")
em = c("ID   EM01; DNA; 4 BP.", "DE   embl entry", "SQ   Sequence", "     ggcc 4", "//")

# splitRecords: every format arm
stopifnot(length(splitRecords(fa, "fasta")) == 2L, length(splitRecords(fq, "fastq")) == 2L)
stopifnot(length(splitRecords(c(gb, gb), "genbank")) == 2L, length(splitRecords(em, "embl")) == 1L)
stopifnot(length(splitRecords(character(0))) == 0L)
stopifnot(inherits(try(splitRecords(fq[1:3], "fastq"), silent = TRUE), "try-error"))
stopifnot(inherits(try(splitRecords(gb[1:4], "genbank"), silent = TRUE), "try-error"))

# parseRecord: every format arm
r1 = parseRecord(splitRecords(fa, "fasta")[[1]], "fasta")
stopifnot(r1$id == "s1", r1$description == "first one", r1$sequence == "ACGTAC", is.na(r1$quality))
r2 = parseRecord(fq[1:4], "fastq")
stopifnot(r2$id == "r1", r2$description == "lane1", r2$quality == "IIII")
r3 = parseRecord(gb, "genbank")
stopifnot(r3$id == "AB001", r3$description == "test locus", r3$sequence == "ACGTAC")
r4 = parseRecord(em, "embl")
stopifnot(r4$id == "EM01", r4$description == "embl entry", r4$sequence == "GGCC")

# formatRecord: every arm round-trips through parseRecord
for (f in c("fasta", "fastq", "genbank", "embl")) {
  rec = if (f == "fastq") r2 else r3
  back = parseRecord(formatRecord(rec, f, width = 4L), f)
  stopifnot(identical(back$sequence, rec$sequence), identical(back$id, rec$id))
}
stopifnot(formatRecord(r1, "fasta", width = 4L)[1] == ">s1 first one")
stopifnot(length(formatRecord(r1, "fasta", width = 4L)) == 3L)

# validateRecord: every field
stopifnot(length(validateRecord(r1, "fasta")) == 0L, length(validateRecord(r2, "fastq")) == 0L)
stopifnot(grepl("`id`", validateRecord(modifyList(r1, list(id = "")), "fasta")))
stopifnot(grepl("`description`", validateRecord(modifyList(r1, list(description = 1)), "fasta")))
stopifnot(grepl("`sequence`", validateRecord(modifyList(r1, list(sequence = "")), "fasta")))
stopifnot(grepl("match", validateRecord(modifyList(r2, list(quality = "II")), "fastq")))
stopifnot(
  grepl("`NA`", validateRecord(r2, "embl")),
  grepl("missing", validateRecord(list(id = "a"))[1])
)

# readRecords: every output shape
tmp = tempfile(fileext = ".fa")
writeLines(fa, tmp)
stopifnot(identical(names(readRecords(tmp, "fasta")), c("s1", "s2")))
df = readRecords(tmp, "fasta", as = "data.frame")
stopifnot(nrow(df) == 2L, df$sequence[2] == "GG")
stopifnot(identical(readRecords(tmp, "fasta", as = "named"), c(s1 = "ACGTAC", s2 = "GG")))
stopifnot(inherits(try(readRecords(tempfile(), "fasta"), silent = TRUE), "try-error"))
unlink(tmp)
