# skipped: needs dplyr (.tidy_annotation, collapse_annotation, annotate_de),
# readr (read_annotation_tsv, read_annotation_csv, read_annotation_gtf,
# read_annotation)
stopifnot(
  identical(strip_id_version(c("ENSG00000141510.17", "ENSG1")), c("ENSG00000141510", "ENSG1")),
  identical(strip_id_version(factor("TP53.2")), "TP53"),
  identical(strip_id_version(character()), character())
)
