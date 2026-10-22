#' @importFrom magrittr %>%
NULL

#' Strip version suffixes from feature identifiers
#'
#' Turns `ENSG00000141510.17` into `ENSG00000141510`; identifiers without a
#' version are returned unchanged.
#'
#' @param ids A character vector of feature identifiers.
#' @return A character vector of the same length.
#' @export
strip_id_version <- function(ids) {
  sub("\\.[0-9]+$", "", as.character(ids))
}

#' Check and tidy an annotation table
#'
#' Ensures the required columns exist, strips identifier versions and
#' removes exact duplicate rows.
#'
#' @param annot A data frame of gene annotations.
#' @param id_col Name of the identifier column.
#' @return A tibble with an `id` column.
#' @noRd
.tidy_annotation <- function(annot, id_col) {
  if (!id_col %in% names(annot)) {
    stop(sprintf("annotation has no column '%s'", id_col), call. = FALSE)
  }

  annot %>%
    dplyr::as_tibble() %>%
    dplyr::rename(id = dplyr::all_of(id_col)) %>%
    dplyr::mutate(id = strip_id_version(id)) %>%
    dplyr::distinct()
}

#' Read an annotation table
#'
#' Readers for tab- and comma-separated annotation files. Every column is
#' read as character so identifiers keep their leading zeros.
#'
#' @param path Path to the annotation file.
#' @param id_col Name of the identifier column.
#' @return A tibble with an `id` column.
#' @export
read_annotation_tsv <- function(path, id_col = "gene_id") {
  annot <- readr::read_tsv(path, col_types = readr::cols(.default = "c"), progress = FALSE)
  .tidy_annotation(annot, id_col)
}

#' @rdname read_annotation_tsv
#' @export
read_annotation_csv <- function(path, id_col = "gene_id") {
  annot <- readr::read_csv(path, col_types = readr::cols(.default = "c"), progress = FALSE)
  .tidy_annotation(annot, id_col)
}

#' Read gene records from a GTF file
#'
#' Keeps `gene` records and parses `gene_id`, `gene_name` and `gene_type`
#' from the attribute column.
#'
#' @inheritParams read_annotation_tsv
#' @return A tibble with an `id` column.
#' @export
read_annotation_gtf <- function(path, id_col = "gene_id") {
  gtf_cols <- c(
    "seqname", "source", "feature", "start", "end",
    "score", "strand", "frame", "attributes"
  )
  raw <- readr::read_tsv(
    path,
    col_names = gtf_cols,
    col_types = "cccii-c-c",
    comment = "#",
    progress = FALSE
  )

  # Attributes look like `gene_id "ENSG..."; gene_name "TP53";`. Only a
  # key followed by a quoted value counts as present; anything else is NA.
  pull_attr <- function(attrs, key) {
    pattern <- sprintf('.*%s "([^"]+)".*', key)
    out <- sub(pattern, "\\1", attrs)
    out[is.na(attrs) | !grepl(pattern, attrs)] <- NA_character_
    out
  }

  annot <- raw %>%
    dplyr::filter(feature == "gene") %>%
    dplyr::transmute(
      gene_id = pull_attr(attributes, "gene_id"),
      gene_name = pull_attr(attributes, "gene_name"),
      gene_type = pull_attr(attributes, "gene_type"),
      chrom = seqname,
      start = start,
      end = end,
      strand = strand
    )

  .tidy_annotation(annot, id_col)
}

#' Read an annotation file by extension
#'
#' @param path Path to the annotation file.
#' @param id_col Name of the identifier column.
#' @param format File format; guessed from the extension when `NULL`.
#' @return A tibble with an `id` column.
#' @export
read_annotation <- function(path, id_col = "gene_id", format = NULL) {
  if (is.null(format)) {
    # Look through a trailing .gz so "genes.gtf.gz" is still a GTF.
    ext <- tolower(tools::file_ext(sub("\\.gz$", "", path)))
    format <- switch(ext,
      tsv = ,
      txt = "tsv",
      csv = "csv",
      gtf = "gtf",
      stop(sprintf("cannot guess format from extension '%s'", ext), call. = FALSE)
    )
  }

  switch(format,
    tsv = read_annotation_tsv(path, id_col),
    csv = read_annotation_csv(path, id_col),
    gtf = read_annotation_gtf(path, id_col),
    stop(sprintf("unsupported format '%s'", format), call. = FALSE)
  )
}

#' Collapse duplicated annotation rows
#'
#' Combines rows sharing an identifier, pasting distinct non-missing values
#' of every other column together.
#'
#' @param annot A data frame of gene annotations.
#' @param sep Separator used when pasting distinct values.
#' @return A tibble with one row per `id`.
#' @export
collapse_annotation <- function(annot, sep = ";") {
  dup_ids <- annot$id[duplicated(annot$id)]
  if (length(dup_ids) == 0L) {
    return(annot)
  }

  annot %>%
    dplyr::group_by(id) %>%
    dplyr::summarise(
      dplyr::across(
        dplyr::everything(),
        ~ paste(unique(stats::na.omit(as.character(.x))), collapse = sep)
      ),
      .groups = "drop"
    ) %>%
    dplyr::mutate(dplyr::across(-id, ~ dplyr::na_if(.x, "")))
}

#' Annotate a differential-expression table
#'
#' Left-joins gene annotation onto a result table by stripped identifier and
#' reports how many features could not be matched.
#'
#' @param de A tibble of standardised differential-expression results.
#' @param annot A data frame of gene annotations.
#' @param columns Annotation columns to add; all columns when `NULL`.
#' @param warn_unmatched Warn when some features have no annotation?
#' @return `de` with the requested annotation columns appended.
#' @export
annotate_de <- function(de, annot, columns = NULL, warn_unmatched = TRUE) {
  annot <- collapse_annotation(annot)
  if (!is.null(columns)) {
    annot <- dplyr::select(annot, id, dplyr::all_of(columns))
  }

  # Join on a version-free copy of the ID so "ENSG...17" matches "ENSG...".
  out <- de %>%
    dplyr::mutate(.join_id = strip_id_version(feature)) %>%
    dplyr::left_join(annot, by = c(.join_id = "id")) %>%
    dplyr::select(-.join_id)

  unmatched <- sum(!strip_id_version(de$feature) %in% annot$id)
  if (warn_unmatched && unmatched > 0L) {
    warning(sprintf(
      "%d of %d features have no annotation",
      unmatched, nrow(de)
    ), call. = FALSE)
  }
  out
}
