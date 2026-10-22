#' Parse FASTA lines into named sequences
#'
#' Header lines start with `>`; everything up to the first whitespace is
#' taken as the sequence name. Sequence lines are concatenated.
#'
#' @param lines A character vector of lines, as returned by [readLines()].
#' @return A named character vector of sequences.
#' @keywords internal
parse_fasta <- function(lines) {
  lines <- trimws(lines)
  # Drop blank lines and old-style ';' comment lines
  lines <- lines[nzchar(lines) & !startsWith(lines, ";")]

  is_header <- startsWith(lines, ">")
  if (length(lines) > 0L && !is_header[1]) {
    stop("FASTA input must start with a '>' header line.", call. = FALSE)
  }

  record <- cumsum(is_header)
  headers <- sub("^>\\s*", "", lines[is_header])
  ids <- sub("\\s.*$", "", headers)
  seqs <- vapply(split(lines[!is_header], record[!is_header]), paste, character(1),
    collapse = "")

  # Records with a header but no sequence lines stay as empty strings
  out <- character(length(ids))
  out[as.integer(names(seqs))] <- seqs
  names(out) <- ids
  return(out)
}

#' Format named sequences as FASTA lines
#'
#' @param seqs A named character vector of sequences.
#' @param line_width Number of bases per sequence line.
#' @return A character vector of FASTA lines.
#' @keywords internal
format_fasta <- function(seqs, line_width = 60L) {
  if (is.null(names(seqs)) || any(!nzchar(names(seqs)))) {
    stop("Every sequence needs a non-empty name.", call. = FALSE)
  }

  out <- vector("list", length(seqs))
  for (i in seq_along(seqs)) {
    len <- nchar(seqs[[i]])
    if (len == 0L) {
      # An empty record is just its header, without a blank sequence line
      body <- character(0)
    } else {
      starts <- seq.int(1L, len, by = line_width)
      body <- substring(seqs[[i]], starts, pmin(starts + line_width - 1L, len))
    }
    out[[i]] <- c(paste0(">", names(seqs)[i]), body)
  }

  return(unlist(out, use.names = FALSE))
}

#' Read a FASTA file
#'
#' @param path Path to a FASTA file, optionally gzip-compressed.
#' @param validate Logical; check that all sequences use the ACGTN alphabet?
#' @return A named character vector of upper-case sequences.
#' @export
#' @seealso [write_fasta()]
read_fasta <- function(path, validate = TRUE) {
  if (!file.exists(path)) {
    stop(sprintf("File '%s' does not exist.", path), call. = FALSE)
  }

  con <- if (grepl("\\.gz$", path)) gzfile(path, "r") else file(path, "r")
  on.exit(close(con), add = TRUE)
  seqs <- toupper(parse_fasta(readLines(con, warn = FALSE)))

  if (validate) {
    check_dna(unname(seqs), arg = basename(path))
  }
  return(seqs)
}

#' Write a FASTA file
#'
#' @param seqs A named character vector of sequences.
#' @param path Path to a FASTA file, optionally gzip-compressed.
#' @param line_width Number of bases per sequence line.
#' @return `path`, invisibly.
#' @export
#' @seealso [read_fasta()]
write_fasta <- function(seqs, path, line_width = 60L) {
  line_width <- check_count(line_width, "line_width")
  con <- if (grepl("\\.gz$", path)) gzfile(path, "w") else file(path, "w")
  on.exit(close(con), add = TRUE)

  writeLines(format_fasta(seqs, line_width), con)
  return(invisible(path))
}

#' Standard genetic code
#'
#' @return A named character vector mapping the 64 codons to one-letter amino
#'   acid codes, with `*` for stop codons.
#' @noRd
codon_table <- function() {
  # Codons in the classic TCAG table order, read row by row
  bases <- c("T", "C", "A", "G")
  codons <- paste0(rep(bases, each = 16), rep(rep(bases, each = 4), 4), rep(bases, 16))
  aa <- strsplit(paste0(
    "FFLLSSSSYY**CC*W",
    "LLLLPPPPHHQQRRRR",
    "IIIMTTTTNNKKSSRR",
    "VVVVAAAADDEEGGGG"
  ), "")[[1]]

  return(stats::setNames(aa, codons))
}

#' Translate DNA sequences to protein
#'
#' Translates each sequence in the given reading frame using the standard
#' genetic code. Codons containing `N` become `X`.
#'
#' @param seqs A character vector of DNA sequences.
#' @param frame Reading frame, one of 1, 2 or 3.
#' @param to_stop Logical; truncate each protein at the first stop codon?
#' @return A character vector of protein sequences, named like `seqs`.
#' @export
translate_dna <- function(seqs, frame = 1L, to_stop = FALSE) {
  seqs <- check_dna(seqs)
  if (!frame %in% 1:3) {
    stop("`frame` must be 1, 2 or 3.", call. = FALSE)
  }

  code <- codon_table()
  out <- character(length(seqs))
  for (i in seq_along(seqs)) {
    s <- substring(seqs[i], frame)
    n_codons <- nchar(s) %/% 3L
    starts <- seq_len(n_codons) * 3L - 2L
    codons <- substring(s, starts, starts + 2L)

    # Any codon with an N has no table entry and becomes X
    aa <- unname(code[codons])
    aa[is.na(aa)] <- "X"
    if (to_stop && any(aa == "*")) {
      aa <- aa[seq_len(match("*", aa) - 1L)]
    }
    out[i] <- paste(aa, collapse = "")
  }

  names(out) <- names(seqs)
  return(out)
}
