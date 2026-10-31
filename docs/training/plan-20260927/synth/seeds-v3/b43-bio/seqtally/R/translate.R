#' Genetic code table
#'
#' @param table One of `"standard"`, `"vertebrate_mito"`, `"yeast_mito"`,
#'   `"mold_mito"` or `"bacterial"`.
#' @return A list with elements `code` (a named character vector mapping all
#'   64 codons to one-letter amino acids, `*` for stop), `starts` and `stops`.
#' @export
genetic_code <- function(table = c("standard", "vertebrate_mito", "yeast_mito",
                                   "mold_mito", "bacterial")) {
  table <- match.arg(table)
  bases <- c("T", "C", "A", "G")
  codons <- paste0(rep(bases, each = 16), rep(rep(bases, each = 4), 4), rep(bases, 16))
  standard <- "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
  code <- setNames(strsplit(standard, "", fixed = TRUE)[[1]], codons)

  switch(table,
    standard = {
      changes <- character(0)
      starts <- c("TTG", "CTG", "ATG")
    },
    vertebrate_mito = {
      changes <- c(AGA = "*", AGG = "*", ATA = "M", TGA = "W")
      starts <- c("ATT", "ATC", "ATA", "ATG", "GTG")
    },
    yeast_mito = {
      changes <- c(ATA = "M", CTT = "T", CTC = "T", CTA = "T", CTG = "T", TGA = "W")
      starts <- c("ATA", "ATG", "GTG")
    },
    mold_mito = {
      changes <- c(TGA = "W")
      starts <- c("TTA", "TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG")
    },
    bacterial = {
      changes <- character(0)
      starts <- c("TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG")
    }
  )
  code[names(changes)] <- changes

  list(
    code = code,
    starts = starts,
    stops = names(code)[code == "*"]

  )
}

#' Translate nucleotide sequences
#'
#' @param x A character vector of DNA or RNA sequences.
#' @param table Passed to [genetic_code()].
#' @param frame The reading frame, 1, 2 or 3.
#' @param to_stop Whether to stop at the first stop codon.
#' @param unknown The symbol used for codons with ambiguous bases.
#' @return A character vector of protein sequences.
#' @export
translate_sequence <- function(x, table = "standard", frame = 1L, to_stop = FALSE,
                               unknown = "X") {
  # Validate `frame`
  if (!is.numeric(frame) || length(frame) != 1L || !frame %in% 1:3) {
    stop(sprintf("`%s` must be 1, 2 or 3.", "frame"), call. = FALSE)
  }
  # Validate `to_stop`
  if (!is.logical(to_stop) || length(to_stop) != 1L || is.na(to_stop)) {
    stop(sprintf("`%s` must be `TRUE` or `FALSE`.", "to_stop"), call. = FALSE)
  }
  # Validate `unknown`
  if (!is.character(unknown) || length(unknown) != 1L || nchar(unknown) != 1L) {
    stop(sprintf("`%s` must be a single character.", "unknown"), call. = FALSE)
  }
  code <- genetic_code(table)$code

  out <- character(length(x))

  for (i in seq_along(x)) {
    s <- chartr("U", "T", toupper(x[i]))
    n_codons <- (nchar(s) - frame + 1L) %/% 3L
    if (n_codons < 1L) {
      out[i] <- ""
      next
    }
    starts <- frame + 3L * (seq_len(n_codons) - 1L)
    aa <- unname(code[substring(s, starts, starts + 2L)])
    aa[is.na(aa)] <- unknown
    if (to_stop && any(aa == "*")) {
      aa <- aa[seq_len(match("*", aa) - 1L)]
    }
    out[i] <- paste(aa, collapse = "")
  }
  names(out) <- names(x)

  out
}

#' Translate a sequence in all six reading frames
#'
#' @param x A single DNA sequence.
#' @param table Passed to [genetic_code()].
#' @return A named character vector with elements `"+1"`, `"+2"`, `"+3"`,
#'   `"-1"`, `"-2"` and `"-3"`.
#' @export
six_frame_translate <- function(x, table = "standard") {
  if (!is.character(x) || length(x) != 1L) {
    stop(sprintf("`%s` must be a single string.", "x"), call. = FALSE)
  }
  code <- genetic_code(table)$code
  forward <- toupper(x)
  reverse <- paste(rev(strsplit(chartr("ACGTacgt", "TGCAtgca", x), "")[[1]]), collapse = "")
  reverse <- toupper(reverse)
  out <- setNames(character(6), c("+1", "+2", "+3", "-1", "-2", "-3"))

  # Forward frames
  for (frame in 1:3) {
    n_codons <- (nchar(forward) - frame + 1L) %/% 3L
    starts <- frame + 3L * (seq_len(max(n_codons, 0L)) - 1L)
    aa <- unname(code[substring(forward, starts, starts + 2L)])
    aa[is.na(aa)] <- "X"
    out[[paste0("+", frame)]] <- paste(aa, collapse = "")
  }
  # Reverse frames
  for (frame in 1:3) {
    n_codons <- (nchar(reverse) - frame + 1L) %/% 3L
    starts <- frame + 3L * (seq_len(max(n_codons, 0L)) - 1L)
    aa <- unname(code[substring(reverse, starts, starts + 2L)])
    aa[is.na(aa)] <- "X"
    out[[paste0("-", frame)]] <- paste(aa, collapse = "")
  }

  out
}

#' Codon usage of a coding sequence
#'
#' @param x A character vector of in-frame coding sequences.
#' @param table Passed to [genetic_code()].
#' @param pseudocount Added to every count before computing `rscu`.
#' @return A data frame with one row per codon and columns `codon`, `aa`,
#'   `count`, `frequency`, `per_thousand`, `aa_fraction` and `rscu` (relative
#'   synonymous codon usage).
#' @export
codon_usage <- function(x, table = "standard", pseudocount = 0) {
  code <- genetic_code(table)$code
  s <- chartr("U", "T", toupper(paste(x, collapse = "")))
  n_codons <- nchar(s) %/% 3L
  starts <- 3L * seq_len(n_codons) - 2L
  seen <- substring(s, starts, starts + 2L)
  seen <- seen[seen %in% names(code)]
  total <- length(seen)
  out <- data.frame(codon = names(code), aa = unname(code))

  # count
  out$count <- as.integer(table(factor(seen, levels = out$codon)))
  # frequency
  out$frequency <- if (total > 0L) out$count / total else NA_real_
  # per_thousand
  out$per_thousand <- if (total > 0L) 1000 * out$count / total else NA_real_
  # aa_fraction
  family_total <- ave(out$count, out$aa, FUN = sum)
  out$aa_fraction <- ifelse(family_total > 0, out$count / family_total, NA_real_)
  # rscu
  adjusted <- out$count + pseudocount
  adjusted_total <- ave(adjusted, out$aa, FUN = sum)
  family_size <- ave(adjusted, out$aa, FUN = length)
  out$rscu <- ifelse(adjusted_total > 0, adjusted * family_size / adjusted_total, NA_real_)

  out <- out[order(out$aa, out$codon), , drop = FALSE]
  rownames(out) <- NULL

  out
}

#' Find open reading frames
#'
#' @param x A single DNA sequence.
#' @param min_codons The shortest ORF to report, in codons including the stop.
#' @param strand Search `"both"` strands, only the `"forward"` or only the
#'   `"reverse"` strand.
#' @param table Passed to [genetic_code()].
#' @return A data frame with columns `start`, `end`, `strand`, `frame` and
#'   `codons`, with positions on the forward strand.
#' @export
find_orfs <- function(x, min_codons = 30L, strand = c("both", "forward", "reverse"),
                      table = "standard") {
  strand <- match.arg(strand)
  gc <- genetic_code(table)
  s <- toupper(x)
  n <- nchar(s)
  scan_frame <- function(seq, frame) {
    n_codons <- (n - frame + 1L) %/% 3L
    pos <- frame + 3L * (seq_len(max(n_codons, 0L)) - 1L)
    codons <- substring(seq, pos, pos + 2L)
    hits <- data.frame(from = integer(0), to = integer(0))
    open <- NA_integer_
    for (j in seq_along(codons)) {
      if (is.na(open) && codons[j] == "ATG") {
        open <- j
      } else if (!is.na(open) && codons[j] %in% gc$stops) {
        if (j - open + 1L >= min_codons) {
          hits[nrow(hits) + 1L, ] <- c(pos[open], pos[j] + 2L)
        }
        open <- NA_integer_
      }
    }
    hits
  }
  out <- data.frame(start = integer(0), end = integer(0), strand = character(0),
                    frame = integer(0), codons = integer(0))

  # Forward strand
  if (strand %in% c("both", "forward")) {
    for (frame in 1:3) {
      hits <- scan_frame(s, frame)
      if (nrow(hits) > 0L) {
        out <- rbind(out, data.frame(start = hits$from, end = hits$to, strand = "+",
                                     frame = frame, codons = (hits$to - hits$from + 1L) %/% 3L))
      }
    }
  }
  # Reverse strand
  if (strand %in% c("both", "reverse")) {
    rc <- paste(rev(strsplit(chartr("ACGT", "TGCA", s), "")[[1]]), collapse = "")
    for (frame in 1:3) {
      hits <- scan_frame(rc, frame)
      if (nrow(hits) > 0L) {
        out <- rbind(out, data.frame(start = n - hits$to + 1L, end = n - hits$from + 1L,
                                     strand = "-", frame = frame,
                                     codons = (hits$to - hits$from + 1L) %/% 3L))
      }
    }
  }

  out
}
