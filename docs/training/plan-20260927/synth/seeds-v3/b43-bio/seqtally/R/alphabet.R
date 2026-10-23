#' Symbol set for a sequence alphabet
#'
#' @param alphabet One of `"dna"`, `"rna"`, `"iupac_dna"`, `"iupac_rna"` or `"protein"`.
#' @return A list with elements `name`, `symbols`, `gap`, `wildcard` and `nucleic`.
#' @export
alphabet_symbols <- function(alphabet = c("dna", "rna", "iupac_dna", "iupac_rna", "protein")) {
  alphabet <- match.arg(alphabet)
  core_dna <- c("A", "C", "G", "T")
  core_rna <- c("A", "C", "G", "U")
  iupac_extra <- c("R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N")
  amino <- c(
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
  )

  switch(alphabet,
    dna = list(
      name = "dna",
      symbols = core_dna,
      gap = "-",
      wildcard = "N",
      nucleic = TRUE
    ),
    rna = list(
      name = "rna",
      symbols = core_rna,
      gap = "-",
      wildcard = "N",
      nucleic = TRUE
    ),
    iupac_dna = list(
      name = "iupac_dna",
      symbols = c(core_dna, iupac_extra),
      gap = "-",
      wildcard = "N",
      nucleic = TRUE
    ),
    iupac_rna = list(
      name = "iupac_rna",
      symbols = c(core_rna, iupac_extra),
      gap = "-",
      wildcard = "N",
      nucleic = TRUE
    ),
    protein = list(
      name = "protein",
      symbols = c(amino, "X", "*"),
      gap = "-",
      wildcard = "X",
      nucleic = FALSE
    )

  )
}

#' Check the arguments shared by the sequence accessors
#'
#' @param seq A single character string holding the sequence.
#' @param start,end Integer positions, 1-based and inclusive.
#' @param width A positive integer window width.
#' @param step A positive integer step between windows.
#' @return `TRUE`, invisibly, when every argument is valid.
#' @keywords internal
check_sequence_args <- function(seq, start, end, width, step) {
  # Validate `seq`
  if (!is.character(seq) || length(seq) != 1L || is.na(seq)) {
    stop(sprintf("`%s` must be a single non-missing string.", "seq"), call. = FALSE)
  }
  n <- nchar(seq)

  # Validate `start`
  if (!is.numeric(start) || length(start) != 1L || is.na(start)) {
    stop(sprintf("`%s` must be a single number.", "start"), call. = FALSE)
  }
  if (start < 1 || start > n) {
    stop(sprintf("`%s` must lie between 1 and %d.", "start", n), call. = FALSE)
  }
  # Validate `end`
  if (!is.numeric(end) || length(end) != 1L || is.na(end)) {
    stop(sprintf("`%s` must be a single number.", "end"), call. = FALSE)
  }
  if (end < 1 || end > n) {
    stop(sprintf("`%s` must lie between 1 and %d.", "end", n), call. = FALSE)
  }
  if (start > end) {
    stop("`start` must not be greater than `end`.", call. = FALSE)
  }
  # Validate `width`
  if (!is.numeric(width) || length(width) != 1L || is.na(width)) {
    stop(sprintf("`%s` must be a single number.", "width"), call. = FALSE)
  }
  if (width < 1 || width > n) {
    stop(sprintf("`%s` must lie between 1 and %d.", "width", n), call. = FALSE)
  }
  # Validate `step`
  if (!is.numeric(step) || length(step) != 1L || is.na(step)) {
    stop(sprintf("`%s` must be a single number.", "step"), call. = FALSE)
  }
  if (step < 1 || step > n) {
    stop(sprintf("`%s` must lie between 1 and %d.", "step", n), call. = FALSE)
  }

  invisible(TRUE)
}

#' Validate sequences against an alphabet
#'
#' @param x A character vector of sequences.
#' @param alphabet Passed to [alphabet_symbols()].
#' @param allow_gaps Whether gap symbols count as valid.
#' @param allow_lower Whether lower-case symbols count as valid.
#' @return A logical vector, one element per sequence, with attribute `invalid`
#'   listing the offending symbols of each sequence.
#' @export
validate_sequence <- function(x, alphabet = "dna", allow_gaps = TRUE, allow_lower = TRUE) {
  # Validate `x`
  if (!is.character(x)) {
    stop(sprintf("`%s` must be a character vector.", "x"), call. = FALSE)
  }
  # Validate `allow_gaps`
  if (!is.logical(allow_gaps) || length(allow_gaps) != 1L || is.na(allow_gaps)) {
    stop(sprintf("`%s` must be `TRUE` or `FALSE`.", "allow_gaps"), call. = FALSE)
  }
  # Validate `allow_lower`
  if (!is.logical(allow_lower) || length(allow_lower) != 1L || is.na(allow_lower)) {
    stop(sprintf("`%s` must be `TRUE` or `FALSE`.", "allow_lower"), call. = FALSE)
  }

  spec <- alphabet_symbols(alphabet)
  allowed <- spec$symbols
  if (allow_gaps) {
    allowed <- c(allowed, spec$gap)
  }
  if (allow_lower) {
    allowed <- c(allowed, tolower(spec$symbols))
  }

  valid <- logical(length(x))
  invalid <- vector("list", length(x))

  for (i in seq_along(x)) {
    if (is.na(x[i])) {
      valid[i] <- FALSE
      invalid[[i]] <- NA_character_
      next
    }
    chars <- strsplit(x[i], "", fixed = TRUE)[[1]]
    bad <- unique(chars[!chars %in% allowed])
    valid[i] <- length(bad) == 0L
    invalid[[i]] <- bad
  }

  names(invalid) <- names(x)
  attr(valid, "invalid") <- invalid

  valid
}

#' Tabulate problems in a set of sequences
#'
#' @param x A character vector of sequences.
#' @param alphabet Passed to [alphabet_symbols()].
#' @return A data frame with one row per sequence and the columns `length`,
#'   `empty`, `invalid`, `gap`, `lower` and `wildcard`, each counting symbols.
#' @export
sequence_issues <- function(x, alphabet = "dna") {
  spec <- alphabet_symbols(alphabet)
  n <- length(x)
  out <- data.frame(
    length = integer(n),
    empty = logical(n),
    invalid = integer(n),
    gap = integer(n),
    lower = integer(n),
    wildcard = integer(n)
  )

  for (i in seq_len(n)) {
    chars <- strsplit(x[i], "", fixed = TRUE)[[1]]
    upper <- toupper(chars)
    # Count `length`
    out$length[i] <- length(chars)
    # Count `empty`
    out$empty[i] <- length(chars) == 0L
    # Count `invalid`
    out$invalid[i] <- sum(!upper %in% c(spec$symbols, spec$gap))
    # Count `gap`
    out$gap[i] <- sum(chars == spec$gap)
    # Count `lower`
    out$lower[i] <- sum(chars != upper)
    # Count `wildcard`
    out$wildcard[i] <- sum(upper == spec$wildcard)
  }
  rownames(out) <- names(x)

  out
}

#' Remove or mask symbols outside an alphabet
#'
#' @param x A character vector of sequences.
#' @param alphabet Passed to [alphabet_symbols()].
#' @param action What to do with invalid symbols: `"drop"` removes them,
#'   `"mask"` replaces them with the wildcard, `"warn"` masks with a warning and
#'   `"error"` stops.
#' @return A character vector of cleaned, upper-case sequences.
#' @export
clean_sequence <- function(x, alphabet = "dna", action = c("drop", "mask", "warn", "error")) {
  action <- match.arg(action)
  spec <- alphabet_symbols(alphabet)
  allowed <- c(spec$symbols, spec$gap)
  out <- character(length(x))

  for (i in seq_along(x)) {
    chars <- toupper(strsplit(x[i], "", fixed = TRUE)[[1]])
    bad <- !chars %in% allowed
    if (!any(bad)) {
      out[i] <- paste(chars, collapse = "")
      next
    }
    switch(action,
      drop = {
        chars <- chars[!bad]
        out[i] <- paste(chars, collapse = "")
      },
      mask = {
        chars[bad] <- spec$wildcard
        out[i] <- paste(chars, collapse = "")
      },
      warn = {
        warning(sprintf("Sequence %d: %d invalid symbol(s) masked.", i, sum(bad)), call. = FALSE)
        chars[bad] <- spec$wildcard
        out[i] <- paste(chars, collapse = "")
      },
      error = {
        stop(sprintf("Sequence %d: %d invalid symbol(s) found.", i, sum(bad)), call. = FALSE)
      }
    )
  }
  names(out) <- names(x)

  out
}

#' Guess the alphabet of a sequence
#'
#' Tries the candidates from the most to the least specific and returns the
#' first whose symbols cover every character.
#'
#' @param x A single character string.
#' @param candidates Alphabet names to try, in order.
#' @return The name of the first matching alphabet, or `NA` if none matches.
#' @export
detect_alphabet <- function(x, candidates = c("dna", "rna", "iupac_dna", "iupac_rna", "protein")) {
  if (!is.character(x) || length(x) != 1L) {
    stop(sprintf("`%s` must be a single string.", "x"), call. = FALSE)
  }
  chars <- unique(toupper(strsplit(x, "", fixed = TRUE)[[1]]))
  if (length(chars) == 0L) {
    return(NA_character_)
  }

  coverage <- numeric(length(candidates))
  names(coverage) <- candidates

  for (alphabet in candidates) {
    spec <- alphabet_symbols(alphabet)
    allowed <- c(spec$symbols, spec$gap)
    hits <- chars %in% allowed
    coverage[[alphabet]] <- mean(hits)
    if (all(hits)) {
      return(alphabet)
    }
  }
  best <- names(coverage)[which.max(coverage)]
  message(sprintf("No alphabet covers `x`; closest is %s.", best))

  NA_character_
}
