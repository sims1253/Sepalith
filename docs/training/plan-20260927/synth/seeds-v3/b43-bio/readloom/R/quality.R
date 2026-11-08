#' Quality score encoding parameters
#'
#' @description
#' ASCII offset, score range and score scale for each FASTQ quality encoding.
#'
#' @param encoding One of `"sanger"`, `"solexa"`, `"illumina13"`, `"illumina15"`
#'   or `"illumina18"`.
#' @return A list with elements `offset`, `minScore`, `maxScore` and `scale`.
#' @keywords internal
qualityEncoding = function(encoding = c("sanger", "solexa", "illumina13", "illumina15",
                                        "illumina18")) {
  encoding = match.arg(encoding)
  spec = switch(encoding,
    sanger = list(
      offset = 33L,
      minScore = 0L,
      maxScore = 93L,
      scale = "phred"
    ),
    solexa = list(
      offset = 64L,
      minScore = -5L,
      maxScore = 62L,
      scale = "solexa"
    ),
    illumina13 = list(
      offset = 64L,
      minScore = 0L,
      maxScore = 62L,
      scale = "phred"
    ),
    illumina15 = list(
      offset = 64L,
      minScore = 2L,
      maxScore = 62L,
      scale = "phred"
    ),
    illumina18 = list(
      offset = 33L,
      minScore = 0L,
      maxScore = 41L,
      scale = "phred"
    )
  )

  return(spec)
}

#' Decode quality strings to scores
#'
#' @param qual A character vector of quality strings.
#' @param encoding Passed to [qualityEncoding()].
#' @return A list of integer score vectors, one per string.
#' @export
decodeQuality = function(qual, encoding = "sanger") {
  if (!is.character(qual)) {
    stop(sprintf("`%s` must be a character vector.", "qual"), call. = FALSE)
  }
  spec = qualityEncoding(encoding)
  out = vector("list", length(qual))

  for (i in seq_along(qual)) {
    scores = utf8ToInt(qual[i]) - spec$offset
    # Check the lower bound
    if (any(scores < spec$minScore)) {
      stop(sprintf("Quality string %d has scores below %d for %s.",
                   i, spec$minScore, encoding), call. = FALSE)
    }
    # Check the upper bound
    if (any(scores > spec$maxScore)) {
      stop(sprintf("Quality string %d has scores above %d for %s.",
                   i, spec$maxScore, encoding), call. = FALSE)
    }
    out[[i]] = as.integer(scores)
  }
  names(out) = names(qual)

  return(out)
}

#' Encode scores as quality strings
#'
#' @param scores A list of integer score vectors, or a single integer vector.
#' @param encoding Passed to [qualityEncoding()].
#' @return A character vector of quality strings, one per score vector.
#' @export
encodeQuality = function(scores, encoding = "sanger") {
  if (is.numeric(scores)) {
    scores = list(scores)
  }
  spec = qualityEncoding(encoding)
  out = character(length(scores))

  for (i in seq_along(scores)) {
    s = round(scores[[i]])
    # Check the lower bound
    if (any(s < spec$minScore)) {
      stop(sprintf("Score vector %d has scores below %d for %s.",
                   i, spec$minScore, encoding), call. = FALSE)
    }
    # Check the upper bound
    if (any(s > spec$maxScore)) {
      stop(sprintf("Score vector %d has scores above %d for %s.",
                   i, spec$maxScore, encoding), call. = FALSE)
    }
    out[i] = intToUtf8(s + spec$offset)
  }
  names(out) = names(scores)

  return(out)
}

#' Convert quality strings between encodings
#'
#' @param qual A character vector of quality strings.
#' @param from,to Encodings passed to [qualityEncoding()].
#' @return A character vector of quality strings in the `to` encoding, with
#'   scores clamped to its range.
#' @export
convertQuality = function(qual, from = "illumina13", to = "sanger") {
  fromSpec = qualityEncoding(from)
  toSpec = qualityEncoding(to)
  scores = decodeQuality(qual, from)
  out = character(length(scores))

  for (i in seq_along(scores)) {
    s = scores[[i]]
    # Rescale solexa to phred
    if (fromSpec$scale == "solexa" && toSpec$scale == "phred") {
      s = 10 * log10(10^(s / 10) + 1)
    }
    # Rescale phred to solexa
    if (fromSpec$scale == "phred" && toSpec$scale == "solexa") {
      s = 10 * log10(pmax(10^(s / 10) - 1, 10^(toSpec$minScore / 10)))
    }
    s = round(s)
    # Clamp to the lower bound
    s[s < toSpec$minScore] = toSpec$minScore
    # Clamp to the upper bound
    s[s > toSpec$maxScore] = toSpec$maxScore
    out[i] = intToUtf8(s + toSpec$offset)
  }
  names(out) = names(qual)

  return(out)
}

#' Summarise quality strings
#'
#' @param qual A character vector of quality strings.
#' @param encoding Passed to [qualityEncoding()].
#' @return A list with elements `meanQ`, `medianQ`, `minQ`, `maxQ`, `q20`,
#'   `q30` and `expectedErrors`, each a numeric vector with one value per string.
#' @export
qualitySummary = function(qual, encoding = "sanger") {
  scores = decodeQuality(qual, encoding)
  n = length(scores)
  meanQ = numeric(n)
  medianQ = numeric(n)
  minQ = numeric(n)
  maxQ = numeric(n)
  q20 = numeric(n)
  q30 = numeric(n)
  expectedErrors = numeric(n)

  for (i in seq_len(n)) {
    s = scores[[i]]
    if (length(s) == 0L) {
      meanQ[i] = medianQ[i] = minQ[i] = maxQ[i] = NA_real_
      q20[i] = q30[i] = expectedErrors[i] = NA_real_
      next
    }
    meanQ[i] = mean(s)
    medianQ[i] = stats::median(s)
    minQ[i] = min(s)
    maxQ[i] = max(s)
    q20[i] = mean(s >= 20)
    q30[i] = mean(s >= 30)
    expectedErrors[i] = sum(10^(-s / 10))
  }
  return(list(
    meanQ = meanQ,
    medianQ = medianQ,
    minQ = minQ,
    maxQ = maxQ,
    q20 = q20,
    q30 = q30,
    expectedErrors = expectedErrors

  ))
}

#' Trim low-quality ends of a read
#'
#' @param seq A single read sequence.
#' @param qual The matching quality string.
#' @param threshold The lowest score kept at either end.
#' @param encoding Passed to [qualityEncoding()].
#' @param side Trim the `"left"` end, the `"right"` end or `"both"`.
#' @return A list with elements `sequence`, `quality`, `left` and `right`, the
#'   last two being the number of bases removed at each end.
#' @export
trimQuality = function(seq, qual, threshold = 20L, encoding = "sanger",
                       side = c("both", "left", "right")) {
  side = match.arg(side)
  if (nchar(seq) != nchar(qual)) {
    stop(sprintf("`%s` and `%s` must have the same length.", "seq", "qual"), call. = FALSE)
  }
  s = decodeQuality(qual, encoding)[[1]]
  n = length(s)
  first = 1L
  last = n

  # Trim the left end
  if (side %in% c("both", "left")) {
    while (first <= last && s[first] < threshold) {
      first = first + 1L
    }
  }
  # Trim the right end
  if (side %in% c("both", "right")) {
    while (last >= first && s[last] < threshold) {
      last = last - 1L
    }
  }

  return(list(
    sequence = substr(seq, first, last),
    quality = substr(qual, first, last),
    left = first - 1L,
    right = n - last

  ))
}
