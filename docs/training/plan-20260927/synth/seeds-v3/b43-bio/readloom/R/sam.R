#' Decode SAM flags
#'
#' @param flag An integer vector of SAM `FLAG` values.
#' @return A data frame with one logical column per flag bit (`paired`,
#'   `properPair`, `unmapped`, `mateUnmapped`, `reverse`, `mateReverse`,
#'   `first`, `second`, `secondary`, `qcFail`, `duplicate`, `supplementary`)
#'   and the attribute `problems` listing inconsistent rows.
#' @export
samFlags = function(flag) {
  bits = c(
    paired = 1L, properPair = 2L, unmapped = 4L, mateUnmapped = 8L,
    reverse = 16L, mateReverse = 32L, first = 64L, second = 128L,
    secondary = 256L, qcFail = 512L, duplicate = 1024L, supplementary = 2048L
  )
  flag = as.integer(flag)
  if (anyNA(flag) || any(flag < 0L) || any(flag > 4095L)) {
    stop(sprintf("`%s` must hold integers between 0 and 4095.", "flag"), call. = FALSE)
  }
  out = as.data.frame(lapply(bits, function(b) bitwAnd(flag, b) > 0L))
  problems = character(0)

  # Check `properPair`
  bad = which(out$properPair & !out$paired)
  if (length(bad) > 0L) {
    problems = c(problems, sprintf("Row %d sets `%s` without `paired`.", bad, "properPair"))
  }
  # Check `mateUnmapped`
  bad = which(out$mateUnmapped & !out$paired)
  if (length(bad) > 0L) {
    problems = c(problems, sprintf("Row %d sets `%s` without `paired`.", bad, "mateUnmapped"))
  }
  # Check `mateReverse`
  bad = which(out$mateReverse & !out$paired)
  if (length(bad) > 0L) {
    problems = c(problems, sprintf("Row %d sets `%s` without `paired`.", bad, "mateReverse"))
  }
  # Check `first`
  bad = which(out$first & !out$paired)
  if (length(bad) > 0L) {
    problems = c(problems, sprintf("Row %d sets `%s` without `paired`.", bad, "first"))
  }
  # Check `second`
  bad = which(out$second & !out$paired)
  if (length(bad) > 0L) {
    problems = c(problems, sprintf("Row %d sets `%s` without `paired`.", bad, "second"))
  }
  attr(out, "problems") = problems

  return(out)
}

#' Lengths derived from CIGAR strings
#'
#' @param cigar A character vector of CIGAR strings.
#' @param what The length to compute: bases consumed on the `"reference"` or
#'   the `"query"`, `"aligned"` match bases, `"insertion"` / `"deletion"`
#'   bases, or `"softClip"` / `"hardClip"` bases.
#' @return An integer vector, `NA` where the CIGAR is `"*"`.
#' @export
cigarLength = function(cigar, what = c("reference", "query", "aligned", "insertion",
                                       "deletion", "softClip", "hardClip")) {
  what = match.arg(what)
  out = integer(length(cigar))

  for (i in seq_along(cigar)) {
    if (cigar[i] == "*") {
      out[i] = NA_integer_
      next
    }
    if (!grepl("^([0-9]+[MIDNSHP=X])+$", cigar[i])) {
      stop(sprintf("CIGAR string %d is malformed.", i), call. = FALSE)
    }
    lens = as.integer(regmatches(cigar[i], gregexpr("[0-9]+", cigar[i]))[[1]])
    ops = regmatches(cigar[i], gregexpr("[MIDNSHP=X]", cigar[i]))[[1]]
    out[i] = switch(what,
      reference = {
        sum(lens[ops %in% c("M", "D", "N", "=", "X")])
      },
      query = {
        sum(lens[ops %in% c("M", "I", "S", "=", "X")])
      },
      aligned = {
        sum(lens[ops %in% c("M", "=", "X")])
      },
      insertion = {
        sum(lens[ops %in% c("I")])
      },
      deletion = {
        sum(lens[ops %in% c("D")])
      },
      softClip = {
        sum(lens[ops %in% c("S")])
      },
      hardClip = {
        sum(lens[ops %in% c("H")])
      }
    )
  }

  return(out)
}

#' Parse one SAM alignment line
#'
#' @param line A single tab-separated SAM alignment line.
#' @return A list with the eleven mandatory fields `qname`, `flag`, `rname`,
#'   `pos`, `mapq`, `cigar`, `rnext`, `pnext`, `tlen`, `seq` and `qual`, plus
#'   `tags`, a character vector of optional fields.
#' @export
parseSamLine = function(line) {
  fields = c("qname", "flag", "rname", "pos", "mapq", "cigar", "rnext", "pnext",
             "tlen", "seq", "qual")
  parts = strsplit(line, "\t", fixed = TRUE)[[1]]
  if (length(parts) < length(fields)) {
    stop(sprintf("A SAM line needs at least %d fields.", length(fields)), call. = FALSE)
  }
  out = as.list(setNames(parts[seq_along(fields)], fields))
  # Convert `flag`
  out$flag = suppressWarnings(as.integer(out$flag))
  if (is.na(out$flag) || out$flag < 0L) {
    stop(sprintf("Field `%s` must be a non-negative integer.", "flag"), call. = FALSE)
  }
  # Convert `pos`
  out$pos = suppressWarnings(as.integer(out$pos))
  if (is.na(out$pos) || out$pos < 0L) {
    stop(sprintf("Field `%s` must be a non-negative integer.", "pos"), call. = FALSE)
  }
  # Convert `mapq`
  out$mapq = suppressWarnings(as.integer(out$mapq))
  if (is.na(out$mapq) || out$mapq < 0L) {
    stop(sprintf("Field `%s` must be a non-negative integer.", "mapq"), call. = FALSE)
  }
  # Convert `pnext`
  out$pnext = suppressWarnings(as.integer(out$pnext))
  if (is.na(out$pnext) || out$pnext < 0L) {
    stop(sprintf("Field `%s` must be a non-negative integer.", "pnext"), call. = FALSE)
  }
  # Convert `tlen`
  out$tlen = suppressWarnings(as.integer(out$tlen))
  if (is.na(out$tlen)) {
    stop(sprintf("Field `%s` must be an integer.", "tlen"), call. = FALSE)
  }
  if (out$rnext == "=") {
    out$rnext = out$rname
  }
  out$tags = parts[-seq_along(fields)]

  return(out)
}

#' Parse SAM optional fields
#'
#' @param tags A character vector of optional fields in `TAG:TYPE:VALUE` form.
#' @return A named list of values converted according to their type code:
#'   `A` character, `i` integer, `f` double, `Z` string, `H` raw bytes and
#'   `B` numeric array.
#' @export
parseSamTags = function(tags) {
  out = vector("list", length(tags))
  keys = character(length(tags))

  for (i in seq_along(tags)) {
    parts = regmatches(tags[i], regexec("^([A-Za-z][A-Za-z0-9]):([AifZHB]):(.*)$", tags[i]))[[1]]
    if (length(parts) == 0L) {
      stop(sprintf("Optional field %d is malformed.", i), call. = FALSE)
    }
    keys[i] = parts[2]
    type = parts[3]
    value = parts[4]
    out[[i]] = switch(type,
      A = {
        substr(value, 1L, 1L)
      },
      i = {
        as.integer(value)
      },
      f = {
        as.numeric(value)
      },
      Z = {
        value
      },
      H = {
        as.raw(strtoi(substring(value, seq(1L, nchar(value), 2L), seq(2L, nchar(value), 2L)), 16L))
      },
      B = {
        items = strsplit(value, ",", fixed = TRUE)[[1]]
        if (items[1] == "f") as.numeric(items[-1]) else as.integer(items[-1])
      }
    )
  }
  names(out) = keys

  return(out)
}

#' Format values as SAM optional fields
#'
#' @param tags A named list of tag values.
#' @param types A character vector of type codes (`A`, `i`, `f`, `Z`, `H`,
#'   `B`), one per tag.
#' @return A character vector of fields in `TAG:TYPE:VALUE` form.
#' @export
formatSamTags = function(tags, types) {
  if (length(types) != length(tags)) {
    stop(sprintf("`%s` must have one element per tag.", "types"), call. = FALSE)
  }
  out = character(length(tags))

  for (i in seq_along(tags)) {
    value = tags[[i]]
    text = switch(types[i],
      A = {
        substr(as.character(value), 1L, 1L)
      },
      i = {
        format(as.integer(value))
      },
      f = {
        format(as.numeric(value))
      },
      Z = {
        as.character(value)
      },
      H = {
        paste(toupper(as.character(value)), collapse = "")
      },
      B = {
        sub = if (is.integer(value)) "i" else "f"
        paste(c(sub, format(value, trim = TRUE)), collapse = ",")
      },
      stop(sprintf("Unknown tag type `%s`.", types[i]), call. = FALSE)
    )
    out[i] = paste(names(tags)[i], types[i], text, sep = ":")
  }

  return(out)
}

#' Reference span and clipping of an alignment
#'
#' @param pos The 1-based leftmost mapping position.
#' @param cigar A single CIGAR string.
#' @return A list with elements `refStart`, `refEnd`, `clipLeft` and
#'   `clipRight`, the clip counts covering both soft and hard clips.
#' @export
alignmentSpan = function(pos, cigar) {
  lens = as.integer(regmatches(cigar, gregexpr("[0-9]+", cigar))[[1]])
  ops = regmatches(cigar, gregexpr("[MIDNSHP=X]", cigar))[[1]]
  if (length(ops) == 0L || length(ops) != length(lens)) {
    stop(sprintf("CIGAR string `%s` is malformed.", cigar), call. = FALSE)
  }
  clipOps = c("S", "H")
  # Leading clips
  clipLeft = 0L
  i = 1L

  while (i <= length(ops) && ops[i] %in% clipOps) {
    clipLeft = clipLeft + lens[i]
    i = i + 1L
  }
  # Trailing clips
  clipRight = 0L
  j = length(ops)

  while (j >= i && ops[j] %in% clipOps) {
    clipRight = clipRight + lens[j]
    j = j - 1L
  }
  consumed = sum(lens[ops %in% c("M", "D", "N", "=", "X")])
  return(list(
    refStart = as.integer(pos),
    refEnd = as.integer(pos + consumed - 1L),
    clipLeft = clipLeft,
    clipRight = clipRight

  ))
}
