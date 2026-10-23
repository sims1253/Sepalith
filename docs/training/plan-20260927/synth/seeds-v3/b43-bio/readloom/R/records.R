#' Split file lines into records
#'
#' @param lines A character vector of lines from a sequence file.
#' @param format One of `"fasta"`, `"fastq"`, `"genbank"` or `"embl"`.
#' @return A list of character vectors, one per record.
#' @export
splitRecords = function(lines, format = c("fasta", "fastq", "genbank", "embl")) {
  format = match.arg(format)
  lines = lines[nzchar(trimws(lines))]
  if (length(lines) == 0L) {
    return(list())
  }
  switch(format,
    fasta = {
      starts = grep("^>", lines)
      ends = c(starts[-1] - 1L, length(lines))
    },
    fastq = {
      if (length(lines) %% 4L != 0L) {
        stop(sprintf("A %s file must have a multiple of 4 lines.", "fastq"), call. = FALSE)
      }
      starts = seq(1L, length(lines), by = 4L)
      ends = starts + 3L
    },
    genbank = {
      starts = grep("^LOCUS", lines)
      ends = grep("^//", lines)
    },
    embl = {
      starts = grep("^ID   ", lines)
      ends = grep("^//", lines)
    }
  )
  if (length(starts) == 0L || length(starts) != length(ends)) {
    stop(sprintf("Could not find complete %s records.", format), call. = FALSE)
  }
  records = vector("list", length(starts))

  for (i in seq_along(starts)) {
    records[[i]] = lines[starts[i]:ends[i]]
  }

  return(records)
}

#' Parse one record
#'
#' @param lines The lines of a single record, as returned by [splitRecords()].
#' @param format One of `"fasta"`, `"fastq"`, `"genbank"` or `"embl"`.
#' @return A list with elements `id`, `description`, `sequence` and `quality`
#'   (`NA` for formats without qualities).
#' @export
parseRecord = function(lines, format = c("fasta", "fastq", "genbank", "embl")) {
  format = match.arg(format)

  switch(format,
    fasta = {
      header = sub("^>", "", lines[1])
      id = sub("\\s.*$", "", header)
      description = trimws(sub("^\\S+", "", header))
      sequence = paste(lines[-1], collapse = "")
      quality = NA_character_
    },
    fastq = {
      header = sub("^@", "", lines[1])
      id = sub("\\s.*$", "", header)
      description = trimws(sub("^\\S+", "", header))
      sequence = lines[2]
      quality = lines[4]
    },
    genbank = {
      header = sub("^LOCUS\\s+", "", lines[1])
      id = sub("\\s.*$", "", header)
      description = trimws(sub("^DEFINITION", "", grep("^DEFINITION", lines, value = TRUE)[1]))
      body = lines[(grep("^ORIGIN", lines) + 1L):(length(lines) - 1L)]
      sequence = gsub("[^A-Za-z]", "", paste(body, collapse = ""))
      quality = NA_character_
    },
    embl = {
      header = sub("^ID\\s+", "", lines[1])
      id = sub(";.*$", "", header)
      description = trimws(sub("^DE", "", grep("^DE   ", lines, value = TRUE)[1]))
      body = lines[(grep("^SQ", lines) + 1L):(length(lines) - 1L)]
      sequence = gsub("[^A-Za-z]", "", paste(body, collapse = ""))
      quality = NA_character_
    }
  )
  return(list(
    id = id,
    description = description,
    sequence = toupper(sequence),
    quality = quality

  ))
}

#' Format one record as file lines
#'
#' @param rec A record list with elements `id`, `description`, `sequence` and
#'   `quality`.
#' @param format One of `"fasta"`, `"fastq"`, `"genbank"` or `"embl"`.
#' @param width The number of bases per sequence line.
#' @return A character vector of lines.
#' @export
formatRecord = function(rec, format = c("fasta", "fastq", "genbank", "embl"), width = 60L) {
  format = match.arg(format)
  seq = rec$sequence
  n = nchar(seq)
  starts = seq(1L, max(n, 1L), by = width)
  chunks = substring(seq, starts, pmin(starts + width - 1L, n))

  switch(format,
    fasta = {
      header = paste0(">", rec$id, if (nzchar(rec$description)) " ", rec$description)
      out = c(header, chunks)
    },
    fastq = {
      header = paste0("@", rec$id, if (nzchar(rec$description)) " ", rec$description)
      out = c(header, seq, "+", rec$quality)
    },
    genbank = {
      header = sprintf("LOCUS       %s %d bp DNA", rec$id, n)
      body = sprintf("%9d %s", starts, tolower(chunks))
      out = c(header, paste("DEFINITION ", rec$description), "ORIGIN", body, "//")
    },
    embl = {
      header = sprintf("ID   %s; DNA; %d BP.", rec$id, n)
      body = sprintf("     %s %d", tolower(chunks), pmin(starts + width - 1L, n))
      out = c(header, paste("DE  ", rec$description), "SQ   Sequence", body, "//")
    }
  )

  return(out)
}

#' Check a parsed record
#'
#' @param rec A record list with elements `id`, `description`, `sequence` and
#'   `quality`.
#' @param format One of `"fasta"`, `"fastq"`, `"genbank"` or `"embl"`.
#' @return A character vector of problems, empty when the record is valid.
#' @keywords internal
validateRecord = function(rec, format = c("fasta", "fastq", "genbank", "embl")) {
  format = match.arg(format)
  fields = c("id", "description", "sequence", "quality")
  problems = character(0)
  missingFields = setdiff(fields, names(rec))
  if (length(missingFields) > 0L) {
    return(sprintf("Field `%s` is missing.", missingFields))
  }
  # Check `id`
  if (!is.character(rec$id) || length(rec$id) != 1L || !nzchar(rec$id)) {
    problems = c(problems, sprintf("Field `%s` must be a non-empty string.", "id"))
  }
  # Check `description`
  if (!is.character(rec$description) || length(rec$description) != 1L) {
    problems = c(problems, sprintf("Field `%s` must be a single string.", "description"))
  }
  # Check `sequence`
  if (!is.character(rec$sequence) || length(rec$sequence) != 1L || !nzchar(rec$sequence)) {
    problems = c(problems, sprintf("Field `%s` must be a non-empty string.", "sequence"))
  }
  # Check `quality`
  if (format == "fastq") {
    if (is.na(rec$quality) || nchar(rec$quality) != nchar(rec$sequence)) {
      problems = c(problems, sprintf("Field `%s` must match the sequence length.", "quality"))
    }
  } else if (!is.na(rec$quality)) {
    problems = c(problems, sprintf("Field `%s` must be `NA` for %s.", "quality", format))
  }

  return(problems)
}

#' Read a sequence file
#'
#' @param path The path to the file.
#' @param format One of `"fasta"`, `"fastq"`, `"genbank"` or `"embl"`.
#' @param as The result shape: a `"list"` of records, a `"data.frame"` with
#'   one row per record, or a `"named"` character vector of sequences.
#' @return The records in the requested shape.
#' @export
readRecords = function(path, format = c("fasta", "fastq", "genbank", "embl"),
                       as = c("list", "data.frame", "named")) {
  format = match.arg(format)
  as = match.arg(as)
  if (!file.exists(path)) {
    stop(sprintf("File `%s` does not exist.", path), call. = FALSE)
  }
  chunks = splitRecords(readLines(path, warn = FALSE), format)
  records = lapply(chunks, parseRecord, format = format)
  problems = lapply(records, validateRecord, format = format)
  bad = lengths(problems) > 0L
  if (any(bad)) {
    warning(sprintf("Dropping %d invalid record(s).", sum(bad)), call. = FALSE)
    records = records[!bad]
  }
  ids = vapply(records, function(r) r$id, character(1))

  switch(as,
    list = {
      out = setNames(records, ids)
    },
    data.frame = {
      out = data.frame(
        id = ids,
        description = vapply(records, function(r) r$description, character(1)),
        sequence = vapply(records, function(r) r$sequence, character(1)),
        quality = vapply(records, function(r) r$quality, character(1))
      )
    },
    named = {
      out = setNames(vapply(records, function(r) r$sequence, character(1)), ids)
    }
  )

  return(out)
}
