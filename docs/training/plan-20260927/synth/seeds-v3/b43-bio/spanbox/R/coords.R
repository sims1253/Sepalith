#' Convert coordinates between conventions
#'
#' `"bed"` and `"psl"` use 0-based half-open intervals, `"bam"` stores a
#' 0-based start with an inclusive end, `"gff"` and `"vcf"` are 1-based closed.
#'
#' @param start,end Integer vectors of positions in the `from` convention.
#' @param from,to One of `"bed"`, `"psl"`, `"bam"`, `"gff"` or `"vcf"`.
#' @return A list with elements `start` and `end` in the `to` convention.
#' @export
convert.coords <- function(start, end, from = c("bed", "psl", "bam", "gff", "vcf"),
                           to = c("bed", "psl", "bam", "gff", "vcf")) {
  from <- match.arg(from)
  to <- match.arg(to)
  if (length(start) != length(end)) {
    stop("`start` and `end` must have the same length.", call. = FALSE)
  }
  # From `from` to 1-based closed
  switch(from,
    bed = ,
    psl = {
      start1 <- start + 1L
      end1 <- end
    },
    bam = {
      start1 <- start + 1L
      end1 <- end + 1L
    },
    gff = ,
    vcf = {
      start1 <- start
      end1 <- end
    }
  )
  # From 1-based closed to `to`
  switch(to,
    bed = ,
    psl = {
      start.out <- start1 - 1L
      end.out <- end1
    },
    bam = {
      start.out <- start1 - 1L
      end.out <- end1 - 1L
    },
    gff = ,
    vcf = {
      start.out <- start1
      end.out <- end1
    }
  )

  list(start = as.integer(start.out), end = as.integer(end.out))
}

#' Parse region strings
#'
#' Accepts `"chr1:1,000-2,000"`, optionally followed by a strand in
#' parentheses such as `"(+)"`.
#'
#' @param region A character vector of region strings.
#' @return A data frame with columns `chrom`, `start`, `end` and `strand`.
#' @export
parse.region <- function(region) {
  pattern <- "^([^:]+):([0-9,]+)-([0-9,]+)(\\(([+*-])\\))?$"
  parts <- regmatches(region, regexec(pattern, region))
  bad <- lengths(parts) == 0L
  if (any(bad)) {
    stop(sprintf("Region `%s` is malformed.", region[bad][1]), call. = FALSE)
  }
  n <- length(region)
  chrom <- character(n)
  start <- integer(n)
  end <- integer(n)
  strand <- character(n)

  for (i in seq_len(n)) {
    p <- parts[[i]]
    # chrom
    chrom[i] <- p[2]
    # start
    start[i] <- as.integer(gsub(",", "", p[3], fixed = TRUE))
    # end
    end[i] <- as.integer(gsub(",", "", p[4], fixed = TRUE))
    # strand
    strand[i] <- if (nzchar(p[6])) p[6] else "*"
  }
  if (any(end < start)) {
    stop("`end` must not be smaller than `start`.", call. = FALSE)
  }

  data.frame(chrom = chrom, start = start, end = end, strand = strand)
}

#' Format intervals as region strings
#'
#' @param x An interval table with columns `chrom`, `start`, `end` and `strand`.
#' @param style `"ucsc"` (`chr1:100-200`), `"ensembl"` (`1:100-200:1`),
#'   `"igv"` (`chr1:100-200` with thousands separators), `"genbank"`
#'   (`100..200`, wrapped in `complement()` on the minus strand), `"stranded"`
#'   (`chr1:100-200(+)`) or `"bed"` (tab-separated, 0-based start).
#' @return A character vector of region strings.
#' @export
region.string <- function(x, style = c("ucsc", "ensembl", "igv", "genbank", "stranded", "bed")) {
  style <- match.arg(style)
  if (nrow(x) == 0L) {
    return(character(0))
  }
  switch(style,
    ucsc = {
      sprintf("%s:%d-%d", x$chrom, x$start, x$end)
    },
    ensembl = {
      chrom <- sub("^chr", "", x$chrom)
      strand <- ifelse(x$strand == "-", "-1", "1")
      sprintf("%s:%d-%d:%s", chrom, x$start, x$end, strand)
    },
    igv = {
      start <- formatC(x$start, format = "d", big.mark = ",")
      end <- formatC(x$end, format = "d", big.mark = ",")
      sprintf("%s:%s-%s", x$chrom, start, end)
    },
    genbank = {
      span <- sprintf("%d..%d", x$start, x$end)
      ifelse(x$strand == "-", sprintf("complement(%s)", span), span)
    },
    stranded = {
      strand <- ifelse(x$strand %in% c("+", "-"), x$strand, "*")
      sprintf("%s:%d-%d(%s)", x$chrom, x$start, x$end, strand)
    },
    bed = {
      sprintf("%s\t%d\t%d", x$chrom, x$start - 1L, x$end)
    }

  )
}

#' Read intervals from text lines
#'
#' @param lines A character vector of tab-separated lines; comment and header
#'   lines starting with `#`, `track` or `browser` are skipped.
#' @param format One of `"bed"`, `"gff"`, `"vcf"` or `"narrowpeak"`.
#' @return A data frame with columns `chrom`, `start`, `end`, `strand` and
#'   `name`, using 1-based closed coordinates.
#' @export
read.interval.lines <- function(lines, format = c("bed", "gff", "vcf", "narrowpeak")) {
  format <- match.arg(format)
  lines <- lines[nzchar(lines) & !grepl("^(#|track|browser)", lines)]
  cols <- strsplit(lines, "\t", fixed = TRUE)
  get <- function(k) vapply(cols, function(f) if (length(f) >= k) f[k] else NA_character_, "")

  switch(format,
    bed = {
      chrom <- get(1)
      start <- as.integer(get(2)) + 1L
      end <- as.integer(get(3))
      name <- get(4)
      strand <- get(6)
    },
    gff = {
      chrom <- get(1)
      start <- as.integer(get(4))
      end <- as.integer(get(5))
      name <- sub(".*ID=([^;]+).*", "\\1", get(9))
      strand <- get(7)
    },
    vcf = {
      chrom <- get(1)
      start <- as.integer(get(2))
      end <- start + nchar(get(4)) - 1L
      name <- get(3)
      strand <- rep("*", length(cols))
    },
    narrowpeak = {
      chrom <- get(1)
      start <- as.integer(get(2)) + 1L
      end <- as.integer(get(3))
      name <- get(4)
      strand <- get(6)
    }
  )
  strand[is.na(strand) | !strand %in% c("+", "-")] <- "*"

  data.frame(chrom = chrom, start = start, end = end, strand = strand, name = name)
}

#' Write intervals as text lines
#'
#' @param x An interval table with columns `chrom`, `start`, `end`, `strand`
#'   and `name`, in 1-based closed coordinates.
#' @param format One of `"bed"`, `"gff"`, `"vcf"` or `"narrowpeak"`.
#' @param source The GFF source column.
#' @return A character vector of tab-separated lines.
#' @export
write.interval.lines <- function(x, format = c("bed", "gff", "vcf", "narrowpeak"),
                                 source = "spanbox") {
  format <- match.arg(format)
  required <- c("chrom", "start", "end", "strand", "name")
  missing.cols <- setdiff(required, names(x))
  if (length(missing.cols) > 0L) {
    stop(sprintf("`x` is missing column `%s`.", missing.cols[1]), call. = FALSE)
  }
  # Check `start`
  if (!is.numeric(x$start) || anyNA(x$start) || any(x$start < 1)) {
    stop(sprintf("Column `%s` must hold positive integers.", "start"), call. = FALSE)
  }
  # Check `end`
  if (!is.numeric(x$end) || anyNA(x$end) || any(x$end < 1)) {
    stop(sprintf("Column `%s` must hold positive integers.", "end"), call. = FALSE)
  }
  strand <- ifelse(x$strand %in% c("+", "-"), x$strand, ".")

  switch(format,
    bed = {
      fields <- list(x$chrom, x$start - 1L, x$end, x$name, 0L, strand)
    },
    gff = {
      fields <- list(x$chrom, source, "region", x$start, x$end, ".", strand, ".",
                     paste0("ID=", x$name))
    },
    vcf = {
      ref <- strrep("N", x$end - x$start + 1L)
      fields <- list(x$chrom, x$start, x$name, ref, ".", ".", "PASS", ".")
    },
    narrowpeak = {
      fields <- list(x$chrom, x$start - 1L, x$end, x$name, 0L, strand, -1, -1, -1, -1)
    }
  )

  do.call(paste, c(fields, sep = "\t"))
}

#' Assign intervals to fixed-width bins
#'
#' @param x An interval table with columns `chrom`, `start` and `end`.
#' @param bin.width The bin width in bases.
#' @return A data frame with columns `index`, `chrom`, `bin`, `bin.start`,
#'   `bin.end` and `overlap`, one row per interval and bin it touches.
#' @export
bin.intervals <- function(x, bin.width = 1000L) {
  if (!is.numeric(bin.width) || length(bin.width) != 1L || bin.width < 1) {
    stop(sprintf("`%s` must be a single positive number.", "bin.width"), call. = FALSE)
  }
  # First bin: the one holding `start`
  first <- (x$start - 1L) %/% bin.width + 1L
  # Last bin: the one holding `end`
  last <- (x$end - 1L) %/% bin.width + 1L
  n.bins <- last - first + 1L
  index <- rep(seq_len(nrow(x)), n.bins)
  bin <- unlist(Map(seq, first, last), use.names = FALSE)

  bin.start <- (bin - 1L) * bin.width + 1L
  bin.end <- bin * bin.width
  overlap <- pmin(bin.end, x$end[index]) - pmax(bin.start, x$start[index]) + 1L
  data.frame(
    index = index,
    chrom = x$chrom[index],
    bin = as.integer(bin),
    bin.start = as.integer(bin.start),
    bin.end = as.integer(bin.end),
    overlap = as.integer(overlap)

  )
}
