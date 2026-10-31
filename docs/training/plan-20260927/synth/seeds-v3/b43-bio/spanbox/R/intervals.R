#' Build an interval table
#'
#' @param chrom A character vector of sequence names.
#' @param start,end Integer vectors of 1-based, inclusive positions.
#' @param strand A character vector of `"+"`, `"-"` or `"*"`, recycled.
#' @param name An optional character vector of interval names.
#' @return A data frame with columns `chrom`, `start`, `end`, `strand` and
#'   `name`, with class `"spanbox_intervals"` added.
#' @export
make.intervals <- function(chrom, start, end, strand = "*", name = NULL) {
  n <- length(chrom)
  # Validate `chrom`
  if (!is.character(chrom) || anyNA(chrom)) {
    stop(sprintf("`%s` must be a character vector without missing values.", "chrom"),
         call. = FALSE)
  }
  # Validate `start`
  if (!is.numeric(start) || length(start) != n || anyNA(start) || any(start < 1)) {
    stop(sprintf("`%s` must be %d positive integer(s).", "start", n), call. = FALSE)
  }
  # Validate `end`
  if (!is.numeric(end) || length(end) != n || anyNA(end) || any(end < 1)) {
    stop(sprintf("`%s` must be %d positive integer(s).", "end", n), call. = FALSE)
  }
  if (any(end < start)) {
    stop("`end` must not be smaller than `start`.", call. = FALSE)
  }
  # Validate `strand`
  strand <- rep_len(strand, n)
  if (!all(strand %in% c("+", "-", "*"))) {
    stop(sprintf("`%s` must only contain \"+\", \"-\" or \"*\".", "strand"), call. = FALSE)
  }
  # Validate `name`
  if (is.null(name)) {
    name <- paste0("int", seq_len(n))
  } else if (!is.character(name) || length(name) != n) {
    stop(sprintf("`%s` must be a character vector of length %d.", "name", n), call. = FALSE)
  }

  out <- data.frame(
    chrom = chrom,
    start = as.integer(start),
    end = as.integer(end),
    strand = strand,
    name = name
  )
  class(out) <- c("spanbox_intervals", class(out))

  out
}

#' Resize intervals around an anchor
#'
#' Keeps the anchor fixed and sets every interval to `width`. Anchors respect
#' strand: on the `-` strand the start of a feature is its right end.
#'
#' @param x An interval table from [make.intervals()].
#' @param width The new width.
#' @param fix The anchor: `"start"`, `"end"` or `"center"`.
#' @return `x` with updated `start` and `end`.
#' @export
resize.intervals <- function(x, width, fix = c("start", "end", "center")) {
  fix <- match.arg(fix)
  width <- rep_len(as.integer(width), nrow(x))
  if (any(width < 1L)) {
    stop(sprintf("`%s` must be at least 1.", "width"), call. = FALSE)
  }
  minus <- x$strand == "-"

  switch(fix,
    start = {
      new.start <- ifelse(minus, x$end - width + 1L, x$start)
      new.end <- ifelse(minus, x$end, x$start + width - 1L)
    },
    end = {
      new.start <- ifelse(minus, x$start, x$end - width + 1L)
      new.end <- ifelse(minus, x$start + width - 1L, x$end)
    },
    center = {
      mid <- (x$start + x$end) %/% 2L
      new.start <- mid - (width - 1L) %/% 2L
      new.end <- new.start + width - 1L
    }
  )
  x$start <- pmax(as.integer(new.start), 1L)
  x$end <- as.integer(new.end)

  x
}

#' Flanking regions of intervals
#'
#' @param x An interval table from [make.intervals()].
#' @param width The flank width.
#' @param side `"upstream"`, `"downstream"` or `"both"`, relative to strand.
#' @return An interval table of flanks; with `side = "both"` the upstream flank
#'   of each interval comes before its downstream flank.
#' @export
flank.intervals <- function(x, width, side = c("upstream", "downstream", "both")) {
  side <- match.arg(side)
  minus <- x$strand == "-"
  # Left flank: upstream on +, downstream on -
  left <- x
  left$end <- x$start - 1L
  left$start <- pmax(x$start - as.integer(width), 1L)
  # Right flank: downstream on +, upstream on -
  right <- x
  right$start <- x$end + 1L
  right$end <- x$end + as.integer(width)

  up <- x
  up[!minus, ] <- left[!minus, ]
  up[minus, ] <- right[minus, ]
  down <- x
  down[!minus, ] <- right[!minus, ]
  down[minus, ] <- left[minus, ]

  out <- switch(side,
    upstream = up,
    downstream = down,
    both = {
      both <- rbind(up, down)
      both[order(rep(seq_len(nrow(x)), 2L)), , drop = FALSE]
    }
  )
  out <- out[out$end >= out$start, , drop = FALSE]
  rownames(out) <- NULL

  out
}

#' Clip intervals to sequence bounds
#'
#' @param x An interval table from [make.intervals()].
#' @param chrom.sizes A named integer vector of sequence lengths.
#' @param drop Whether to drop intervals that fall entirely outside.
#' @return `x` with clipped coordinates and the attributes `n.clipped.start`
#'   and `n.clipped.end`.
#' @export
clip.intervals <- function(x, chrom.sizes, drop = TRUE) {
  # Validate `chrom.sizes`
  if (!is.numeric(chrom.sizes) || is.null(names(chrom.sizes))) {
    stop(sprintf("`%s` must be a named numeric vector.", "chrom.sizes"), call. = FALSE)
  }
  # Validate `drop`
  if (!is.logical(drop) || length(drop) != 1L || is.na(drop)) {
    stop(sprintf("`%s` must be `TRUE` or `FALSE`.", "drop"), call. = FALSE)
  }
  unknown <- setdiff(unique(x$chrom), names(chrom.sizes))
  if (length(unknown) > 0L) {
    stop(sprintf("No size for sequence `%s`.", unknown[1]), call. = FALSE)
  }

  limit <- as.integer(chrom.sizes[x$chrom])
  outside <- x$start > limit | x$end < 1L
  # Clip the lower bound
  low <- x$start < 1L
  x$start[low] <- 1L
  # Clip the upper bound
  high <- x$end > limit
  x$end[high] <- limit[high]

  if (drop) {
    x <- x[!outside, , drop = FALSE]
  }
  attr(x, "n.clipped.start") <- sum(low & !outside)
  attr(x, "n.clipped.end") <- sum(high & !outside)

  x
}

#' Promoter regions
#'
#' @param x An interval table from [make.intervals()].
#' @param upstream Bases before the transcription start.
#' @param downstream Bases after the transcription start, including it.
#' @return An interval table of promoters, strand-aware, with `name` suffixed
#'   by `".promoter"`.
#' @export
promoter.intervals <- function(x, upstream = 2000L, downstream = 200L) {
  # Validate `x`
  if (!is.data.frame(x) || !all(c("start", "end", "strand") %in% names(x))) {
    stop(sprintf("`%s` must be an interval table.", "x"), call. = FALSE)
  }
  # Validate `upstream`
  if (!is.numeric(upstream) || length(upstream) != 1L || upstream < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "upstream"), call. = FALSE)
  }
  # Validate `downstream`
  if (!is.numeric(downstream) || length(downstream) != 1L || downstream < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "downstream"), call. = FALSE)
  }

  out <- x
  plus <- x$strand != "-"
  minus <- !plus
  # Plus strand: the start is the transcription start
  tss.plus <- x$start[plus]
  out$start[plus] <- tss.plus - as.integer(upstream)
  out$end[plus] <- tss.plus + as.integer(downstream) - 1L
  # Minus strand: the end is the transcription start
  tss.minus <- x$end[minus]
  out$start[minus] <- tss.minus - as.integer(downstream) + 1L
  out$end[minus] <- tss.minus + as.integer(upstream)

  out$start <- pmax(out$start, 1L)
  if ("name" %in% names(out)) {
    out$name <- paste0(out$name, ".promoter")
  }

  out
}

#' Summarise an interval table
#'
#' @param x An interval table from [make.intervals()].
#' @return A list with elements `n`, `total.width`, `mean.width`,
#'   `median.width`, `min.width`, `max.width`, `n.chrom` and `strand.counts`.
#' @export
interval.stats <- function(x) {
  widths <- x$end - x$start + 1L
  empty <- length(widths) == 0L
  # n
  n <- nrow(x)
  # total.width
  total.width <- sum(widths)
  # mean.width
  mean.width <- if (empty) NA_real_ else mean(widths)
  # median.width
  median.width <- if (empty) NA_real_ else stats::median(widths)
  # min.width
  min.width <- if (empty) NA_integer_ else min(widths)
  # max.width
  max.width <- if (empty) NA_integer_ else max(widths)
  # n.chrom
  n.chrom <- length(unique(x$chrom))
  # strand.counts
  strand.counts <- factor(x$strand, levels = c("+", "-", "*")) |> table() |> c()

  list(
    n = n,
    total.width = total.width,
    mean.width = mean.width,
    median.width = median.width,
    min.width = min.width,
    max.width = max.width,
    n.chrom = n.chrom,
    strand.counts = strand.counts

  )
}
