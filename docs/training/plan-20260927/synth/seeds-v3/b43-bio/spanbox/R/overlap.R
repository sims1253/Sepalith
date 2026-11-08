#' Find overlapping pairs of intervals
#'
#' @param a,b Interval tables with columns `chrom`, `start` and `end`.
#' @param type The relation that counts as a hit: `"any"` overlap, `a` lying
#'   `"within"` `b`, `a` that `"contains"` `b`, a shared `"start"` or `"end"`,
#'   or `"equal"` coordinates.
#' @param ignore.strand Whether to match intervals on opposite strands.
#' @return A data frame with columns `a.index` and `b.index`.
#' @export
overlap.pairs <- function(a, b, type = c("any", "within", "contains", "start", "end", "equal"),
                          ignore.strand = TRUE) {
  type <- match.arg(type)
  hits <- list()

  for (i in seq_len(nrow(a))) {
    same <- b$chrom == a$chrom[i]
    if (!ignore.strand) {
      same <- same & (b$strand == a$strand[i] | b$strand == "*" | a$strand[i] == "*")
    }
    keep <- switch(type,
      any = {
        same & b$start <= a$end[i] & b$end >= a$start[i]
      },
      within = {
        same & b$start <= a$start[i] & b$end >= a$end[i]
      },
      contains = {
        same & b$start >= a$start[i] & b$end <= a$end[i]
      },
      start = {
        same & b$start == a$start[i]
      },
      end = {
        same & b$end == a$end[i]
      },
      equal = {
        same & b$start == a$start[i] & b$end == a$end[i]
      }
    )
    idx <- which(keep)
    if (length(idx) > 0L) {
      hits[[length(hits) + 1L]] <- data.frame(a.index = i, b.index = idx)
    }
  }
  if (length(hits) == 0L) {
    return(data.frame(a.index = integer(0), b.index = integer(0)))
  }

  do.call(rbind, hits)
}

#' Pairwise intersection of intervals
#'
#' @param a,b Interval tables with columns `chrom`, `start` and `end`.
#' @return A data frame with columns `chrom`, `start`, `end`, `width`,
#'   `a.index`, `b.index`, `a.fraction` and `b.fraction`, one row per
#'   overlapping pair; the fractions give the share of each input covered.
#' @export
pairwise.intersect <- function(a, b) {
  pairs <- overlap.pairs(a, b, type = "any")
  if (nrow(pairs) == 0L) {
    return(data.frame(chrom = character(0), start = integer(0), end = integer(0),
                      width = integer(0), a.index = integer(0), b.index = integer(0),
                      a.fraction = numeric(0), b.fraction = numeric(0)))
  }
  # Left bound: the later of the two starts
  a.start <- a$start[pairs$a.index]
  b.start <- b$start[pairs$b.index]
  start <- pmax(a.start, b.start)
  # Right bound: the earlier of the two ends
  a.end <- a$end[pairs$a.index]
  b.end <- b$end[pairs$b.index]
  end <- pmin(a.end, b.end)

  width <- end - start + 1L
  # Share of `a` covered
  a.fraction <- width / (a.end - a.start + 1L)
  # Share of `b` covered
  b.fraction <- width / (b.end - b.start + 1L)
  data.frame(
    chrom = a$chrom[pairs$a.index],
    start = start,
    end = end,
    width = width,
    a.index = pairs$a.index,
    b.index = pairs$b.index,
    a.fraction = a.fraction,
    b.fraction = b.fraction

  )
}

#' Merge overlapping or nearby intervals
#'
#' @param x An interval table with columns `chrom`, `start` and `end`.
#' @param gap Intervals separated by at most this many bases are merged.
#' @return A data frame with columns `chrom`, `start`, `end` and `n.merged`,
#'   sorted by sequence and position.
#' @export
reduce.intervals <- function(x, gap = 0L) {
  if (nrow(x) == 0L) {
    return(data.frame(chrom = character(0), start = integer(0), end = integer(0),
                      n.merged = integer(0)))
  }
  x <- x[order(x$chrom, x$start, x$end), , drop = FALSE]
  chrom <- character(nrow(x))
  start <- integer(nrow(x))
  end <- integer(nrow(x))
  n.merged <- integer(nrow(x))
  k <- 1L
  chrom[k] <- x$chrom[1]
  start[k] <- x$start[1]
  end[k] <- x$end[1]
  n.merged[k] <- 1L

  for (i in seq_len(nrow(x))[-1]) {
    if (x$chrom[i] == chrom[k] && x$start[i] <= end[k] + gap + 1L) {
      end[k] <- max(end[k], x$end[i])
      n.merged[k] <- n.merged[k] + 1L
    } else {
      k <- k + 1L
      chrom[k] <- x$chrom[i]
      start[k] <- x$start[i]
      end[k] <- x$end[i]
      n.merged[k] <- 1L
    }
  }
  data.frame(
    chrom = chrom[seq_len(k)],
    start = start[seq_len(k)],
    end = end[seq_len(k)],
    n.merged = n.merged[seq_len(k)]

  )
}

#' Distance from each interval to its nearest neighbour
#'
#' @param a,b Interval tables with columns `chrom`, `start` and `end`.
#' @return A data frame with columns `upstream`, `downstream` and `nearest`:
#'   the gap to the closest `b` interval on each side (0 when overlapping,
#'   `NA` when there is none), and the smaller of the two.
#' @export
interval.distance <- function(a, b) {
  n <- nrow(a)
  upstream <- rep(NA_integer_, n)
  downstream <- rep(NA_integer_, n)

  for (i in seq_len(n)) {
    cand <- b[b$chrom == a$chrom[i], , drop = FALSE]
    if (nrow(cand) == 0L) {
      next
    }
    if (any(cand$start <= a$end[i] & cand$end >= a$start[i])) {
      upstream[i] <- 0L
      downstream[i] <- 0L
      next
    }
    # Upstream: intervals ending before `a` starts
    left <- cand$end[cand$end < a$start[i]]
    if (length(left) > 0L) {
      upstream[i] <- as.integer(a$start[i] - max(left))
    }
    # Downstream: intervals starting after `a` ends
    right <- cand$start[cand$start > a$end[i]]
    if (length(right) > 0L) {
      downstream[i] <- as.integer(min(right) - a$end[i])
    }
  }
  nearest <- pmin(upstream, downstream, na.rm = TRUE)

  data.frame(upstream = upstream, downstream = downstream, nearest = nearest)
}

#' Remove the parts of intervals covered by others
#'
#' @param x An interval table with columns `chrom`, `start` and `end`.
#' @param y An interval table of regions to subtract.
#' @return A data frame with columns `chrom`, `start`, `end` and `source`,
#'   the row of `x` each remaining piece came from.
#' @export
subtract.intervals <- function(x, y) {
  y <- reduce.intervals(y)
  pieces <- list()

  for (i in seq_len(nrow(x))) {
    cut <- y[y$chrom == x$chrom[i] & y$start <= x$end[i] & y$end >= x$start[i], , drop = FALSE]
    pos <- x$start[i]
    for (j in seq_len(nrow(cut))) {
      # Keep the piece left of the cut
      if (cut$start[j] > pos) {
        pieces[[length(pieces) + 1L]] <- data.frame(
          chrom = x$chrom[i], start = pos, end = cut$start[j] - 1L, source = i
        )
      }
      pos <- max(pos, cut$end[j] + 1L)
    }
    # Keep the piece right of the last cut
    if (pos <= x$end[i]) {
      pieces[[length(pieces) + 1L]] <- data.frame(
        chrom = x$chrom[i], start = pos, end = x$end[i], source = i
      )
    }
  }
  if (length(pieces) == 0L) {
    return(data.frame(chrom = character(0), start = integer(0), end = integer(0),
                      source = integer(0)))
  }

  do.call(rbind, pieces)
}

#' Per-base coverage depth on one sequence
#'
#' @param x An interval table with columns `chrom`, `start` and `end`.
#' @param chrom The sequence to compute coverage on.
#' @param chrom.length The sequence length.
#' @param min.depth Runs below this depth are reported with depth 0.
#' @return A data frame of constant-depth runs with columns `start`, `end`
#'   and `depth`.
#' @export
coverage.depth <- function(x, chrom, chrom.length, min.depth = 1L) {
  # Validate `chrom`
  if (!is.character(chrom) || length(chrom) != 1L) {
    stop(sprintf("`%s` must be a single string.", "chrom"), call. = FALSE)
  }
  # Validate `chrom.length`
  if (!is.numeric(chrom.length) || length(chrom.length) != 1L || chrom.length < 1) {
    stop(sprintf("`%s` must be a single positive number.", "chrom.length"), call. = FALSE)
  }
  # Validate `min.depth`
  if (!is.numeric(min.depth) || length(min.depth) != 1L || min.depth < 1) {
    stop(sprintf("`%s` must be a single positive number.", "min.depth"), call. = FALSE)
  }
  x <- x[x$chrom == chrom, , drop = FALSE]
  levels <- seq_len(chrom.length + 1L)
  delta <- integer(chrom.length + 1L)

  # Each interval opens at its start
  opens <- table(factor(pmax(x$start, 1L), levels = levels))
  delta <- delta + as.integer(opens)
  # Each interval closes after its end
  closes <- table(factor(pmin(x$end, chrom.length) + 1L, levels = levels))
  delta <- delta - as.integer(closes)

  depth <- cumsum(delta)[seq_len(chrom.length)]
  depth[depth < min.depth] <- 0L
  runs <- rle(depth)
  ends <- cumsum(runs$lengths)
  data.frame(
    start = ends - runs$lengths + 1L,
    end = ends,
    depth = runs$values

  )
}
