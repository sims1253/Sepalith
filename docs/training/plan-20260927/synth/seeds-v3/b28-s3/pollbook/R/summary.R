#' Summarise a tally
#'
#' @param object An object of class `pb_tally`.
#' @param ... Unused.
#' @return An object of class `summary.pb_tally`: a list with elements
#'   `contest`, `n_votes`, `n_options`, `leader`, `margin` and `shares`.
#'   `leader` is `NA` when no votes were cast or the top options are tied.
#' @export
summary.pb_tally <- function(object, ...) {
  counts <- unclass(object)
  n.votes <- sum(counts)
  shares <- if (n.votes > 0L) counts / n.votes else counts * 0

  # Leader and margin over the runner-up
  ordered <- sort(counts, decreasing = TRUE)
  leader <- NA_character_
  margin <- 0L
  if (length(ordered) > 0L && n.votes > 0L) {
    runner.up <- if (length(ordered) > 1L) ordered[[2L]] else 0L
    margin <- ordered[[1L]] - runner.up
    if (margin > 0L) {
      leader <- names(ordered)[1L]
    }
  }

  out <- list(
    contest = attr(object, "contest"),
    n_votes = n.votes,
    n_options = length(counts),
    leader = leader,
    margin = margin,
    shares = c(shares)
  )
  structure(out, class = "summary.pb_tally")
}

#' Summarise a roster
#'
#' @param object An object of class `pb_roster`.
#' @param ... Unused.
#' @return An object of class `summary.pb_roster`: a list with elements
#'   `n_voters`, `n_districts`, `by_status` and `by_district`, the last two
#'   being named integer counts.
#' @export
summary.pb_roster <- function(object, ...) {
  fields <- unclass(object)
  n.voters <- length(fields$voter)

  # Counts per status and per district, in sorted order
  by.status <- table(fields$status)
  by.status <- stats::setNames(as.integer(by.status), names(by.status))
  by.district <- table(fields$district)
  by.district <- stats::setNames(as.integer(by.district), names(by.district))

  out <- list(
    n_voters = n.voters,
    n_districts = length(by.district),
    by_status = by.status,
    by_district = by.district
  )
  structure(out, class = "summary.pb_roster")
}

#' Summarise ranked ballots
#'
#' @param object An object of class `pb_ballot`.
#' @param ... Unused.
#' @return An object of class `summary.pb_ballot`: a list with elements
#'   `n_ballots`, `n_options`, `n_blank`, `mean_length` and `first_choice`,
#'   the last being a named integer count per option.
#' @export
summary.pb_ballot <- function(object, ...) {
  rankings <- unclass(object)
  options <- attr(object, "options")
  lengths <- vapply(rankings, length, integer(1))

  # First preferences of the non-blank ballots
  firsts <- vapply(rankings[lengths > 0L], `[`, character(1), 1L)
  first.choice <- tabulate(match(firsts, options), nbins = length(options))
  names(first.choice) <- options

  out <- list(
    n_ballots = length(rankings),
    n_options = length(options),
    n_blank = sum(lengths == 0L),
    mean_length = if (length(lengths) > 0L) mean(lengths) else NA_real_,
    first_choice = first.choice
  )
  structure(out, class = "summary.pb_ballot")
}

#' Print summaries
#'
#' @param x A summary object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @rdname print.summary
#' @export
print.summary.pb_tally <- function(x, ...) {
  title <- if (nzchar(x$contest)) x$contest else "(unnamed contest)"
  cat("Tally: ", title, "\n", sep = "")
  cat("  votes:   ", x$n_votes, "\n", sep = "")
  cat("  options: ", x$n_options, "\n", sep = "")
  if (is.na(x$leader)) {
    cat("  leader:  none\n")
  } else {
    cat("  leader:  ", x$leader, " by ", x$margin, "\n", sep = "")
  }
  if (x$n_votes > 0L) {
    pct <- sprintf("%.1f%%", 100 * x$shares)
    cat(paste0("    ", formatC(names(x$shares), width = -10L), pct), sep = "\n")
  }
  invisible(x)
}

#' @rdname print.summary
#' @export
print.summary.pb_roster <- function(x, ...) {
  cat("Roster\n")
  cat("  voters:    ", x$n_voters, "\n", sep = "")
  cat("  districts: ", x$n_districts, "\n", sep = "")
  if (x$n_voters > 0L) {
    cat("  by status:\n")
    cat(paste0("    ", formatC(names(x$by_status), width = -10L), x$by_status), sep = "\n")
    cat("  by district:\n")
    cat(paste0("    ", formatC(names(x$by_district), width = -10L), x$by_district),
      sep = "\n")
  }
  invisible(x)
}

#' @rdname print.summary
#' @export
print.summary.pb_ballot <- function(x, ...) {
  cat("Ballots\n")
  cat("  ballots: ", x$n_ballots, "\n", sep = "")
  cat("  options: ", x$n_options, "\n", sep = "")
  cat("  blank:   ", x$n_blank, "\n", sep = "")
  if (x$n_ballots > 0L) {
    cat("  mean ranked: ", format(x$mean_length, digits = 3), "\n", sep = "")
    cat("  first choices:\n")
    cat(paste0("    ", formatC(names(x$first_choice), width = -10L), x$first_choice),
      sep = "\n")
  }
  invisible(x)
}
