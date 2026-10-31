#' Number of voters on a roster
#'
#' @param x An object of class `pb_roster`.
#' @return A single integer.
#' @export
length.pb_roster <- function(x) {
  length(unclass(x)$voter)
}

#' Number of ballots
#'
#' @param x An object of class `pb_ballot`.
#' @return A single integer.
#' @export
length.pb_ballot <- function(x) {
  length(unclass(x))
}

#' Subset a tally
#'
#' @param x An object of class `pb_tally`.
#' @param i Positions, a logical vector, or option names.
#' @return An object of class `pb_tally` with the same contest.
#' @export
`[.pb_tally` <- function(x, i) {
  if (missing(i)) {
    return(x)
  }
  counts <- unclass(x)
  if (is.character(i) && !all(i %in% names(counts))) {
    stop(sprintf("Unknown option '%s'.", setdiff(i, names(counts))[1L]), call. = FALSE)
  }
  pb.new.tally(counts[i], attr(x, "contest"))
}

#' Subset a roster
#'
#' @param x An object of class `pb_roster`.
#' @param i Positions, a logical vector, or voter ids.
#' @return An object of class `pb_roster`.
#' @export
`[.pb_roster` <- function(x, i) {
  if (missing(i)) {
    return(x)
  }
  fields <- unclass(x)
  if (is.character(i)) {
    i <- match(i, fields$voter)
  }
  if (anyNA(i)) {
    stop("Subscript out of bounds.", call. = FALSE)
  }
  pb.new.roster(fields$voter[i], fields$district[i], fields$status[i])
}

#' Subset ballots
#'
#' @param x An object of class `pb_ballot`.
#' @param i Positions or a logical vector.
#' @return An object of class `pb_ballot` over the same options.
#' @export
`[.pb_ballot` <- function(x, i) {
  if (missing(i)) {
    return(x)
  }
  rankings <- unclass(x)
  if (is.character(i)) {
    stop("Ballots have no names to subset by.", call. = FALSE)
  }
  if (anyNA(i)) {
    stop("Subscript out of bounds.", call. = FALSE)
  }
  structure(rankings[i], options = attr(x, "options"), class = "pb_ballot")
}

#' Combine and compare tallies
#'
#' Both tallies are first aligned on the union of their options, with zero
#' counts for options missing on one side. `+` adds the counts, `-` removes
#' the counts of `e2` from `e1`, and `==` and `!=` compare counts per option.
#' The result keeps the contest of `e1`.
#'
#' @param e1,e2 Objects of class `pb_tally`.
#' @return A `pb_tally` for `+` and `-`, a named logical vector otherwise.
#' @export
Ops.pb_tally <- function(e1, e2) {
  if (nargs() == 1L || !(inherits(e1, "pb_tally") && inherits(e2, "pb_tally"))) {
    stop(sprintf("`%s` needs a <%s> on both sides.", .Generic, "pb_tally"), call. = FALSE)
  }
  lhs <- unclass(e1)
  rhs <- unclass(e2)
  options <- union(names(lhs), names(rhs))

  # Align both sides on the union of options
  left <- stats::setNames(integer(length(options)), options)
  left[names(lhs)] <- lhs
  right <- stats::setNames(integer(length(options)), options)
  right[names(rhs)] <- rhs

  contest <- attr(e1, "contest")
  switch(.Generic,
    "+" = pb.new.tally(left + right, contest),
    "-" = {
      short <- names(which(right > left))
      if (length(short) > 0L) {
        stop(sprintf("Can't remove more votes than '%s' has.", short[1L]), call. = FALSE)
      }
      pb.new.tally(left - right, contest)
    },
    "==" = left == right,
    "!=" = left != right,
    stop(sprintf("`%s` is not supported for <%s>.", .Generic, "pb_tally"), call. = FALSE)
  )
}

#' Convert a roster to a data frame
#'
#' @param x An object of class `pb_roster`.
#' @param row.names Optional row names for the result.
#' @param optional Ignored.
#' @param ... Unused.
#' @return A data frame with columns `voter`, `district` and `status`.
#' @export
as.data.frame.pb_roster <- function(x, row.names = NULL, optional = FALSE, ...) {
  fields <- unclass(x)
  out <- data.frame(
    voter = fields$voter,
    district = fields$district,
    status = fields$status,
    stringsAsFactors = FALSE
  )
  if (!is.null(row.names)) {
    rownames(out) <- row.names
  }
  out
}

#' Convert ballots to a data frame
#'
#' Blank ballots contribute no rows.
#'
#' @param x An object of class `pb_ballot`.
#' @param row.names Ignored; the result has one row per ranked option.
#' @param optional Ignored.
#' @param ... Unused.
#' @return A data frame in long form with columns `ballot`, `rank` and
#'   `option`.
#' @export
as.data.frame.pb_ballot <- function(x, row.names = NULL, optional = FALSE, ...) {
  rankings <- unclass(x)
  lengths <- vapply(rankings, length, integer(1))
  data.frame(
    ballot = rep(seq_along(rankings), lengths),
    rank = sequence(lengths),
    option = as.character(unlist(rankings, use.names = FALSE)),
    stringsAsFactors = FALSE
  )
}
