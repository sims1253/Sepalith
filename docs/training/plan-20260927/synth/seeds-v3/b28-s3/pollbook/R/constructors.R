#' Create a tally without checks
#'
#' @param counts A named integer vector of vote counts, one per option.
#' @param contest A single string naming the contest.
#' @return An object of class `pb_tally`.
#' @keywords internal
pb.new.tally <- function(counts = integer(), contest = "") {
  stopifnot(is.integer(counts), is.character(contest))
  structure(counts, contest = contest, class = "pb_tally")
}

#' Validate a tally
#'
#' @details Checks, in order: 1. `contest` is a single non-missing string;
#'   2. every count is named after a distinct option; 3. counts are
#'   non-negative and not missing.
#'
#' @param x An object of class `pb_tally`.
#' @return `x`, unchanged, if it is valid.
#' @keywords internal
pb.validate.tally <- function(x) {
  counts <- unclass(x)
  contest <- attr(x, "contest")
  # 1. contest
  if (!is.character(contest) || length(contest) != 1L || is.na(contest)) {
    stop("`contest` must be a single non-missing string.", call. = FALSE)
  }
  # 2. option names
  opts <- names(counts)
  if (length(counts) > 0L && (is.null(opts) || anyNA(opts) || any(opts == ""))) {
    stop("Every count must be named after its option.", call. = FALSE)
  }
  if (anyDuplicated(opts) > 0L) {
    stop(sprintf("Option '%s' appears more than once.", opts[anyDuplicated(opts)]),
      call. = FALSE)
  }
  # 3. counts
  if (anyNA(counts) || any(counts < 0L)) {
    stop("Counts must be non-negative and not missing.", call. = FALSE)
  }
  x
}

#' Tally single-choice votes
#'
#' @param votes A character vector with one chosen option per voter;
#'   missing values are abstentions and are not counted.
#' @param options The options on the ballot. Defaults to the sorted distinct
#'   non-missing `votes`.
#' @param contest A single string naming the contest.
#' @return A validated object of class `pb_tally`.
#' @export
pb.tally <- function(votes, options = NULL, contest = "") {
  votes <- as.character(votes)
  cast <- votes[!is.na(votes)]
  if (is.null(options)) {
    options <- sort(unique(cast))
  }
  unknown <- setdiff(cast, options)
  if (length(unknown) > 0L) {
    stop(sprintf("Vote for unknown option '%s'.", unknown[1L]), call. = FALSE)
  }
  counts <- tabulate(match(cast, options), nbins = length(options))
  names(counts) <- options
  x <- pb.new.tally(counts, contest)
  pb.validate.tally(x)
}

#' Create a roster without checks
#'
#' @param voter A character vector of voter ids.
#' @param district A character vector of district codes.
#' @param status A character vector of registration statuses.
#' @return An object of class `pb_roster`.
#' @keywords internal
pb.new.roster <- function(voter = character(), district = character(),
                          status = character()) {
  stopifnot(is.character(voter), is.character(district), is.character(status))
  structure(list(voter = voter, district = district, status = status),
    class = "pb_roster")
}

#' Validate a roster
#'
#' @details `voter`, `district` and `status` are checked in turn: each must
#'   be a character vector without missing values and as long as `voter`.
#'   Voter ids must be unique and every status one of `statuses`.
#'
#' @param x An object of class `pb_roster`.
#' @param statuses The allowed registration statuses.
#' @return `x`, unchanged, if it is valid.
#' @keywords internal
pb.validate.roster <- function(x, statuses = c("active", "inactive", "pending")) {
  fields <- unclass(x)
  n <- length(fields$voter)

  # Check `voter`
  if (!is.character(fields$voter) || anyNA(fields$voter)) {
    stop("`voter` must be a character vector without missing values.", call. = FALSE)
  }
  if (anyDuplicated(fields$voter) > 0L) {
    stop(sprintf("Voter '%s' is listed twice.", fields$voter[anyDuplicated(fields$voter)]),
      call. = FALSE)
  }

  # Check `district`
  if (!is.character(fields$district) || anyNA(fields$district)) {
    stop("`district` must be a character vector without missing values.", call. = FALSE)
  }
  if (length(fields$district) != n) {
    stop("`district` must have one element per voter.", call. = FALSE)
  }

  # Check `status`
  if (!is.character(fields$status) || anyNA(fields$status)) {
    stop("`status` must be a character vector without missing values.", call. = FALSE)
  }
  if (length(fields$status) != n) {
    stop("`status` must have one element per voter.", call. = FALSE)
  }
  bad <- setdiff(fields$status, statuses)
  if (length(bad) > 0L) {
    stop(sprintf("Unknown status '%s'.", bad[1L]), call. = FALSE)
  }
  x
}

#' Create a voter roster
#'
#' @param voter A character vector of voter ids.
#' @param district A character vector of district codes, recycled if of
#'   length one.
#' @param status A character vector of registration statuses, recycled if of
#'   length one.
#' @return A validated object of class `pb_roster`.
#' @export
pb.roster <- function(voter, district, status = "active") {
  # Coerce `voter`; its length sets the roster size
  if (is.factor(voter)) {
    voter <- levels(voter)[voter]
  }
  voter <- as.character(voter)
  n <- length(voter)

  # Coerce and recycle `district`
  if (is.factor(district)) {
    district <- levels(district)[district]
  }
  district <- as.character(district)
  if (length(district) == 1L) {
    district <- rep(district, n)
  }

  # Coerce and recycle `status`
  if (is.factor(status)) {
    status <- levels(status)[status]
  }
  status <- as.character(status)
  if (length(status) == 1L) {
    status <- rep(status, n)
  }

  x <- pb.new.roster(voter, district, status)
  pb.validate.roster(x)
}

#' Validate ranked ballots
#'
#' @details `options` must be a character vector of distinct, non-missing
#'   options. Each ballot must be a character vector that ranks only known
#'   options, each at most once; an empty ballot is a blank ballot.
#'
#' @param x An object of class `pb_ballot`.
#' @return `x`, unchanged, if it is valid.
#' @keywords internal
pb.validate.ballot <- function(x) {
  rankings <- unclass(x)
  options <- attr(x, "options")

  # Check `options`
  if (!is.character(options) || length(options) == 0L) {
    stop("`options` must be a non-empty character vector.", call. = FALSE)
  }
  if (anyNA(options) || anyDuplicated(options) > 0L) {
    stop("`options` must be distinct and not missing.", call. = FALSE)
  }

  # Check each ballot
  for (i in seq_along(rankings)) {
    ranked <- rankings[[i]]
    if (!is.character(ranked) || anyNA(ranked)) {
      stop(sprintf("Ballot %d must be a character vector without missing values.", i),
        call. = FALSE)
    }
    unknown <- setdiff(ranked, options)
    if (length(unknown) > 0L) {
      stop(sprintf("Ballot %d ranks unknown option '%s'.", i, unknown[1L]), call. = FALSE)
    }
    if (anyDuplicated(ranked) > 0L) {
      stop(sprintf("Ballot %d ranks '%s' twice.", i, ranked[anyDuplicated(ranked)]),
        call. = FALSE)
    }
  }
  x
}

#' Create ranked ballots
#'
#' Each ballot is a character vector of options in order of preference; an
#' empty vector is a blank ballot.
#'
#' @param rankings A list of character vectors, one per ballot.
#' @param options The options on the ballot.
#' @return A validated object of class `pb_ballot`.
#' @export
pb.ballot <- function(rankings, options) {
  if (!is.list(rankings)) {
    stop("`rankings` must be a list with one element per ballot.", call. = FALSE)
  }
  rankings <- lapply(unname(rankings), as.character)
  x <- structure(rankings, options = as.character(options), class = "pb_ballot")
  pb.validate.ballot(x)
}
