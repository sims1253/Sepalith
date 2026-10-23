#' Format pollbook objects
#'
#' @description One string per element:
#'   * tally: `option: count (share%)`;
#'   * roster: `voter (district, status)`;
#'   * ballot: options joined by `" > "`, or `(blank)` for a blank ballot.
#'
#' @param x A `pb_tally`, `pb_roster` or `pb_ballot` object.
#' @param ... Unused.
#' @return A character vector.
#' @rdname format.pollbook
#' @export
format.pb_tally <- function(x, ...) {
  counts <- unclass(x)
  if (length(counts) == 0L) {
    return(character())
  }
  total <- sum(counts)
  share <- if (total > 0L) 100 * counts / total else counts * 0
  sprintf("%s: %d (%.1f%%)", names(counts), as.integer(counts), share)
}

#' @rdname format.pollbook
#' @export
format.pb_roster <- function(x, ...) {
  fields <- unclass(x)
  if (length(fields$voter) == 0L) {
    return(character())
  }
  sprintf("%s (%s, %s)", fields$voter, fields$district, fields$status)
}

#' @rdname format.pollbook
#' @export
format.pb_ballot <- function(x, ...) {
  rankings <- unclass(x)
  if (length(rankings) == 0L) {
    return(character())
  }
  out <- vapply(rankings, paste, character(1), collapse = " > ")
  out[out == ""] <- "(blank)"
  out
}

#' Print pollbook objects
#'
#' Each method prints a `<class: detail>` header followed by at most `n`
#' formatted elements, one per line.
#'
#' @inheritParams format.pb_tally
#' @param n Maximum number of elements to show.
#' @return `x`, invisibly.
#' @rdname print.pollbook
#' @export
print.pb_tally <- function(x, ..., n = 10L) {
  lines <- format(x, ...)
  contest <- attr(x, "contest")
  cat("<pb_tally: ", if (nzchar(contest)) contest else "unnamed", ">\n", sep = "")
  if (length(lines) > n) {
    lines <- c(lines[seq_len(n)], sprintf("# ... with %d more", length(lines) - n))
  }
  if (length(lines) > 0L) {
    cat(paste0("  ", lines), sep = "\n")
  }
  invisible(x)
}

#' @rdname print.pollbook
#' @export
print.pb_roster <- function(x, ..., n = 10L) {
  lines <- format(x, ...)
  districts <- length(unique(unclass(x)$district))
  cat("<pb_roster: ", length(lines), " voters in ", districts, " districts>\n", sep = "")
  if (length(lines) > n) {
    lines <- c(lines[seq_len(n)], sprintf("# ... with %d more", length(lines) - n))
  }
  if (length(lines) > 0L) {
    cat(paste0("  ", lines), sep = "\n")
  }
  invisible(x)
}

#' @rdname print.pollbook
#' @export
print.pb_ballot <- function(x, ..., n = 10L) {
  lines <- format(x, ...)
  options <- attr(x, "options")
  cat("<pb_ballot: ", length(lines), " ballots over ", length(options), " options>\n", sep = "")
  if (length(lines) > n) {
    lines <- c(lines[seq_len(n)], sprintf("# ... with %d more", length(lines) - n))
  }
  if (length(lines) > 0L) {
    cat(paste0("  ", lines), sep = "\n")
  }
  invisible(x)
}
