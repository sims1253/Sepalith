#' Run a git command and capture its output
#'
#' @param args A character vector of arguments passed to `git`.
#' @param repo Path to the git repository.
#' @return A character vector of output lines.
#' @keywords internal
git_run <- function(args, repo = ".") {
  git <- Sys.which("git")
  if (!nzchar(git)) {
    stop("Could not find a `git` executable on the PATH.", call. = FALSE)
  }
  out <- suppressWarnings(
    system2(git, c("-C", shQuote(repo), args), stdout = TRUE, stderr = TRUE)
  )
  status <- attr(out, "status")
  if (!is.null(status) && status != 0L) {
    stop(
      "`git ", paste(args, collapse = " "), "` failed with status ", status, ":\n",
      paste(out, collapse = "\n"),
      call. = FALSE
    )
  }
  return(out)
}

#' Read the commit log of a repository
#'
#' @param repo Path to the git repository.
#' @param range A revision range such as `"v1.0.0..HEAD"`, or `NULL` for the
#'   full history.
#' @param max_count Maximum number of commits to read.
#' @return A data frame with columns `hash`, `author`, `email`, `date` and
#'   `subject`, newest first.
#' @export
git_log <- function(repo = ".", range = NULL, max_count = 500L) {
  sep <- "\x1f"
  fmt <- paste0("--pretty=format:", paste("%H", "%an", "%ae", "%aI", "%s", sep = "%x1f"))
  args <- c("log", fmt, paste0("--max-count=", as.integer(max_count)), range)
  lines <- git_run(args, repo = repo)
  lines <- lines[nzchar(lines)]
  if (length(lines) == 0L) {
    return(data.frame(
      hash = character(), author = character(), email = character(),
      date = as.POSIXct(character()), subject = character(),
      stringsAsFactors = FALSE
    ))
  }
  parts <- strsplit(lines, sep, fixed = TRUE)
  field <- function(i) vapply(parts, function(p) p[[i]], character(1))
  out <- data.frame(
    hash = field(1L),
    author = field(2L),
    email = field(3L),
    date = as.POSIXct(sub("([+-][0-9]{2}):([0-9]{2})$", "\\1\\2", field(4L)),
      format = "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"
    ),
    subject = field(5L),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' List the tags of a repository, newest first
#'
#' @param repo Path to the git repository.
#' @param pattern Glob pattern restricting the tags, e.g. `"v*"`.
#' @return A character vector of tag names.
#' @export
git_tags <- function(repo = ".", pattern = "v*") {
  tags <- git_run(c("tag", "--list", shQuote(pattern), "--sort=-creatordate"), repo = repo)
  return(tags[nzchar(tags)])
}

#' Read the commits made since the most recent tag
#'
#' @param repo Path to the git repository.
#' @param pattern Glob pattern restricting the tags, e.g. `"v*"`.
#' @return A data frame of commits as returned by [git_log()].
#' @export
git_log_since_tag <- function(repo = ".", pattern = "v*") {
  tags <- git_tags(repo = repo, pattern = pattern)
  if (length(tags) == 0L) {
    message("No tags match '", pattern, "'; reading the full history.")
    return(git_log(repo = repo))
  }
  return(git_log(repo = repo, range = paste0(tags[[1L]], "..HEAD")))
}

#' Summarise a commit log by author
#'
#' @param log A data frame of commits as returned by [git_log()].
#' @return A data frame with one row per author: `author`, `commits`,
#'   `first` and `last`, ordered by commit count.
#' @family log summaries
#' @export
summarise_by_author <- function(log) {
  keys <- sort(unique(log$author))
  out <- data.frame(
    author = keys,
    commits = integer(length(keys)),
    first = as.POSIXct(rep(NA, length(keys)), tz = "UTC"),
    last = as.POSIXct(rep(NA, length(keys)), tz = "UTC"),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(keys)) {
    dates <- log$date[log$author == keys[[i]]]
    out$commits[[i]] <- length(dates)
    out$first[[i]] <- min(dates)
    out$last[[i]] <- max(dates)
  }
  return(out[order(-out$commits, out$author), , drop = FALSE])
}

#' Summarise a commit log by ISO week
#'
#' @param log A data frame of commits as returned by [git_log()].
#' @return A data frame with one row per week: `week`, `commits`,
#'   `authors` and `subjects`, ordered by week.
#' @family log summaries
#' @export
summarise_by_week <- function(log) {
  week <- format(log$date, "%G-W%V")
  keys <- sort(unique(week))
  out <- data.frame(
    week = keys,
    commits = integer(length(keys)),
    authors = integer(length(keys)),
    subjects = character(length(keys)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(keys)) {
    rows <- log[week == keys[[i]], , drop = FALSE]
    out$commits[[i]] <- nrow(rows)
    out$authors[[i]] <- length(unique(rows$author))
    out$subjects[[i]] <- paste(utils::head(rows$subject, 3L), collapse = "; ")
  }
  return(out)
}

#' Summarise the size of a diff
#'
#' Parses the final line of `git diff --shortstat`.
#'
#' @param range A revision range such as `"v1.0.0..HEAD"`.
#' @param repo Path to the git repository.
#' @return A named integer vector with elements `files`, `insertions` and
#'   `deletions`.
#' @export
git_diff_stat <- function(range, repo = ".") {
  out <- git_run(c("diff", "--shortstat", range), repo = repo)
  stat <- c(files = 0L, insertions = 0L, deletions = 0L)
  if (length(out) == 0L) {
    return(stat)
  }
  line <- out[[length(out)]]
  patterns <- c(
    files = "([0-9]+) files? changed",
    insertions = "([0-9]+) insertions?\\(\\+\\)",
    deletions = "([0-9]+) deletions?\\(-\\)"
  )
  for (key in names(patterns)) {
    hit <- regmatches(line, regexec(patterns[[key]], line))[[1L]]
    if (length(hit) == 2L) {
      stat[[key]] <- as.integer(hit[[2L]])
    }
  }
  return(stat)
}
