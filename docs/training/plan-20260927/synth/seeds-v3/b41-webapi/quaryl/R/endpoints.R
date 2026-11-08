#' Path of an API endpoint
#'
#' @param resource One of `"user"`, `"repo"`, `"issues"`, `"issue"`,
#'   `"comments"` or `"releases"`.
#' @param owner Account name; needed by every resource.
#' @param repo Repository name; needed by every resource except `"user"`.
#' @param number Issue number; needed by `"issue"` and `"comments"`.
#' @return A character vector of path segments.
#' @export
endpoint_path <- function(resource = c("user", "repo", "issues", "issue", "comments",
                                       "releases"),
                          owner, repo = NULL, number = NULL) {
  resource <- match.arg(resource)
  require_arg <- function(value, name) {
    if (is.null(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
      stop(sprintf("Resource `%s` needs `%s`.", resource, name), call. = FALSE)
    }
    as.character(value)
  }
  owner <- require_arg(owner, "owner")

  path <- switch(resource,
    user = {
      c("users", owner)
    },
    repo = {
      repo <- require_arg(repo, "repo")
      c("repos", owner, repo)
    },
    issues = {
      repo <- require_arg(repo, "repo")
      c("repos", owner, repo, "issues")
    },
    issue = {
      repo <- require_arg(repo, "repo")
      number <- require_arg(number, "number")
      c("repos", owner, repo, "issues", number)
    },
    comments = {
      repo <- require_arg(repo, "repo")
      number <- require_arg(number, "number")
      c("repos", owner, repo, "issues", number, "comments")
    },
    releases = {
      repo <- require_arg(repo, "repo")
      c("repos", owner, repo, "releases")
    }
  )
  return(path)
}

#' Default query parameters of an endpoint
#'
#' @param resource One of `"user"`, `"repo"`, `"issues"`, `"issue"`,
#'   `"comments"` or `"releases"`.
#' @param per_page Page size used by the listing resources.
#' @return A named list of query parameters, empty for single-object
#'   resources.
#' @export
endpoint_defaults <- function(resource = c("user", "repo", "issues", "issue", "comments",
                                           "releases"),
                              per_page = 100L) {
  resource <- match.arg(resource)
  defaults <- switch(resource,
    user = {
      list()
    },
    repo = {
      list()
    },
    issues = {
      list(state = "all", sort = "created", direction = "asc", per_page = per_page)
    },
    issue = {
      list()
    },
    comments = {
      list(sort = "created", direction = "asc", per_page = per_page)
    },
    releases = {
      list(per_page = min(per_page, 50L))
    }
  )
  return(defaults)
}

#' Recognise the resource behind an API URL
#'
#' The inverse of [endpoint_path()]: patterns are tried from the most
#' specific (`comments`) to the least specific (`user`).
#'
#' @param url A request URL.
#' @return A list with elements `resource`, `owner`, `repo` and `number`;
#'   unknown parts are `NA`. `resource` is `NA` when nothing matches.
#' @export
parse_endpoint_url <- function(url) {
  path <- sub("^https?://[^/]+", "", sub("[?#].*$", "", url))
  out <- list(resource = NA_character_, owner = NA_character_, repo = NA_character_,
    number = NA_integer_)
  # Resource `comments`
  m <- regmatches(path, regexec("/repos/([^/]+)/([^/]+)/issues/([0-9]+)/comments/?$", path))[[1]]
  if (length(m) > 0L) {
    out[c("resource", "owner", "repo")] <- list("comments", m[2], m[3])
    out$number <- as.integer(m[4])
    return(out)
  }

  # Resource `issue`
  m <- regmatches(path, regexec("/repos/([^/]+)/([^/]+)/issues/([0-9]+)/?$", path))[[1]]
  if (length(m) > 0L) {
    out[c("resource", "owner", "repo")] <- list("issue", m[2], m[3])
    out$number <- as.integer(m[4])
    return(out)
  }

  # Resource `issues`
  m <- regmatches(path, regexec("/repos/([^/]+)/([^/]+)/issues/?$", path))[[1]]
  if (length(m) > 0L) {
    out[c("resource", "owner", "repo")] <- list("issues", m[2], m[3])
    return(out)
  }

  # Resource `releases`
  m <- regmatches(path, regexec("/repos/([^/]+)/([^/]+)/releases/?$", path))[[1]]
  if (length(m) > 0L) {
    out[c("resource", "owner", "repo")] <- list("releases", m[2], m[3])
    return(out)
  }

  # Resource `repo`
  m <- regmatches(path, regexec("/repos/([^/]+)/([^/]+)/?$", path))[[1]]
  if (length(m) > 0L) {
    out[c("resource", "owner", "repo")] <- list("repo", m[2], m[3])
    return(out)
  }

  # Resource `user`
  m <- regmatches(path, regexec("/users/([^/]+)/?$", path))[[1]]
  if (length(m) > 0L) {
    out[c("resource", "owner")] <- list("user", m[2])
    return(out)
  }

  return(out)
}

#' Cache lifetime of an endpoint
#'
#' Listings change more often than single objects, and closed issues hardly
#' change at all.
#'
#' @param resource One of `"user"`, `"repo"`, `"issues"`, `"issue"`,
#'   `"comments"` or `"releases"`.
#' @param state Issue state, `"open"` or `"closed"`; only used by `"issue"`.
#' @param base Cache lifetime in seconds of the most volatile resource.
#' @return The lifetime in seconds.
#' @export
endpoint_cache_ttl <- function(resource = c("user", "repo", "issues", "issue", "comments",
                                            "releases"),
                               state = "open", base = 60) {
  resource <- match.arg(resource)
  multipliers <- c(user = 60, repo = 10, issues = 1, issue = 5, comments = 1, releases = 30)

  ttl <- switch(resource,
    user = {
      base * multipliers[["user"]]
    },
    repo = {
      base * multipliers[["repo"]]
    },
    issues = {
      base * multipliers[["issues"]]
    },
    issue = {
      factor <- if (identical(state, "closed")) multipliers[["user"]] else multipliers[["issue"]]
      base * factor
    },
    comments = {
      base * multipliers[["comments"]]
    },
    releases = {
      base * multipliers[["releases"]]
    }
  )
  return(ttl)
}

#' Describe a batch of endpoint calls
#'
#' @param urls Character vector of request URLs.
#' @return A data frame with one row per resource (`user`, `repo`,
#'   `issues`, `issue`, `comments`, `releases`) and columns `resource`,
#'   `calls` and `owners` (number of distinct owners).
#' @export
endpoint_usage <- function(urls) {
  resources <- c("user", "repo", "issues", "issue", "comments", "releases")
  parsed <- lapply(urls, parse_endpoint_url)
  resource <- vapply(parsed, function(p) p$resource, character(1))
  owner <- vapply(parsed, function(p) p$owner, character(1))
  calls <- integer(length(resources))
  owners <- integer(length(resources))
  names(calls) <- names(owners) <- resources
  for (r in resources) {
    hit <- !is.na(resource) & resource == r
    calls[[r]] <- sum(hit)
    owners[[r]] <- length(unique(owner[hit]))
  }
  unmatched <- sum(is.na(resource))
  if (unmatched > 0L) {
    warning(sprintf("%d URL(s) did not match any resource.", unmatched), call. = FALSE)
  }

  return(data.frame(resource = resources, calls = unname(calls), owners = unname(owners)))
}
