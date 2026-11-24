#' Get a value from a nested list by path
#'
#' Paths use `.` between names and `[i]` for list positions, for example
#' `"owner.login"` or `"labels[1].name"`.
#'
#' @param x A nested list, as parsed from JSON.
#' @param path A single path string.
#' @param default Value returned when the path does not exist.
#' @return The value at `path`, or `default`.
#' @export
pluck.path <- function(x, path, default = NULL) {
  steps <- strsplit(gsub("\\[([0-9]+)\\]", ".[\\1]", path), ".", fixed = TRUE)[[1]]
  steps <- steps[nzchar(steps)]
  for (step in steps) {
    if (is.null(x)) {
      return(default)
    }
    if (grepl("^\\[[0-9]+\\]$", step)) {
      # Position step
      i <- as.integer(gsub("[^0-9]", "", step))
      x <- if (is.list(x) && i <= length(x)) x[[i]] else NULL
    } else {
      # Name step
      x <- if (is.list(x) && step %in% names(x)) x[[step]] else NULL
    }
  }
  if (is.null(x)) default else x
}

#' Extract a user record
#'
#' Pulls the fields `id`, `login`, `name`, `email`, `company`, `followers`
#' and `created_at` from one user object of the API.
#'
#' @param item One user object, a nested list.
#' @return A list with elements `id`, `login`, `name`, `email`, `company`,
#'   `followers` and `created_at`; missing fields are `NA` of the right type.
#' @export
extract.user <- function(item) {
  out <- list()
  # Field `id`
  value <- item[["id"]]
  out$id <- if (is.null(value)) NA_integer_ else as.integer(value)
  # Field `login`
  value <- item[["login"]]
  out$login <- if (is.null(value)) NA_character_ else as.character(value)
  # Field `name`
  value <- item[["name"]]
  out$name <- if (is.null(value)) NA_character_ else as.character(value)
  # Field `email`
  value <- item[["email"]]
  out$email <- if (is.null(value)) NA_character_ else tolower(value)
  # Field `company`: strip the leading @ of organisation handles
  value <- item[["company"]]
  out$company <- if (is.null(value)) NA_character_ else sub("^@", "", value)
  # Field `followers`
  value <- item[["followers"]]
  out$followers <- if (is.null(value)) NA_integer_ else as.integer(value)
  # Field `created_at`
  value <- item[["created_at"]]
  out$created_at <- if (is.null(value)) {
    as.POSIXct(NA)
  } else {
    as.POSIXct(value, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  }

  out
}

#' Extract a repository record
#'
#' Nested fields are flattened: `owner` is `owner.login`, `license` is
#' `license.spdx_id` and `topics` is collapsed to one string.
#'
#' @param item One repository object, a nested list.
#' @return A list with elements `id`, `full_name`, `owner`, `stars`,
#'   `forks`, `language`, `license`, `topics` and `archived`.
#' @export
extract.repo <- function(item) {
  out <- list()
  # Field `id`
  value <- pluck.path(item, "id")
  out$id <- if (is.null(value)) NA_integer_ else as.integer(value)
  # Field `full_name`
  value <- pluck.path(item, "full_name")
  out$full_name <- if (is.null(value)) NA_character_ else as.character(value)
  # Field `owner`
  value <- pluck.path(item, "owner.login")
  out$owner <- if (is.null(value)) NA_character_ else as.character(value)
  # Field `stars`
  value <- pluck.path(item, "stargazers_count")
  out$stars <- if (is.null(value)) NA_integer_ else as.integer(value)
  # Field `forks`
  value <- pluck.path(item, "forks_count")
  out$forks <- if (is.null(value)) NA_integer_ else as.integer(value)
  # Field `language`
  value <- pluck.path(item, "language")
  out$language <- if (is.null(value)) NA_character_ else as.character(value)
  # Field `license`
  value <- pluck.path(item, "license.spdx_id")
  out$license <- if (is.null(value)) NA_character_ else as.character(value)
  # Field `topics`
  value <- pluck.path(item, "topics")
  out$topics <- if (length(value) == 0L) NA_character_ else paste(unlist(value), collapse = ",")
  # Field `archived`
  value <- pluck.path(item, "archived")
  out$archived <- if (is.null(value)) NA else isTRUE(value)
  out
}

#' Extract an issue record
#'
#' @param item One issue object, a nested list.
#' @return A list with elements `number`, `title`, `state`, `author`,
#'   `labels`, `assignees`, `comments` and `closed_at`.
#' @export
extract.issue <- function(item) {
  out <- list()
  # Field `number`
  value <- pluck.path(item, "number")
  out$number <- if (is.null(value)) NA_integer_ else as.integer(value)
  # Field `title`
  value <- pluck.path(item, "title")
  out$title <- if (is.null(value)) NA_character_ else trimws(value)
  # Field `state`
  value <- pluck.path(item, "state")
  out$state <- if (is.null(value)) NA_character_ else tolower(value)
  # Field `author`
  value <- pluck.path(item, "user.login")
  out$author <- if (is.null(value)) NA_character_ else as.character(value)
  # Field `labels`: one name per label object
  value <- pluck.path(item, "labels")
  out$labels <- if (length(value) == 0L) NA_character_ else
    paste(vapply(value, function(l) l[["name"]], character(1)), collapse = ",")
  # Field `assignees`: one login per user object
  value <- pluck.path(item, "assignees")
  out$assignees <- if (length(value) == 0L) NA_character_ else
    paste(vapply(value, function(u) u[["login"]], character(1)), collapse = ",")
  # Field `comments`
  value <- pluck.path(item, "comments")
  out$comments <- if (is.null(value)) NA_integer_ else as.integer(value)
  # Field `closed_at`
  value <- pluck.path(item, "closed_at")
  out$closed_at <- if (is.null(value)) {
    as.POSIXct(NA)
  } else {
    as.POSIXct(value, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  }

  out
}

#' Extract typed columns from a list of items
#'
#' @param items A list of objects, each a nested list.
#' @param spec A named character vector: names are output columns, values
#'   are `"path:type"` with `type` one of `"chr"`, `"int"`, `"num"`,
#'   `"lgl"`, `"time"` or `"list"`.
#' @return A data frame with one row per item and one column per `spec`
#'   entry.
#' @export
extract.fields <- function(items, spec) {
  if (is.null(names(spec)) || any(!grepl(":", spec, fixed = TRUE))) {
    stop("`spec` must be a named vector of \"path:type\" strings.", call. = FALSE)
  }

  out <- data.frame(row.names = seq_along(items))
  for (col in names(spec)) {
    path <- sub(":[^:]*$", "", spec[[col]])
    type <- sub("^.*:", "", spec[[col]])
    values <- lapply(items, pluck.path, path = path)
    missing <- vapply(values, is.null, logical(1))
    values[missing] <- list(NA)
    out[[col]] <- switch(type,
      chr = {
        vapply(values, function(v) as.character(v)[1], character(1))
      },
      int = {
        vapply(values, function(v) as.integer(v)[1], integer(1))
      },
      num = {
        vapply(values, function(v) as.numeric(v)[1], numeric(1))
      },
      lgl = {
        vapply(values, function(v) as.logical(v)[1], logical(1))
      },
      time = {
        text <- vapply(values, function(v) as.character(v)[1], character(1))
        as.POSIXct(text, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
      },
      list = {
        I(values)
      },
      stop(sprintf("Unknown type `%s` in `spec` for column `%s`.", type, col), call. = FALSE)
    )
  }

  out
}
