#' Build the query list for a search request
#'
#' Each argument becomes one query parameter; `NULL` arguments are dropped.
#'
#' @param q Search text.
#' @param page Page number, starting at 1.
#' @param per_page Number of results per page, between 1 and 100.
#' @param sort Field to sort by.
#' @param order Sort direction, `"asc"` or `"desc"`.
#' @param since A `Date` or `POSIXct`; only items updated after it are returned.
#' @param fields Character vector of fields to include in each item.
#' @return A named list of character scalars with the elements `q`, `page`,
#'   `per_page`, `sort`, `order`, `since` and `fields` that were supplied.
#' @export
build_search_query <- function(q = NULL, page = NULL, per_page = NULL, sort = NULL,
                               order = NULL, since = NULL, fields = NULL) {
  query <- list()
  # Parameter `q`
  if (!is.null(q)) {
    if (!is.character(q) || length(q) != 1L || !nzchar(q)) {
      stop(sprintf("`%s` must be a single non-empty string.", "q"), call. = FALSE)
    }
    query[["q"]] <- trimws(q)
  }
  # Parameter `page`
  if (!is.null(page)) {
    if (!is.numeric(page) || length(page) != 1L || page < 1) {
      stop(sprintf("`%s` must be a single number of at least %d.", "page", 1L), call. = FALSE)
    }
    query[["page"]] <- format(as.integer(page))
  }
  # Parameter `per_page`
  if (!is.null(per_page)) {
    if (!is.numeric(per_page) || length(per_page) != 1L || per_page < 1 || per_page > 100) {
      stop(sprintf("`%s` must be a single number between %d and %d.", "per_page", 1L, 100L),
        call. = FALSE)
    }
    query[["per_page"]] <- format(as.integer(per_page))
  }
  # Parameter `sort`
  if (!is.null(sort)) {
    if (!is.character(sort) || length(sort) != 1L || !nzchar(sort)) {
      stop(sprintf("`%s` must be a single non-empty string.", "sort"), call. = FALSE)
    }
    query[["sort"]] <- trimws(sort)
  }
  # Parameter `order`
  if (!is.null(order)) {
    if (!is.character(order) || length(order) != 1L || !order %in% c("asc", "desc")) {
      stop(sprintf("`%s` must be one of %s.", "order", "\"asc\", \"desc\""), call. = FALSE)
    }
    query[["order"]] <- order
  }
  # Parameter `since`
  if (!is.null(since)) {
    if (!inherits(since, c("Date", "POSIXct")) || length(since) != 1L) {
      stop(sprintf("`%s` must be a single Date or POSIXct.", "since"), call. = FALSE)
    }
    query[["since"]] <- format(as.POSIXct(since, tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  }
  # Parameter `fields`
  if (!is.null(fields)) {
    if (!is.character(fields) || length(fields) == 0L || anyNA(fields)) {
      stop(sprintf("`%s` must be a non-empty character vector.", "fields"), call. = FALSE)
    }
    query[["fields"]] <- paste(unique(fields), collapse = ",")
  }

  return(query)
}

#' Encode one query parameter as URL text
#'
#' Vectors are serialised according to `style`, following the common
#' OpenAPI conventions.
#'
#' @param name Parameter name.
#' @param value An atomic vector of values.
#' @param style One of `"explode"` (`id=1&id=2`), `"comma"` (`id=1,2`),
#'   `"space"` (`id=1%202`), `"pipe"` (`id=1|2`) or `"brackets"`
#'   (`id[]=1&id[]=2`).
#' @return A single string, or `""` when `value` has no non-missing values.
#' @keywords internal
encode_query_pair <- function(name, value,
                              style = c("explode", "comma", "space", "pipe", "brackets")) {
  style <- match.arg(style)
  if (!is.character(name) || length(name) != 1L || !nzchar(name)) {
    stop("`name` must be a single non-empty string.", call. = FALSE)
  }
  value <- value[!is.na(value)]
  if (length(value) == 0L) {
    return("")
  }

  # Normalise logicals and dates before encoding
  if (is.logical(value)) {
    value <- tolower(as.character(value))
  } else if (inherits(value, "Date")) {
    value <- format(value, "%Y-%m-%d")
  } else {
    value <- as.character(value)
  }
  key <- utils::URLencode(name, reserved = TRUE)
  vals <- vapply(value, utils::URLencode, character(1), reserved = TRUE, USE.NAMES = FALSE)

  pairs <- switch(style,
    explode = {
      paste0(key, "=", vals)
    },
    comma = {
      paste0(key, "=", paste(vals, collapse = ","))
    },
    space = {
      paste0(key, "=", paste(vals, collapse = "%20"))
    },
    pipe = {
      paste0(key, "=", paste(vals, collapse = "|"))
    },
    brackets = {
      paste0(key, "%5B%5D=", vals)
    }
  )
  return(paste(pairs, collapse = "&"))
}

#' Check query parameters against the documented limits
#'
#' Unknown parameters are reported but kept.
#'
#' @param params A named list of query parameters.
#' @param limits A named list with elements `limit`, `offset`, `q`, `lang`,
#'   `from` and `to` giving the allowed range or values of each parameter.
#' @return A list with elements `params` (the checked parameters) and
#'   `problems` (a character vector, empty when all parameters are valid).
#' @export
check_query_params <- function(params,
                               limits = list(limit = c(1, 500), offset = c(0, Inf),
                                             q = c(1, 256), lang = c("en", "de", "fr"),
                                             from = "date", to = "date")) {
  problems <- character(0)
  known <- c("limit", "offset", "q", "lang", "from", "to")
  extra <- setdiff(names(params), known)
  if (length(extra) > 0L) {
    problems <- c(problems, sprintf("`%s` is not a known parameter.", extra))
  }

  # Check `limit`
  if (!is.null(params$limit)) {
    x <- suppressWarnings(as.numeric(params$limit))
    if (is.na(x) || x < limits$limit[1] || x > limits$limit[2]) {
      problems <- c(problems, sprintf("`%s` is out of range.", "limit"))
    }
  }

  # Check `offset`
  if (!is.null(params$offset)) {
    x <- suppressWarnings(as.numeric(params$offset))
    if (is.na(x) || x < limits$offset[1] || x > limits$offset[2]) {
      problems <- c(problems, sprintf("`%s` is out of range.", "offset"))
    }
  }

  # Check `q`
  if (!is.null(params$q)) {
    x <- nchar(params$q)
    if (is.na(x) || x < limits$q[1] || x > limits$q[2]) {
      problems <- c(problems, sprintf("`%s` is out of range.", "q"))
    }
  }

  # Check `lang`
  if (!is.null(params$lang)) {
    if (!params$lang %in% limits$lang) {
      problems <- c(problems, sprintf("`%s` is not an allowed value.", "lang"))
    }
  }

  # Check `from`
  if (!is.null(params$from)) {
    x <- as.Date(params$from, optional = TRUE)
    if (is.na(x)) {
      problems <- c(problems, sprintf("`%s` is not a valid %s.", "from", limits$from))
    }
  }

  # Check `to`
  if (!is.null(params$to)) {
    x <- as.Date(params$to, optional = TRUE)
    if (is.na(x)) {
      problems <- c(problems, sprintf("`%s` is not a valid %s.", "to", limits$to))
    }
  }

  return(list(params = params, problems = problems))
}

#' Assemble a request URL
#'
#' Joins the base URL, path segments, query and fragment, encoding each part.
#'
#' @param base_url Scheme and host, for example `"https://api.example.org"`.
#' @param path Character vector of path segments.
#' @param query A named list of query parameters.
#' @param style Passed to [encode_query_pair()] for every parameter.
#' @param fragment Optional fragment, without the leading `#`.
#' @return A single string.
#' @export
build_request_url <- function(base_url, path = character(0), query = list(),
                              style = "explode", fragment = NULL) {
  if (!grepl("^https?://", base_url)) {
    stop("`base_url` must start with http:// or https://.", call. = FALSE)
  }

  # 1. Base: drop trailing slashes
  url <- sub("/+$", "", base_url)

  # 2. Path: encode each segment and join with slashes
  path <- path[nzchar(path)]
  if (length(path) > 0L) {
    segments <- vapply(path, utils::URLencode, character(1), reserved = TRUE,
      USE.NAMES = FALSE)
    url <- paste0(url, "/", paste(segments, collapse = "/"))
  }

  # 3. Query: drop NULL values and encode each parameter
  query <- query[!vapply(query, is.null, logical(1))]
  if (length(query) > 0L) {
    pairs <- vapply(names(query), function(nm) {
      encode_query_pair(nm, query[[nm]], style = style)
    }, character(1), USE.NAMES = FALSE)
    pairs <- pairs[nzchar(pairs)]
    if (length(pairs) > 0L) {
      url <- paste0(url, "?", paste(pairs, collapse = "&"))
    }
  }

  # 4. Fragment: encode and append
  if (!is.null(fragment) && nzchar(fragment)) {
    url <- paste0(url, "#", utils::URLencode(fragment, reserved = TRUE))
  }

  return(url)
}

#' Parse a query string into typed values
#'
#' Repeated keys are collected into vectors before conversion.
#'
#' @param x A query string, with or without the leading `?`.
#' @param types A named character vector mapping parameter names to one of
#'   `"character"`, `"integer"`, `"numeric"`, `"logical"`, `"date"` or
#'   `"csv"`; unlisted parameters stay character.
#' @return A named list of converted values.
#' @export
parse_query_string <- function(x, types = character(0)) {
  x <- sub("^\\?", "", x)
  if (!nzchar(x)) {
    return(list())
  }
  pieces <- strsplit(strsplit(x, "&", fixed = TRUE)[[1]], "=", fixed = TRUE)
  keys <- vapply(pieces, function(p) utils::URLdecode(p[1]), character(1))
  vals <- vapply(pieces, function(p) {
    if (length(p) < 2L) "" else utils::URLdecode(paste(p[-1], collapse = "="))
  }, character(1))
  out <- split(vals, factor(keys, levels = unique(keys)))
  for (nm in names(out)) {
    type <- if (nm %in% names(types)) types[[nm]] else "character"
    out[[nm]] <- switch(type,
      character = {
        out[[nm]]
      },
      integer = {
        as.integer(out[[nm]])
      },
      numeric = {
        as.numeric(out[[nm]])
      },
      logical = {
        tolower(out[[nm]]) %in% c("true", "1", "yes")
      },
      date = {
        as.Date(out[[nm]])
      },
      csv = {
        unlist(strsplit(out[[nm]], ",", fixed = TRUE), use.names = FALSE)
      },
      stop(sprintf("Unknown type `%s` for parameter `%s`.", type, nm), call. = FALSE)
    )
  }

  return(out)
}
