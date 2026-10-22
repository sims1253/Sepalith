#' Send a successful JSON response
#'
#' Sets the status code on the response and wraps the payload in the
#' standard `data` envelope.
#'
#' @param res The plumber response object.
#' @param data The payload to return to the client.
#' @param status HTTP status code.
#' @return The response envelope as a list.
#' @keywords internal
sendOk <- function(res, data, status = 200L) {
  res$status <- status
  list(data = data, error = NULL)
}

#' Send a created response
#'
#' Sets status 201 and a `Location` header pointing at the new resource.
#'
#' @param res The plumber response object.
#' @param data The payload to return to the client.
#' @param location Path of the created resource.
#' @keywords internal
sendCreated <- function(res, data, location) {
  res$status <- 201L
  res$setHeader("Location", location)
  list(data = data, error = NULL)
}

#' Send an error response
#'
#' Sets the status code and wraps a machine-readable code, a message and
#' optional details in the standard `error` envelope.
#'
#' @param res The plumber response object.
#' @param status HTTP status code.
#' @param code Short machine-readable error code.
#' @param message Human-readable error message.
#' @param details Optional named list or vector with more information.
#' @return The response envelope as a list.
#' @keywords internal
sendError <- function(res, status, code, message, details = NULL) {
  res$status <- status

  error <- list(code = code, message = message)
  if (!is.null(details)) {
    error$details <- as.list(details)
  }
  list(data = NULL, error = error)
}

#' @describeIn sendError Responds with status 422 and lists every invalid
#'   field in the details.
#' @param errors Named character vector of field errors.
#' @keywords internal
sendInvalid <- function(res, errors) {
  sendError(res, 422L, "invalid_input", "The request contains invalid fields.", errors)
}

#' @describeIn sendError Responds with status 404 and names the resource
#'   that was requested.
#' @param what Description of the missing resource.
#' @keywords internal
sendNotFound <- function(res, what) {
  sendError(res, 404L, "not_found", sprintf("%s was not found.", what))
}

#' @describeIn sendError Responds with status 401 and a `WWW-Authenticate`
#'   challenge header.
#' @param realm Authentication realm announced to the client.
#' @keywords internal
sendUnauthorized <- function(res, realm = "relaydesk") {
  res$setHeader("WWW-Authenticate", sprintf('Bearer realm="%s"', realm))
  sendError(res, 401L, "unauthorized", "A valid API key is required.")
}

#' Paginate a data frame of results
#'
#' Slices the requested page and adds paging metadata, including links to
#' the previous and next pages when they exist.
#'
#' @param items A data frame of results.
#' @param page Page number, starting at 1.
#' @param perPage Number of rows per page.
#' @param path Endpoint path used to build page links.
#' @return A list with `items` and `meta`.
#' @noRd
paginate <- function(items, page, perPage, path) {
  page <- max(1L, as.integer(page))
  perPage <- max(1L, as.integer(perPage))
  total <- nrow(items)
  pages <- max(1L, as.integer(ceiling(total / perPage)))

  # Pages past the end are not an error; they just come back empty.
  start <- (page - 1L) * perPage + 1L
  end <- min(total, page * perPage)
  rows <- if (start <= total) {
    items[seq.int(start, end), , drop = FALSE]
  } else {
    items[0, , drop = FALSE]
  }

  link <- function(p) sprintf("%s?page=%d&perPage=%d", path, p, perPage)
  list(
    items = rows,
    meta = list(
      page = page,
      perPage = perPage,
      total = total,
      pages = pages,
      prev = if (page > 1L) link(min(page - 1L, pages)) else NULL,
      next_ = if (page < pages) link(page + 1L) else NULL
    )
  )
}

#' Convert an R condition into an error response
#'
#' Maps known condition classes raised by the helpers to suitable HTTP
#' errors and hides internal details for everything else.
#'
#' @param res The plumber response object.
#' @param cond The condition object caught by the endpoint.
#' @return The response envelope as a list.
#' @noRd
sendCondition <- function(res, cond) {
  if (inherits(cond, "relaydesk_not_found")) {
    return(sendNotFound(res, conditionMessage(cond)))
  }
  if (inherits(cond, "relaydesk_conflict")) {
    return(sendError(res, 409L, "conflict", conditionMessage(cond)))
  }
  if (inherits(cond, "relaydesk_invalid")) {
    return(sendError(res, 422L, "invalid_input", conditionMessage(cond)))
  }

  # Anything else is a bug on our side: log it, but keep the details private.
  message("Unhandled error: ", conditionMessage(cond))
  sendError(res, 500L, "internal_error", "Something went wrong on our side.")
}
