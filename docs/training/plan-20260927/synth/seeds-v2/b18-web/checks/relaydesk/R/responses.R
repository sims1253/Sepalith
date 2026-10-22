# A tiny stand-in for the plumber response object.
mockRes <- function() {
  res <- new.env()
  res$status <- 200L
  res$headers <- list()
  res$setHeader <- function(name, value) res$headers[[name]] <- value
  res
}

res <- mockRes()
out <- sendOk(res, list(a = 1), status = 202L)
stopifnot(res$status == 202L, identical(out$data, list(a = 1)), is.null(out$error))

res <- mockRes()
out <- sendCreated(res, list(id = 3), "/tickets/3")
stopifnot(res$status == 201L, identical(res$headers$Location, "/tickets/3"))

res <- mockRes()
out <- sendInvalid(res, c(title = "`title` is required"))
stopifnot(
  res$status == 422L,
  identical(out$error$code, "invalid_input"),
  identical(out$error$details, list(title = "`title` is required"))
)

res <- mockRes()
out <- sendUnauthorized(res)
stopifnot(res$status == 401L, grepl("relaydesk", res$headers[["WWW-Authenticate"]]))
stopifnot(identical(sendNotFound(mockRes(), "Ticket 9")$error$message, "Ticket 9 was not found."))

items <- data.frame(id = 1:5)
p2 <- paginate(items, 2, 2, "/tickets")
p9 <- paginate(items, 9, 2, "/tickets")
stopifnot(
  identical(p2$items$id, 3:4),
  identical(p2$meta$pages, 3L),
  identical(p2$meta$next_, "/tickets?page=3&perPage=2"),
  is.data.frame(p9$items),
  nrow(p9$items) == 0L,
  identical(p9$meta$prev, "/tickets?page=3&perPage=2")
)

cond <- structure(class = c("relaydesk_conflict", "error", "condition"),
                  list(message = "nope", call = NULL))
res <- mockRes()
stopifnot(identical(sendCondition(res, cond)$error$code, "conflict"), res$status == 409L)
res <- mockRes()
out <- suppressMessages(sendCondition(res, simpleError("boom")))
stopifnot(res$status == 500L, !grepl("boom", out$error$message))
