library(plumber)

for (helper in list.files("R", pattern = "[.]R$", full.names = TRUE)) {
  source(helper)
}

con <- openStore(Sys.getenv("RELAYDESK_DB", "relaydesk.sqlite"))
apiKeys <- strsplit(Sys.getenv("RELAYDESK_API_KEYS"), ",", fixed = TRUE)[[1]]
webhookSecrets <- list(
  git = Sys.getenv("RELAYDESK_GIT_SECRET"),
  form = Sys.getenv("RELAYDESK_FORM_SECRET"),
  alert = Sys.getenv("RELAYDESK_ALERT_SECRET")
)

#* @apiTitle relaydesk ticket intake
#* @apiDescription Accepts tickets from clients and webhooks from external services.

#* Log every request with its method, path and client address
#* @filter logger
function(req) {
  message(sprintf(
    "%s %s %s from %s",
    format(Sys.time(), "%H:%M:%S"),
    req$REQUEST_METHOD,
    req$PATH_INFO,
    req$REMOTE_ADDR %||% "-"
  ))
  forward()
}

#* Require an API key on everything except health checks and webhooks
#* @filter apiKey
function(req, res) {
  path <- req$PATH_INFO
  # Webhooks authenticate with their own signatures instead.
  if (path == "/health" || startsWith(path, "/webhooks/")) {
    return(forward())
  }

  auth <- req$HTTP_AUTHORIZATION %||% ""
  key <- sub("^Bearer\\s+", "", auth)
  if (!nzchar(key) || !key %in% apiKeys) {
    return(sendUnauthorized(res))
  }
  forward()
}

#* Report service health
#* @get /health
#* @serializer unboxedJSON
function(res) {
  dbOk <- tryCatch(
    {
      DBI::dbGetQuery(con, "SELECT 1")
      TRUE
    },
    error = function(e) FALSE
  )

  if (!dbOk) {
    return(sendError(res, 503L, "unavailable", "Database is not reachable."))
  }
  sendOk(res, list(status = "ok", time = storeNow()))
}

#* List tickets, optionally filtered by status and queue
#* @param status:str Filter by ticket status
#* @param queue:str Filter by queue name
#* @param page:int Page number, starting at 1
#* @param perPage:int Number of tickets per page
#* @get /tickets
#* @serializer unboxedJSON
function(res, status = NULL, queue = NULL, page = NULL, perPage = NULL) {
  paging <- validatePaging(page, perPage)
  if (!paging$ok) {
    return(sendInvalid(res, paging$errors))
  }

  if (!is.null(status)) {
    checked <- requireChoice(status, "status", c("open", "in_progress", "waiting", "closed"))
    if (!is.null(checked$error)) {
      return(sendInvalid(res, c(status = checked$error)))
    }
    status <- checked$value
  }

  tickets <- listTickets(con, status = status, queue = queue)
  sendOk(res, paginate(tickets, paging$data$page, paging$data$perPage, "/tickets"))
}

#* Fetch a single ticket
#* @param id:int The ticket id
#* @get /tickets/<id:int>
#* @serializer unboxedJSON
function(res, id) {
  tryCatch(
    sendOk(res, getTicket(con, id)),
    error = function(e) sendCondition(res, e)
  )
}

#* Create a ticket
#* @post /tickets
#* @serializer unboxedJSON
function(req, res) {
  body <- req$body
  checked <- validateTicketInput(body)
  if (!checked$ok) {
    return(sendInvalid(res, checked$errors))
  }

  tryCatch(
    {
      ticket <- insertTicket(con, checked$data)
      sendCreated(res, ticket, sprintf("/tickets/%d", ticket$id))
    },
    error = function(e) sendCondition(res, e)
  )
}

#* Change the status of a ticket
#* @param id:int The ticket id
#* @patch /tickets/<id:int>/status
#* @serializer unboxedJSON
function(req, res, id) {
  checked <- requireChoice(
    req$body$status,
    "status",
    c("open", "in_progress", "waiting", "closed")
  )
  if (!is.null(checked$error)) {
    return(sendInvalid(res, c(status = checked$error)))
  }

  tryCatch(
    sendOk(res, updateTicketStatus(con, id, checked$value)),
    error = function(e) sendCondition(res, e)
  )
}

#* Receive a webhook from an external service
#* @param source:str One of git, form or alert
#* @post /webhooks/<source>
#* @serializer unboxedJSON
function(req, res, source) {
  if (!source %in% names(webhookSecrets)) {
    return(sendNotFound(res, sprintf("Webhook source '%s'", source)))
  }

  result <- handleWebhook(source, req, webhookSecrets)
  if (result$status == "rejected") {
    return(sendError(res, 401L, "rejected", result$reason))
  }
  if (result$status == "ignored") {
    return(sendOk(res, list(status = "ignored", reason = result$reason), status = 202L))
  }

  # Providers retry on timeouts; the delivery id makes the import idempotent.
  event <- result$event
  deliveryId <- req$HTTP_X_DELIVERY_ID %||% paste(source, event$externalId, sep = ":")
  if (!recordDelivery(con, deliveryId, source)) {
    return(sendOk(res, list(status = "duplicate", deliveryId = deliveryId), status = 200L))
  }

  ticket <- insertTicket(con, list(
    title = substr(event$title, 1, 120),
    description = paste("Imported from", event$url %||% source),
    reporter = event$actor,
    priority = if (event$kind == "incident") "urgent" else "normal",
    queue = source,
    tags = c(source, event$kind)
  ))
  sendCreated(res, ticket, sprintf("/tickets/%d", ticket$id))
}
