#' Open the ticket store
#'
#' Connects to the SQLite database and creates the tables on first use.
#'
#' @param path Path to the SQLite database file.
#' @return A DBI connection.
#' @noRd
openStore <- function(path) {
  con <- DBI::dbConnect(RSQLite::SQLite(), path)
  # WAL lets the health check read while a webhook is writing.
  DBI::dbExecute(con, "PRAGMA journal_mode = WAL")

  if (!DBI::dbExistsTable(con, "tickets")) {
    DBI::dbExecute(con, "
      CREATE TABLE tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        reporter TEXT NOT NULL,
        priority TEXT NOT NULL,
        queue TEXT NOT NULL,
        tags TEXT,
        status TEXT NOT NULL DEFAULT 'open',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )")
  }
  if (!DBI::dbExistsTable(con, "deliveries")) {
    DBI::dbExecute(con, "
      CREATE TABLE deliveries (
        delivery_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        ticket_id INTEGER,
        received_at TEXT NOT NULL
      )")
  }

  con
}

#' Format the current time for storage
#'
#' All timestamps are stored as UTC text so that they sort correctly.
#'
#' @noRd
storeNow <- function() {
  format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
}

#' Insert a new ticket
#'
#' Stores a validated ticket and returns it as it now exists in the
#' database, including its generated id.
#'
#' @param con A DBI connection returned by `openStore()`.
#' @param ticket A named list of validated ticket fields.
#' @return A one-row data frame with the stored ticket.
#' @noRd
insertTicket <- function(con, ticket) {
  now <- storeNow()

  # Tags are stored as one comma-separated string; getTicket() splits them.
  DBI::dbExecute(
    con,
    "INSERT INTO tickets
       (title, description, reporter, priority, queue, tags, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    params = list(
      ticket$title,
      ticket$description %||% "",
      ticket$reporter,
      ticket$priority %||% "normal",
      ticket$queue,
      paste(ticket$tags %||% character(), collapse = ","),
      now,
      now
    )
  )

  id <- DBI::dbGetQuery(con, "SELECT last_insert_rowid() AS id")$id
  getTicket(con, id)
}

#' Fetch a single ticket
#'
#' Looks a ticket up by id and signals a `relaydesk_not_found` condition
#' when it does not exist.
#'
#' @param con A DBI connection returned by `openStore()`.
#' @param id Integer ticket id.
#' @return A one-row data frame with the stored ticket.
#' @noRd
getTicket <- function(con, id) {
  row <- DBI::dbGetQuery(con, "SELECT * FROM tickets WHERE id = ?", params = list(id))
  if (nrow(row) == 0) {
    stop(structure(
      class = c("relaydesk_not_found", "error", "condition"),
      list(message = sprintf("Ticket %s", id), call = NULL)
    ))
  }

  row$tags <- strsplit(row$tags, ",", fixed = TRUE)
  row
}

#' List tickets with optional filters
#'
#' Builds the `WHERE` clause from whichever filters are supplied and orders
#' the result by most recent first. Tags are split into a list column, as in
#' `getTicket()`.
#'
#' @param con A DBI connection returned by `openStore()`.
#' @param status Optional status to filter on.
#' @param queue Optional queue name to filter on.
#' @return A data frame of tickets.
#' @noRd
listTickets <- function(con, status = NULL, queue = NULL) {
  clauses <- character()
  params <- list()

  # Values always go through placeholders, never into the SQL text.
  if (!is.null(status)) {
    clauses <- c(clauses, "status = ?")
    params <- c(params, list(status))
  }
  if (!is.null(queue)) {
    clauses <- c(clauses, "queue = ?")
    params <- c(params, list(queue))
  }

  sql <- "SELECT * FROM tickets"
  if (length(clauses) > 0) {
    sql <- paste(sql, "WHERE", paste(clauses, collapse = " AND "))
  }
  sql <- paste(sql, "ORDER BY created_at DESC, id DESC")

  rows <- if (length(params) == 0) {
    DBI::dbGetQuery(con, sql)
  } else {
    DBI::dbGetQuery(con, sql, params = params)
  }
  rows$tags <- strsplit(rows$tags, ",", fixed = TRUE)
  rows
}

#' Change the status of a ticket
#'
#' Enforces the allowed status transitions and signals a
#' `relaydesk_conflict` condition for anything else.
#'
#' @param con A DBI connection returned by `openStore()`.
#' @param id Integer ticket id.
#' @param status The new status.
#' @return A one-row data frame with the updated ticket.
#' @noRd
updateTicketStatus <- function(con, id, status) {
  current <- getTicket(con, id)$status
  allowed <- switch(current,
    open = c("in_progress", "closed"),
    in_progress = c("open", "waiting", "closed"),
    waiting = c("in_progress", "closed"),
    closed = "open",
    character()
  )

  # Closed tickets can only be reopened; everything else follows the table.
  if (!status %in% allowed) {
    stop(structure(
      class = c("relaydesk_conflict", "error", "condition"),
      list(message = sprintf("Cannot move ticket from '%s' to '%s'", current, status), call = NULL)
    ))
  }

  DBI::dbExecute(
    con,
    "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
    params = list(status, storeNow(), id)
  )
  getTicket(con, id)
}

#' Record a webhook delivery
#'
#' Stores the delivery id so repeated deliveries of the same event can be
#' recognised and skipped.
#'
#' @param con A DBI connection returned by `openStore()`.
#' @param deliveryId Unique delivery id sent by the webhook provider.
#' @param source Name of the webhook source.
#' @param ticketId Id of the ticket created from the delivery, if any.
#' @return `TRUE` if the delivery was new, `FALSE` if it was seen before.
#' @noRd
recordDelivery <- function(con, deliveryId, source, ticketId = NA_integer_) {
  seen <- DBI::dbGetQuery(
    con,
    "SELECT COUNT(*) AS n FROM deliveries WHERE delivery_id = ?",
    params = list(deliveryId)
  )$n
  if (seen > 0) {
    return(FALSE)
  }

  DBI::dbExecute(
    con,
    "INSERT INTO deliveries (delivery_id, source, ticket_id, received_at) VALUES (?, ?, ?, ?)",
    params = list(deliveryId, source, ticketId, storeNow())
  )
  TRUE
}
