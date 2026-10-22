# skipped: needs DBI and RSQLite (openStore, insertTicket, getTicket,
# listTickets, updateTicketStatus, recordDelivery)
stamp <- storeNow()
stopifnot(
  is.character(stamp),
  length(stamp) == 1L,
  grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$", stamp)
)
