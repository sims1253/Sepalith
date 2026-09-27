.check_db_session <- function(
  dbc,
  session_timeout = NA
) {
  r <- .send_query(
    dbc = dbc,
    query = "SELECT currentUser() as user",
    session_timeout = session_timeout
  )
  toRet <- .query_success(r)
  if (toRet) {
    attr(toRet, "user") <- sub("\n$", "", rawToChar(httr2::resp_body_raw(r)))
  }
  return(toRet)
}