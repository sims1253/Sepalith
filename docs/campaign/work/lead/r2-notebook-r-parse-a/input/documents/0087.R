.query_success <- function(r) {
  exc <- httr2::resp_header(r, "x-clickhouse-exception-code")
  if (httr2::resp_status(r) >= 300 || !is.null(exc)) {
    if (!is.null(exc)) {
      m <- rawToChar(httr2::resp_body_raw(r))
    } else {
      m <- sprintf(
        "Connection error. Status code: %s",
        httr2::resp_status(r)
      )
    }
    toRet <- FALSE
    attr(toRet, "m") <- m
  } else {
    toRet <- TRUE
  }
  return(toRet)
}