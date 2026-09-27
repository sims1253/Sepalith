.build_http_req <- function(
  host,
  port,
  https,
  host_path,
  session,
  session_timeout = NA,
  settings,
  query = ""
) {
  use_session <- !is.na(session)
  if (use_session) {
    to_ret <- sprintf(
      "%s://%s:%s/%s?session_id=%s&session_check=%s%s%s%s",
      ifelse(https, "https", "http"),
      host,
      port,
      ifelse(is.na(host_path), "", paste0(host_path, "/")),
      session,
      as.integer(is.na(session_timeout)),
      ifelse(
        is.na(session_timeout),
        "",
        sprintf("&session_timeout=%s", as.integer(session_timeout))
      ),
      ifelse(
        settings == "",
        "",
        paste0("&", settings)
      ),
      ifelse(
        is.na(query) || query == "",
        "",
        sprintf("&query=%s", utils::URLencode(query))
      )
    )
  } else {
    to_ret <- sprintf(
      "%s://%s:%s/%s%s%s",
      ifelse(https, "https", "http"),
      host,
      port,
      ifelse(is.na(host_path), "", paste0(host_path, "/")),
      ifelse(
        settings == "",
        "",
        paste0("?", settings)
      ),
      ifelse(
        is.na(query) || query == "",
        "",
        sprintf(
          "%squery=%s",
          ifelse(
            settings == "",
            "?",
            "&"
          ),
          utils::URLencode(query)
        )
      )
    )
  }
  return(to_ret)
}