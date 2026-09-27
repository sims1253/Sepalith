fg_get_dates_of_interest <- function(
  search_categories = "",
  use_default = TRUE,
  startdt = NULL,
  totoday = FALSE
) {
  DT_ENTRY <- NULL
  message_if(
    the_fg$verbose,
    "fg_get_dates_of_interest(",
    search_categories,
    ")"
  )
  rtn <- the_fg$doi_dates[
    grepl(search_categories, the_fg$doi_dates$category, ignore.case = TRUE),
  ][order(DT_ENTRY)]
  enddt <- ifelse(is.logical(totoday), Sys.Date(), lubridate::as_date(totoday))
  if (!is.null(startdt)) {
    rtn <- rtn[END_DT_ENTRY >= as.Date(startdt), ]
  }
  rtn <- rtn[DT_ENTRY <= enddt, ]
  if (nrow(rtn) > 0 & totoday) {
    if (!is.na(rtn[.N][["END_DT_ENTRY"]])) {
      newdate <- ifelse(
        totoday,
        Sys.Date(),
        lubridate::as_date(totoday)
      )
      rtn[.N, let(END_DT_ENTRY = newdate)]
    }
  }
  return(rtn[])
}