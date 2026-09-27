.check_coords <- function(lat1, lon1, lat2, lon2) {
  if (
    identical(as.numeric(lat1), as.numeric(lat2)) &
      identical(as.numeric(lon1), as.numeric(lon2))
  ) {
    match <- TRUE
  } else {
    match <- FALSE
  }
  return(match)
}