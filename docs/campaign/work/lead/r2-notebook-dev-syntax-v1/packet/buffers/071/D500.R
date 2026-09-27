yes.no.menu <- function(title = NULL) {

  # function for appropriate response
  if (is.null(title)) {
    return("yes.no.menu")
  } else {
    return(paste("yes.no.menu:", title))
  }
}