yes.no.menu <- function(title = NULL) {

  # function for appropriate response
  if (is.null(title)) {
    return("yes")
  } else {
    message(paste0("Please enter 'yes' or 'no' to ", title, "."))
    choice <- readline(": ")
    if (tolower(choice) == "yes") {
      return("yes")
    } else if (tolower(choice) == "no") {
      return("no")
    } else {
      message("Invalid input. Please enter 'yes' or 'no'.")
      return(yes.no.menu(title))
    }
  }
