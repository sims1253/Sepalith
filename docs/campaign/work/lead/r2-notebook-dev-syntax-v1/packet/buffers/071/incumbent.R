yes.no.menu <- function(title = NULL) {

  # function for appropriate response
  if (is.null(title)) {
    title <- "Yes or No?"
  }
  message("Do you want to use the SemNetDictionaries?")
  message("Press 'y' for yes or 'n' for no.")
  message("Press 'q' to quit.")
  choice <- readline(paste("Enter your choice: "))
  if (choice == "y") {
    return(TRUE)
  } else if (choice == "n") {
    return(FALSE)
  } else if (choice == "q") {
    return(FALSE)
  } else {
    message("Invalid choice. Please try again.")
    return(NULL)
  }
}