yes.no.menu <- function(title = NULL) {

  # function for appropriate response
  if (is.null(title)) {
    title <- "Yes or No"
  }
  if (interactive()) {
    menu <- c("Yes", "No")
    choice <- menu[as.character(menu)[input()]]
  } else {
    choice <- "Yes"
  }
  return(choice)
}