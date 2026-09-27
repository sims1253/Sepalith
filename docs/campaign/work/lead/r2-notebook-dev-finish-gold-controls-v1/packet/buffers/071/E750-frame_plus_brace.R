yes.no.menu <- function(title = NULL) {

  # function for appropriate response
  if (is.null(title)) {
    title <- "SemNet Dictionaries"
  }
  options(dictionary.yesno.menu.title = title)
}
}