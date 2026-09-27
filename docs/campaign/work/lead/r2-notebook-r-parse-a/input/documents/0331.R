padStrings <- function(strings) {
  max_length <- max(nchar(strings))
  padded_strings <- sapply(strings, function(x) {
    paste0(x, strrep(" ", max_length - nchar(x)))
  })

  return(padded_strings)
}