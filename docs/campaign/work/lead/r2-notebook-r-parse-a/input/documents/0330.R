length_of_region <- function(sequence, region) {
  region_length <- nchar(sequence)
  names(region_length) <- paste0(region, "_length")
  return(region_length)
}