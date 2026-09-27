gcenter <- function(var) {
  centered <- var - (mean(var, na.rm = T))
  return(centered)
}