summary_calculate_centralities <- function(x){

  x <- x[!sapply(x, is.null)]

  x <- na.omit(x)

  summary_list <- lapply(x, function(x) summary(x))
  return(summary_list)
}