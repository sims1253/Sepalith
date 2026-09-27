BNPpart <- function(
  partitions = NULL,
  scores = NULL,
  psm = NULL){
  value <- list(partitions = partitions,
                scores = scores,
                psm = psm)
  attr(value, "class") <- "BNPpart"
  value
}