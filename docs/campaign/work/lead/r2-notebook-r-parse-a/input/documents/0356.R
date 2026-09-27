plogpFun <- function(p) {
  val <- ifelse(p == 0, 0, p * log(p))
  return(val)
}