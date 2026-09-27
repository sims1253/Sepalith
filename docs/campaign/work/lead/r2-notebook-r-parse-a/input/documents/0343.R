thr1.once <- function(X, thr, func){
  output   = list()
  output$S = func(X, thr)
  return(output)
}