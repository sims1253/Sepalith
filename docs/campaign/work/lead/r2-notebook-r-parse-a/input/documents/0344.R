thr1.once.givenS <- function(S, thr, func_S){
  output   = list()
  output$S = func_S(S, thr)
  return(output)
}