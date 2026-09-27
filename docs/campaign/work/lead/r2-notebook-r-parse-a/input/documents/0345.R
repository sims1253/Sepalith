thr1.singlesum <- function(S1, S2, func_S, thr){
  N = dim(S1)[3]
  output = 0
  for (i in 1:N){
    S1tmp = func_S(S1[,,i], thr)
    S2tmp = S2[,,i]
    output = output + norm(S1tmp-S2tmp,"f")
  }
  return(output)
}