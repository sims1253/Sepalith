get.random.beta = function(mar1,mar2){
  ret = rbeta(1, mar1, mar2, ncp = 0)
  return(ret)
}