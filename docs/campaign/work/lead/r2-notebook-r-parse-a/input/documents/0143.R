expect_frec_iid <- function(ndata, tau, scale, shape, estMethod) {
  data <- rfrechet(ndata, scale = scale, shape = shape)
  res <- estExpectiles(data, tau, estMethod)$ExpctHat
  return(res)
}