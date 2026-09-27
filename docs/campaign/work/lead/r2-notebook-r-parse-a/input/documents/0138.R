optFunc <- function(pars, data.y, data.x, kernel, lType, xIs, yIs, nPatients) {
  res <- cmp_uFunc(
    pars = pars,
    data.y = data.y,
    data.x = data.x,
    kernel = kernel,
    lType = lType,
    xIs = xIs,
    yIs = yIs,
    nPatients = nPatients
  )

  return(res %*% res)
}