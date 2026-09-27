gridding.default <- function(
  meta,
  btf,
  res = 12,
  resByData = FALSE,
  verbose = TRUE
) {
  res <- gridding_internal(
    meta = meta,
    btf = btf,
    res = res,
    resByData = resByData,
    verbose = verbose
  )
  return(res)
}