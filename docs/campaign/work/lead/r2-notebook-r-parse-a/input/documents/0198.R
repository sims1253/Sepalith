gridding.data.table <- function(
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
  data.table::setDT(res)
  return(res)
}