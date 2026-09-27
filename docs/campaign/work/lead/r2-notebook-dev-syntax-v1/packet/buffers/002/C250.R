#' @title Deprecated: Use \code{flply} instead
#'
#' @description \code{fdply} is deprecated. Use \code{flply} instead.
#'
#' @param input A \code{data.frame} or \code{list} of \code{data.frame}s.
#' @inheritParams flply
#'
#' @export
#' @keywords internal
fdply <- function(
  input,
  nblocks = 1,
  key.sep = "\t",
  sep = "\t",
  skip = 0,
  colClasses = NULL,
  header = TRUE,
  stringsAsFactors = FALSE,
  select = NULL,
  drop = NULL,
  col.names = NULL,
  parallel = 1
) {
  .Deprecated("ftply")
  l <- flply(
    input,
    function(d) d,
    key.sep = key.sep,
    sep = sep,
    skip = skip,
    header = header,
    nblocks = nblocks,
    colClasses = colClasses,
    stringsAsFactors = stringsAsFactors,
    select = select,
    drop = drop,
    col.names = col.names,
    parallel = parallel
  )
  rbindlist(l)
}