
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