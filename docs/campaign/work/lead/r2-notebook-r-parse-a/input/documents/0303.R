read_dimension_metadata <- function(dataset_code, dimension_code) {
  response <- get_fao(sprintf(
    "/definitions/domain/%s/%s",
    dataset_code,
    dimension_code
  ))
  content <- content(response)

  data <- as.data.frame(rbindlist(lapply(content$data, as.data.table)))

  attr(data, "metadata") <- content$metadata

  return(data)
}