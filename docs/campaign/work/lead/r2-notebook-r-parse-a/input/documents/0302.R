read_dataset_dimension <- function(dataset_code) {
  response <- get_fao(sprintf("/definitions/domain/%s", dataset_code))
  content <- content(response)

  data <- as.data.frame(rbindlist(lapply(content$data, as.data.table)))

  attr(data, "metadata") <- content$metadata

  return(data)
}