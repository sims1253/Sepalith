transfer <- function(r_data) {
  if (inherits(r_data, "sim")) {
    r_data <- r_data$data
  }

  n <- dim(r_data)[1]
  p <- dim(r_data)[2]
  data <- matrix(0, nrow = n, ncol = p + 1)
  size_unique_data <- 0

  result <- .C(
    "transfer_data",
    as.integer(r_data),
    data = as.integer(data),
    as.integer(n),
    as.integer(p),
    size_unique_data = as.integer(size_unique_data),
    PACKAGE = "BDgraph"
  )

  size_unique_data <- result$size_unique_data

  label <- colnames(r_data)
  if (is.null(label)) {
    label <- 1:p
  }

  data <- matrix(result$data, n, p + 1, dimnames = list(NULL, c(label, "ferq")))
  data <- data[(1:size_unique_data), ]

  return(data)
}