cwt <- function(t, scales, variance = 1, summed_wavelet = FALSE, threads = 1L) {
  # Check input argument types and lengths
  is_vector_input <- is.numeric(t) && is.null(dim(t))

  if (is_vector_input) {
    frame <- matrix(t, nrow = 1, byrow = TRUE)
    colnames(frame) <- names(t)
  } else {
    frame <- as.matrix(t)
  }

  if (min(scales) <= 0) {
    stop("scales must be positive numbers")
  }

  if (min(variance) <= 0) {
    stop("variance must be a positive number")
  }

  if (min(threads) < 1) {
    stop("threads must be a positive interger higher than 1")
  }

  # Apply CWT_rcpp
  transformation <- cwt_rcpp(
    t = frame,
    scales = scales,
    variance = variance,
    threads = threads
  )

  # Apply summed wavelet
  if (summed_wavelet) {
    transformation <- rowSums(transformation, dims = 2)

    if (is_vector_input) {
      transformation <- as.vector(transformation)
      names(transformation) <- names(t)
    } else {
      transformation <- as.data.table(transformation)
      colnames(transformation) <- colnames(t)
    }
  }

  return(transformation)
}