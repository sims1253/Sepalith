meu_mv_impr <- function(data, num_meu) {
  meu_matrix <- matrix()

  # Select the next set of centroids based on weighted probability
  if (num_meu == 1) {
    # Select the first centroid randomly
    meu_matrix <- meu_mv(data, num_meu)
  } else if (num_meu == 2) {
    meu_matrix <- meu_mv(data, 1)
    dist <- rowSums((data - meu_matrix)^2)
    meu_matrix <- rbind(meu_matrix, data[which.max(dist), ])
  } else if (num_meu > 2) {
    # Select the first centroid randomly
    meu_matrix <- meu_mv(data, 1)

    # Select the 2nd centroid
    dist <- rowSums((data - meu_matrix)^2)
    meu_matrix <- rbind(meu_matrix, data[which.max(dist), ])

    # Select the next centroids
    for (k in 3:num_meu) {
      dist_matrix <- data.frame(ncol = k - 1, nrow = nrow(data))

      # Calculate the distance of every point from existing centers
      for (i in seq_len(nrow(meu_matrix))) {
        dist_matrix <- cbind(dist_matrix, rowSums((data - meu_matrix[i, ])^2))
      }
      # Store the minimum distance
      dist_matrix <- apply(dist_matrix, 1, min)
      dist_matrix <- dist_matrix / sum(dist_matrix)
      meu_matrix <- rbind(meu_matrix, data[which.max(dist_matrix), ])
    }
  }
  return(meu_matrix)
}