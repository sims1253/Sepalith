ko.weights <- function(tree) {
  # extract sample values for each path with terminal node counts and
  # informative paths
  x <- tree$Get(
    'targetEst_samples',
    filterFun = function(node) node$isLeaf,
    traversal = 'post-order'
  )

  ## incase the above is not in matrix form...
  if (is.list(x)) {
    mat.x <- NULL
    for (i in seq_along(x)) {
      if (length(x[[i]]) > 0) {
        mat.x <- cbind(mat.x, x[[i]])
        colnames(mat.x)[dim(mat.x)[2]] <- names(x)[[i]]
      }
    }
    x <- mat.x
  }

  # make this a data frame
  x <- as.data.frame(x)

  # select only leaves with marginal counts
  getleaves <- which(tree$Get(
    'TerminalCount',
    filterFun = isLeaf,
    traversal = 'post-order'
  ))
  # if number of columns of x is greater than number of leaves, choose leaves only
  if (dim(x)[2] > length(getleaves)) {
    x <- x %>%
      select(all_of(getleaves))
  }

  # calculate weights
  sig <- cov(log(x)) # covariance matrix- use log to avoid numerical errors
  #prec <- Inverse(sig) # precision matrix
  prec <- MASS::ginv(sig) # precision matrix
  e <- seq(1, 1, length.out = dim(sig)[1])
  num.w <- t(e) %*% prec
  den.w <- t(e) %*% prec %*% e
  w <- 1 / den.w[1] * num.w
  colnames(w) <- names(getleaves)
  return(w)
}