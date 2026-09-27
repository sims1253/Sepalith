precision <- function(object, progress = TRUE) {
  if (is(object, "estimate") & is(object, "default")) {
    iter <- object$iter

    p <- object$p

    cors <- pcor_to_cor(object)$R

    if (isTRUE(progress)) {
      pb <- utils::txtProgressBar(min = 0, max = iter, style = 3)
    }

    precision <- vapply(
      1:iter,
      function(s) {
        Theta <- solve(cors[,, s])

        if (isTRUE(progress)) {
          utils::setTxtProgressBar(pb, s)
        }
        Theta
      },
      FUN.VALUE = matrix(0, p, p)
    )
  } else {
    stop("class not currently supported")
  }

  precision_mean <- apply(precision, 1:2, mean)

  returned_object <- list(
    precision_mean = precision_mean,
    precision = precision
  )

  class(returned_object) <- c("BGGM", "precision")

  return(returned_object)
}