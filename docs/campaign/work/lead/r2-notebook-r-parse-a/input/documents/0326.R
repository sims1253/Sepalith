multiJI <- function(data, winsize = 50, theta_seq = NULL, scale = TRUE) {
  data <- as.data.frame(data)

  if (NCOL(data) <= 2) {
    stop("Data only contains two columns. multiJI require 2+ timeseries")
  }
  if (!all(apply(data[, -1], 2, is.numeric))) {
    stop("Not all timeseries are numeric")
  }

  window <- round(dim(data)[1] * winsize / 100)

  out <- lapply(1:(dim(data)[1] - window + 1), function(i) {
    jac <- multi_smap_jacobian(
      data = data[i:(i + window - 1), ],
      theta_seq = theta_seq,
      scale = scale
    )

    jac_out <- Reduce("+", jac$smapJ) / length(jac$smapJ) #elementwise mean of time varying Jacobians

    #j_dom_eig <- max(abs(Re(eigen(jac$smapJ[[length(jac$smapJ)]])$values))) #extract last Jacobian only
    j_dom_eig <- max(abs(Re(eigen(jac_out)$values))) #average across timevarying Jacobians

    return(data.frame("time" = data[i + window - 1, 1], "smap_J" = j_dom_eig))
  })
  out <- do.call("rbind", out)
  return(out)
}