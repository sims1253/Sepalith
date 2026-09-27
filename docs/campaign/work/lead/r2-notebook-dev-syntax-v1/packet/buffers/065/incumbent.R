#' @name global_variables
#' @title Global Variables for IBDInfer
#' @description This section declares global variables used in the IBDInfer package to prevent R CMD check warnings.
#' @keywords internal
utils::globalVariables(c("blk_id", "Y.obs", "n.vec", "R_k", "n_kt", "s2"))


#' @title Design-based Inference for Incomplete Block Designs
#' @description Conduct the design-based inference for incomplete block designs.
#'
#'
#' @param y Observed outcomes.
#' @param b Block identifier (ID).
#' @param z Assigned treatments.
#' @param g A contrast vector, must sum to zero.
#' @param w A weight vector, must sum to one and contain non-negative values.
#' @param alpha Confidence level, default set to 0.05.
#' @param data A data frame; if provided, y, b, and z should be column names in the data frame.
#'
#' @returns \code{IBDInfer} returns an object of class "IBD", which is a list containing the following components: :
#' \item{tau.ht}{The Horvitz-Thompson estimator of tau.}
#' \item{tau.haj}{The Hajek estimator of tau.}
#' \item{var_tau_ht_bb}{Variance estimator for the Horvitz-Thompson estimator with between-block bias.}
#' \item{var_tau_ht_wb}{Variance estimator for the Horvitz-Thompson estimator with within-block bias.}
#' \item{var_tau_haj_bb}{Variance estimator for the Hajek estimator with between-block bias.}
#' \item{var_tau_haj_wb}{Variance estimator for the Hajek estimator with within-block bias.}
#' \item{CI_ht_bb}{Confidence interval with the Horvitz-Thompson estimator and variance estimator with between-block bias.}
#' \item{CI_ht_wb}{Confidence interval with the Horvitz-Thompson estimator and variance estimator with within-block bias.}
#' \item{CI_haj_bb}{Confidence interval with the Hajek estimator and variance estimator with between-block bias.}
#' \item{CI_haj_wb}{Confidence interval with the Hajek estimator and variance estimator with within-block bias.}
#' \item{yht}{The Horvitz-Thompson estimator for each treatment.}
#' \item{yhaj}{The Hajek estimator for each treatment.}
#' \item{Sht_bb}{Covariance estimator for the Horvitz-Thompson estimator for each treatment with between-block bias.}
#' \item{Sht_wb}{Covariance estimator for the Horvitz-Thompson estimator for each treatment with within-block bias.}
#' \item{Shaj_bb}{Covariance estimator for the Hajek estimator for each treatment with between-block bias.}
#' \item{Shaj_wb}{Covariance estimator for the Hajek estimator for each treatment with within-block bias.}
#' \item{alpha}{Confidence level}
#'
#' @importFrom dplyr group_by summarise mutate %>% n
#' @importFrom tidyr pivot_wider
#' @importFrom stats qnorm var
#' @export
#'
#' @examples
#' K <- 6
#' n.trt <- 3
#' t <- 2
#' n.vec <- rep(4, K)
#' df <- IBDgen(K = K, n.trt = n.trt, t = t, n.vec = n.vec)$blk_assign
#' df$y <- rnorm(nrow(df), 0, 1)
#' IBDInfer <- IBDInfer(y = y, b = blk_id, z = assign, g = c(1, -1, 0), w = "Block", data = df)
#'
#' @references {
#' Koo, T., Pashley, N.E. (2024), Design-based Causal Inference for Incomplete Block Designs, \emph{arXiv preprint arXiv:2405.19312}. \cr
#' }
#'
IBDInfer <- function(
  y,
  b,
  z,
  g,
  w = c("Unit", "Block"),
  alpha = 0.05,
  data = NULL
) {
  # Check whether g is contrast vector or not
  stopifnot(is.vector(g), sum(g) == 0)
  if (!is.null(data)) {
    # Convert variable names to actual columns in the data frame
    y <- eval(substitute(y), data)
    b <- eval(substitute(b), data)
    z <- eval(substitute(z), data)
    # g <- eval(substitute(g), data)
  }
  # Check y is numeric outcomes of vector without missing values
  stopifnot(!missing(y), is.numeric(y))
  # Check b has missing values
  stopifnot(!missing(b))
  # Check z has missing values
  stopifnot(!missing(z))

  # Check whether dimension of g and unique(z) are the same
  stopifnot(length(unique(z)) == length(g))

  # Original assignment vector
  z.origin <- z
  # convert assign vector to numeric according to (alphabetical) order
  z <- as.numeric(factor(z, levels = sort(unique(z))))

  data.ibd <- data.frame(Y.obs = y, blk_id = as.factor(b), assign = z)

  K <- length(unique(data.ibd$blk_id))
  trt.set <- unique(data.ibd$assign) # {1,...,T}
  n.trt <- length(unique(data.ibd$assign)) # T
  data.sum <- data.ibd %>%
    group_by(blk_id) %>%
    summarise(
      t = length(unique(assign)),
      n.vec = length(Y.obs),
      R_k = list(unique(assign))
    )
  # t is constant over block.
  t <- data.sum$t[1]
  n.vec <- data.sum$n.vec # n_k

  # Check whether w is weight vector or not
  if (is.character(w)) {
    w <- match.arg(w)
    if (w %in% c("Unit", "Block")) {
      if (w == "Unit") {
        w <- n.vec / sum(n.vec)
      } else {
        w <- rep(1, K) / K
      }
    }
  }

  stopifnot(is.vector(w), round(sum(w), 5) == 1, all(w >= 0))

  R_k <- matrix(as.numeric(unlist(data.sum$R_k)), ncol = t, byrow = T)
  R_k.table <- t(apply(R_k, 1, function(x) table(factor(x, unique(c(R_k))))))
  R_k.table <- R_k.table[, order(colnames(R_k.table))]

  L <- colSums(R_k.table) # L_z
  l <- t(R_k.table) %*% R_k.table # l_{z,z'}

  #######################
  ### Point Estimator ###
  #######################

  data.obs.alt <- data.ibd %>%
    group_by(blk_id, assign) %>%
    summarise(mean = mean(Y.obs), s2 = var(Y.obs, na.rm = TRUE), n_kt = n())
  data.obs.alt <- data.obs.alt %>%
    group_by(blk_id) %>%
    mutate(block.mean = mean(mean(na.rm = TRUE)), n_k = sum(n_kt))
  # temp <- data.obs.alt
  # temp$w <- rep(1,K)/K
  data.obs.wide <- data.obs.alt %>%
    tidyr::pivot_wider(names_from = assign, values_from = c(mean, s2))
  yestk <- data.obs.wide[, grepl("mean_", names(data.obs.wide))] # yestk(z)
  yestk <- yestk[, order(names(yestk))]
  # The Horvitz-Thompson estimator
  # w <- rep(1,K)/K
  Kw_yestk <- sweep(yestk, 1, K * w, '*')
  yht <- colMeans(Kw_yestk, na.rm = T)
  # HT estimator for one
  one_ht <- colSums(sweep(!is.na(Kw_yestk), 1, K * w, '*')) / L
  # The Hajek estimator
  yhaj <- yht / one_ht

  names(yhaj) <- names(yht) <- colnames(R_k.table)
  tau.ht <- as.numeric(t(g) %*% yht)
  tau.haj <- as.numeric(t(g) %*% yhaj)

  ##########################
  ### Variance Estimator ###
  ##########################

  # Within-block component #
  s_k <- data.obs.wide[, grepl("s2_", names(data.obs.wide))]
  s_k <- s_k[, order(names(s_k))]
  Kws_k <- sweep(s_k, 1, K * w, '*')
  mean_Kws_k <- sapply(seq_len(ncol(s_k)), function(j) {
    # Extract the relevant column and indicator
    col <- (Kws_k / (n.vec / t))[[j]]
    indicator <- R_k.table[, j]
    # Calculate mean only for elements where indicator is 1
    sum(col[indicator > 0]) / sum(indicator)
  })

  # Between-block component for HT #
  s_ht <- sapply(Kw_yestk, function(x) {
    var(x, na.rm = T)
  })
  cross_yht <- matrix(0, nrow = n.trt, ncol = n.trt)

  for (j in 1:n.trt) {
    mean_trt_j <- paste0("mean_", j)
    for (k in 1:n.trt) {
      mean_trt_k <- paste0("mean_", k)
      cross_yht[j, k] <- var(
        Kw_yestk[, mean_trt_j] - Kw_yestk[, mean_trt_k],
        na.rm = T
      )
    }
  }

  # Diag + off-diag for HT #
  diag_ht_bb <- s_ht / L
  diag_ht_wb <- (1 / L - 1 / K) * s_ht + 1 / K * mean_Kws_k

  off_diag_ht_wb <- matrix(0, nrow = n.trt, ncol = n.trt)
  off_diag_ht_bb <- matrix(0, nrow = n.trt, ncol = n.trt)
  for (j in 1:n.trt) {
    for (k in 1:n.trt) {
      if (k == j) {
        next
      }
      if (l[j, k] < 2) {
        off_diag_ht_wb[j, k] <- NA
        off_diag_ht_bb[j, k] <- 0
      } else {
        off_diag_ht_wb[j, k] <- 1 /
          2 *
          (l[j, k] / (L[j] * L[k]) - 1 / K) *
          (s_ht[j] + s_ht[k] - cross_yht[j, k])
        off_diag_ht_bb[j, k] <- 1 /
          2 *
          (l[j, k] / (L[j] * L[k])) *
          (s_ht[j] + s_ht[k] - cross_yht[j, k])
      }
    }
  }

  Sht_bb <- diag(diag_ht_bb) + off_diag_ht_bb
  Sht_wb <- diag(diag_ht_wb) + off_diag_ht_wb
  zero_idx <- which(g == 0)

  Sht_bb_temp <- Sht_bb
  Sht_wb_temp <- Sht_wb
  Sht_bb_temp[zero_idx, ] <- 0
  Sht_bb_temp[, zero_idx] <- 0
  Sht_wb_temp[zero_idx, ] <- 0
  Sht_wb_temp[, zero_idx] <- 0

  var_tau_ht_bb <- as.numeric(t(g) %*% Sht_bb_temp %*% g) #as.numeric(t(g[g.idx])%*%Sht_bb[g.idx,g.idx]%*%g[g.idx])
  var_tau_ht_wb <- as.numeric(t(g) %*% Sht_wb_temp %*% g) #as.numeric(t(g[g.idx])%*%Sht_wb[g.idx,g.idx]%*%g[g.idx])

  CI_ht_bb <- c(
    tau.ht - qnorm(1 - alpha / 2) * sqrt(var_tau_ht_bb),
    tau.ht + qnorm(1 - alpha / 2) * sqrt(var_tau_ht_bb)
  )
  CI_ht_wb <- c(
    tau.ht - qnorm(1 - alpha / 2) * sqrt(var_tau_ht_wb),
    tau.ht + qnorm(1 - alpha / 2) * sqrt(var_tau_ht_wb)
  )

  # Between-block component for Haj #
  yhaj_center <- sweep(yestk, 2, yhaj, '-')
  Kwyhaj_center <- sweep(yhaj_center, 1, K * w, '*')
  s_haj <- sapply(Kwyhaj_center, function(x) {
    n_non_na <- sum(!is.na(x)) # Count non-NA elements
    if (n_non_na > 1) {
      sum(x^2, na.rm = TRUE) / (n_non_na - 1) # Divide by (n-1)
    } else {
      NA # Avoid division by zero if there's only 1 non-NA value
    }
  })

  cross_yhaj <- matrix(0, nrow = n.trt, ncol = n.trt)
  for (j in 1:n.trt) {
    mean_trt_j <- paste0("mean_", j)
    for (k in 1:n.trt) {
      mean_trt_k <- paste0("mean_", k)
      if (l[j, k] < 2) {
        cross_yhaj[j, k] <- NA
      } else {
        one_jk <- sum(
          (!is.na(Kw_yestk)[, mean_trt_j]) *
            (!is.na(Kw_yestk)[, mean_trt_k]) *
            w *
            K
        ) /
          l[j, k]
        haj_diff_jk <- mean(
          Kw_yestk[, mean_trt_j] - Kw_yestk[, mean_trt_k],
          na.rm = T
        ) /
          one_jk
        haj_center_jk <- yestk[, mean_trt_j] - yestk[, mean_trt_k] - haj_diff_jk
        Kw_haj_center_jk <- K * w * haj_center_jk
        cross_yhaj[j, k] <- ifelse(
          sum(!is.na(Kw_haj_center_jk)) > 1,
          sum(Kw_haj_center_jk^2, na.rm = T) /
            (sum(!is.na(Kw_haj_center_jk)) - 1),
          NA
        )
        # cross_yhaj[j,k] <-var(Kw_yestk[,mean_trt_j]-Kw_yestk[,mean_trt_k],na.rm = T)
      }
    }
  }

  # Diag + off-diag for HT #
  diag_haj_bb <- s_haj / L
  diag_haj_wb <- (1 / L - 1 / K) * s_haj + 1 / K * mean_Kws_k

  off_diag_haj_wb <- matrix(0, nrow = n.trt, ncol = n.trt)
  off_diag_haj_bb <- matrix(0, nrow = n.trt, ncol = n.trt)
  for (j in 1:n.trt) {
    for (k in 1:n.trt) {
      if (k == j) {
        next
      }
      if (l[j, k] < 2) {
        off_diag_haj_wb[j, k] <- NA
        off_diag_haj_bb[j, k] <- 0
      } else {
        off_diag_haj_wb[j, k] <- 1 /
          2 *
          (l[j, k] / (L[j] * L[k]) - 1 / K) *
          (s_haj[j] + s_haj[k] - cross_yhaj[j, k])
        off_diag_haj_bb[j, k] <- 1 /
          2 *
          (l[j, k] / (L[j] * L[k])) *
          (s_haj[j] + s_haj[k] - cross_yhaj[j, k])
      }
    }
  }

  Shaj_bb <- diag(diag_haj_bb) + off_diag_haj_bb
  Shaj_wb <- diag(diag_haj_wb) + off_diag_haj_wb
  Shaj_bb_temp <- Shaj_bb
  Shaj_wb_temp <- Shaj_wb
  Shaj_bb_temp[zero_idx, ] <- 0
  Shaj_bb_temp[, zero_idx] <- 0
  Shaj_wb_temp[zero_idx, ] <- 0
  Shaj_wb_temp[, zero_idx] <- 0

  var_tau_haj_bb <- as.numeric(t(g) %*% Shaj_bb_temp %*% g) #as.numeric(t(g[g.idx])%*%Shaj_bb[g.idx,g.idx]%*%g[g.idx])
  var_tau_haj_wb <- as.numeric(t(g) %*% Shaj_wb_temp %*% g) #as.numeric(t(g[g.idx])%*%Shaj_wb[g.idx,g.idx]%*%g[g.idx])

  CI_haj_bb <- c(
    tau.haj - qnorm(1 - alpha / 2) * sqrt(var_tau_haj_bb),
    tau.haj + qnorm(1 - alpha / 2) * sqrt(var_tau_haj_bb)
  )
  CI_haj_wb <- c(
    tau.haj - qnorm(1 - alpha / 2) * sqrt(var_tau_haj_wb),
    tau.haj + qnorm(1 - alpha / 2) * sqrt(var_tau_haj_wb)
  )

  returnList <- list(
    tau.ht = tau.ht,
    tau.haj = tau.haj,
    var_tau_ht_bb = var_tau_ht_bb,
    var_tau_ht_wb = var_tau_ht_wb,
    var_tau_haj_bb = var_tau_haj_bb,
    var_tau_haj_wb = var_tau_haj_wb,
    CI_ht_bb = CI_ht_bb,
    CI_ht_wb = CI_ht_wb,
    CI_haj_bb = CI_haj_bb,
    CI_haj_wb = CI_haj_wb,
    yht = yht,
    yhaj = yhaj,
    Sht_bb = Sht_bb,
    Sht_wb = Sht_wb,
    Shaj_bb = Shaj_bb,
    Shaj_wb = Shaj_wb,
    alpha = alpha
  )

  class(returnList) <- "IBD"
  return(returnList)
}
