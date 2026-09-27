#' Calculates Proactive Conservation Index
#'
#' \code{pci} Calculates the Proactive Conservation Index, a new tool to prioritize species for conservation, which incorporates information about future threats.
#'
#' @param sp character. Names of the taxa being evaluated.
#' @param var_out numeric. Threat variables. higher values must indicate increased threat.
#' @param var_in numeric. Interacting variables. Will modulate the effect of threat variables.
#' @param weight_out numeric. Weights for threat variables
#' @param weight_in numeric. Matrix of weights for the combination of interacting variables and threat variables.
#'
#' @return Data frame with PCI and rank.
#'
#' @examples
#'
#' # Invert variables that are negatively correlated with conservation priority
#'
#' vert_df$inv_range_area <- 1/vert_df$range_area
#' vert_df$inv_brood_size <- 1/vert_df$brood_size
#' vert_df$inv_protected_area <- 1/((vert_df$protected_area*vert_df$range_area+0.0001))
#'
#' # Select trait variables
#'
#' traits_vertebrates <-
#'    vert_df[c("body_mass",
#'              "inv_range_area",
#'              "inv_brood_size",
#'              "inv_protected_area",
#'              "AHI")]
#'
#' # Select threat variables for the year 2100, under scenarion SSP 5.85
#'
#' threats_2100_585 <-
#'    vert_df[c("clim_2100_585",
#'              "landuse_2100_585",
#'              "popdens_2100_585",
#'              "inv_threat")]
#'
#' # Calculate PCI
#'
#' vertebrates_pci <-
#'    pci(sp = vert_df$binomial,
#'        var_out = threats_2100_585,
#'        var_in = traits_vertebrates)
#'
#' @importFrom stats predict
#'
#' @export

pci <- function(
  sp,
  var_out,
  var_in = NULL,
  weight_out = NULL,
  weight_in = NULL
) {
  # geometric mean function
  am_mean <- function(x, weights, na.rm = TRUE) {
    sum(x * weights) / sum(weights)
  }

  # create matrix with 1 if "var_in" is not supplied

  if (is.null(var_in)) {
    var_in <- matrix(rep(1, nrow(var_out)))

    colnames(var_in) <- c("var_in")
  }

  if (all(class(var_in) == "numeric")) {
    var_in <- matrix(var_in)

    colnames(var_in) <- c("var_in")
  }

  # if weights are not supplied, assign 1
  if (is.null(weight_out)) {
    weight_out <- rep(1, ncol(var_out))
  }

  if (is.null(weight_in)) {
    weight_in <- matrix(1, ncol(var_out), ncol(var_in))
  }

  if (all(class(weight_in) == "numeric")) {
    weight_in <- matrix(weight_in)
  }

  # small number to add to zeros
  f <- 0

  # calculate in weights
  weight_in_list <-
    lapply(seq_len(nrow(weight_in)), function(i) {
      t(weight_in[i, ] * t(var_in))
    })

  # apply in weights
  wv_list <- lapply(seq_len(ncol(var_out)), function(i) {
    weight_mat <- weight_in_list[[i]]

    wm_list <- lapply(seq_len(ncol(weight_mat)), function(j) {
      var_out[, i] * weight_mat[, j]
    })

    wm_mat <- do.call(cbind, wm_list)

    out <-
      lapply(seq_len(nrow(wm_mat)), function(k) {
        sum(wm_mat[k, ])
      })

    do.call(rbind, out)
  })

  weighted_vars <- do.call(cbind, wv_list)

  # weight and average variables

  scaled_vars <-
    apply(weighted_vars, 2, function(x) {
      ps <- caret::preProcess(
        data.frame(x),
        method = c("range"),
        rangeBounds = c(f, 1)
      )

      predict(ps, data.frame(x))[, 1]
    })

  pci <- apply(scaled_vars, 1, am_mean, weight_out, na.rm = TRUE)

  rank <- rank(-pci)

  out <- data.frame(sp, pci, rank)
}
