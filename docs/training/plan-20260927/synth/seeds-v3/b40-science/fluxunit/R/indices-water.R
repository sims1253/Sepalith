#' Rating curves for the water quality index
#'
#' Each curve maps a measured value to a quality rating between 0 and 100
#' by linear interpolation between the listed points.
#'
#' @return A named list with one element per parameter (`do`, `ph`,
#'   `turbidity`, `nitrate`, `phosphate`), each a list with numeric
#'   vectors `x` and `q`.
#' @keywords internal
wqi_rating_curves <- function() {
  list(
    do = list(
      x = c(0, 20, 40, 60, 80, 100, 120, 140),
      q = c(0, 15, 35, 60, 85, 100, 90, 70)
    ),
    ph = list(
      x = c(2, 4, 6, 7, 7.5, 8, 9, 10, 12),
      q = c(0, 10, 55, 90, 93, 85, 50, 20, 0)
    ),
    turbidity = list(
      x = c(0, 5, 10, 25, 50, 100),
      q = c(100, 85, 75, 55, 30, 5)
    ),
    nitrate = list(
      x = c(0, 1, 5, 10, 20, 50, 100),
      q = c(100, 95, 70, 50, 30, 10, 2)
    ),
    phosphate = list(
      x = c(0, 0.1, 0.3, 0.6, 1, 3, 10),
      q = c(100, 90, 70, 50, 35, 10, 2)
    )
  )
}

#' Water quality index from field measurements
#'
#' Rates each supplied parameter on its curve from [wqi_rating_curves()]
#' and averages the ratings with weights. Missing parameters are dropped
#' and the remaining weights rescaled.
#'
#' @param do Dissolved oxygen, percent saturation.
#' @param ph pH.
#' @param turbidity Turbidity, NTU.
#' @param nitrate Nitrate, mg/l.
#' @param phosphate Total phosphate, mg/l.
#' @return A data frame with columns `q_do`, `q_ph`, `q_turbidity`,
#'   `q_nitrate`, `q_phosphate` and `wqi`.
#' @export
water_quality_index <- function(do = NULL, ph = NULL, turbidity = NULL,
                                nitrate = NULL, phosphate = NULL) {
  curves <- wqi_rating_curves()
  weights <- c(do = 0.30, ph = 0.20, turbidity = 0.15, nitrate = 0.15, phosphate = 0.20)
  lens <- lengths(list(do, ph, turbidity, nitrate, phosphate))
  n <- max(lens)
  if (n == 0L) {
    stop("Supply at least one parameter.", call. = FALSE)
  }

  # Rate `do`
  q_do <- rep(NA_real_, n)
  if (!is.null(do)) {
    q_do <- stats::approx(curves$do$x, curves$do$q, xout = do, rule = 2)$y
  }

  # Rate `ph`
  q_ph <- rep(NA_real_, n)
  if (!is.null(ph)) {
    q_ph <- stats::approx(curves$ph$x, curves$ph$q, xout = ph, rule = 2)$y
  }

  # Rate `turbidity`
  q_turbidity <- rep(NA_real_, n)
  if (!is.null(turbidity)) {
    q_turbidity <- stats::approx(curves$turbidity$x, curves$turbidity$q,
                                 xout = turbidity, rule = 2)$y
  }

  # Rate `nitrate`
  q_nitrate <- rep(NA_real_, n)
  if (!is.null(nitrate)) {
    q_nitrate <- stats::approx(curves$nitrate$x, curves$nitrate$q,
                               xout = nitrate, rule = 2)$y
  }

  # Rate `phosphate`
  q_phosphate <- rep(NA_real_, n)
  if (!is.null(phosphate)) {
    q_phosphate <- stats::approx(curves$phosphate$x, curves$phosphate$q,
                                 xout = phosphate, rule = 2)$y
  }

  q <- cbind(do = q_do, ph = q_ph, turbidity = q_turbidity,
             nitrate = q_nitrate, phosphate = q_phosphate)
  w <- matrix(weights[colnames(q)], nrow = n, ncol = ncol(q), byrow = TRUE)
  w[is.na(q)] <- 0
  wqi <- rowSums(q * w, na.rm = TRUE) / rowSums(w)
  data.frame(
    q_do = q_do, q_ph = q_ph, q_turbidity = q_turbidity,
    q_nitrate = q_nitrate, q_phosphate = q_phosphate, wqi = wqi
  )
}

#' Trophic state index of a lake
#'
#' One sub-index per indicator on the logarithmic trophic scale; the
#' overall value is their mean.
#'
#' @param chl Chlorophyll a, ug/l.
#' @param tp Total phosphorus, ug/l.
#' @param secchi Secchi depth, m.
#' @return A data frame with columns `tsi_chl`, `tsi_tp`, `tsi_secchi`,
#'   `tsi` and `state`.
#' @export
trophic_state_index <- function(chl, tp, secchi) {
  n <- max(length(chl), length(tp), length(secchi))

  # Sub-index `chl`
  if (any(chl <= 0, na.rm = TRUE)) {
    stop("`chl` must be strictly positive.", call. = FALSE)
  }
  tsi_chl <- rep_len(9.81 * log(chl) + 30.6, n)

  # Sub-index `tp`
  if (any(tp <= 0, na.rm = TRUE)) {
    stop("`tp` must be strictly positive.", call. = FALSE)
  }
  tsi_tp <- rep_len(14.42 * log(tp) + 4.15, n)

  # Sub-index `secchi`
  if (any(secchi <= 0, na.rm = TRUE)) {
    stop("`secchi` must be strictly positive.", call. = FALSE)
  }
  tsi_secchi <- rep_len(60 - 14.41 * log(secchi), n)

  tsi <- rowMeans(cbind(tsi_chl, tsi_tp, tsi_secchi), na.rm = TRUE)
  state <- cut(tsi,
    breaks = c(-Inf, 40, 50, 70, Inf),
    labels = c("oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic")
  )
  data.frame(
    tsi_chl = tsi_chl,
    tsi_tp = tsi_tp,
    tsi_secchi = tsi_secchi,
    tsi = tsi,
    state = state
  )
}

#' Ion balance of a water sample
#'
#' Converts each major ion from mg/l to meq/l with its molar mass and
#' charge, sums cations and anions and reports the balance error.
#'
#' @param ca,mg,na,k Cation concentrations, mg/l.
#' @param hco3,cl,so4,no3 Anion concentrations, mg/l.
#' @return A list with elements `cations` and `anions` (named meq/l
#'   vectors), `sum_cations`, `sum_anions` and `error_pct`.
#' @export
ion_balance <- function(ca, mg, na, k, hco3, cl, so4, no3) {
  molar_mass <- c(ca = 40.078, mg = 24.305, na = 22.990, k = 39.098,
                  hco3 = 61.017, cl = 35.453, so4 = 96.06, no3 = 62.004)
  charge <- c(ca = 2, mg = 2, na = 1, k = 1, hco3 = 1, cl = 1, so4 = 2, no3 = 1)

  # Cations
  cations <- c(
    ca = ca / molar_mass[["ca"]] * charge[["ca"]],
    mg = mg / molar_mass[["mg"]] * charge[["mg"]],
    na = na / molar_mass[["na"]] * charge[["na"]],
    k = k / molar_mass[["k"]] * charge[["k"]]
  )

  # Anions
  anions <- c(
    hco3 = hco3 / molar_mass[["hco3"]] * charge[["hco3"]],
    cl = cl / molar_mass[["cl"]] * charge[["cl"]],
    so4 = so4 / molar_mass[["so4"]] * charge[["so4"]],
    no3 = no3 / molar_mass[["no3"]] * charge[["no3"]]
  )

  sum_cations <- sum(cations)
  sum_anions <- sum(anions)
  list(
    cations = cations,
    anions = anions,
    sum_cations = sum_cations,
    sum_anions = sum_anions,
    error_pct = 100 * (sum_cations - sum_anions) / (sum_cations + sum_anions)
  )
}

#' Classify an index value into named bands
#'
#' @param value A numeric vector of index values.
#' @param index Which index the values come from: `"wqi"`, `"tsi"`,
#'   `"ion_error"` or `"comfort"`.
#' @return A character vector of band labels.
#' @export
classify_index <- function(value, index = c("wqi", "tsi", "ion_error", "comfort")) {
  index <- match.arg(index)
  switch(index,
    wqi = {
      breaks <- c(-Inf, 25, 50, 70, 90, Inf)
      labels <- c("very_bad", "bad", "medium", "good", "excellent")
      as.character(cut(value, breaks = breaks, labels = labels, right = FALSE))
    },
    tsi = {
      breaks <- c(-Inf, 40, 50, 70, Inf)
      labels <- c("oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic")
      as.character(cut(value, breaks = breaks, labels = labels, right = TRUE))
    },
    ion_error = {
      breaks <- c(-Inf, 2, 5, 10, Inf)
      labels <- c("excellent", "acceptable", "doubtful", "rejected")
      as.character(cut(abs(value), breaks = breaks, labels = labels, right = TRUE))
    },
    comfort = {
      breaks <- c(-Inf, 0.25, 0.5, 0.75, Inf)
      labels <- c("poor", "fair", "good", "ideal")
      as.character(cut(value, breaks = breaks, labels = labels, right = FALSE))
    }
  )
}
