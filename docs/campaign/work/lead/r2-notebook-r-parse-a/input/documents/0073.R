resample.CoxBoost <- function(
  time,
  status,
  x,
  rep = 100,
  maxstepno = 200,
  multicore = TRUE,
  mix.list = c(0.001, 0.01, 0.05, 0.1, 0.25, 0.35, 0.5, 0.7, 0.9, 0.99),
  stratum,
  stratnotinfocus = 0,
  penalty = sum(status) * (1 / 0.02 - 1),
  criterion = "hscore",
  unpen.index = NULL,
  trace = FALSE
) {
  rep <- rep
  trainind <- list()
  for (i in 1:rep) {
    trainind[[length(trainind) + 1]] <- sample(
      seq_len(nrow(x)),
      round(nrow(x) * 0.632),
      replace = F
    )
  }

  out <- list()
  for (iter in 1:rep) {
    message('iter=', iter)
    outbeta <- c()
    outCV.opt <- c()
    for (mix.prop in mix.list) {
      message('weight=', mix.prop)
      obs.weights <- rep(1, length(status))
      case.weights <- ifelse(stratum == stratnotinfocus, mix.prop, 1)
      obs.weights <- case.weights / sum(case.weights) * length(case.weights)

      CV <- cv.CoxBoost(
        time = time[trainind[[iter]]],
        status = status[trainind[[iter]]],
        x = x[trainind[[iter]], ],
        stratum = stratum[trainind[[iter]]],
        unpen.index = unpen.index,
        coupled.strata = FALSE,
        weights = obs.weights[trainind[[iter]]],
        maxstepno = maxstepno,
        K = 10,
        penalty = penalty,
        standardize = TRUE,
        trace = trace,
        multicore = multicore,
        criterion = criterion
      )

      CB <- CoxBoost(
        time = time[trainind[[iter]]],
        status = status[trainind[[iter]]],
        x = x[trainind[[iter]], ],
        stratum = stratum[trainind[[iter]]],
        unpen.index = unpen.index,
        coupled.strata = FALSE,
        weights = obs.weights[trainind[[iter]]],
        stepsize.factor = 1,
        stepno = CV$optimal.step,
        penalty = penalty,
        standardize = TRUE,
        trace = trace,
        criterion = criterion
      )
      outbeta <- c(outbeta, CB$model[[1]][[5]][nrow(CB$model[[1]][[5]]), ])
      outCV.opt <- c(outCV.opt, CV$optimal.step)
    }
    out[[iter]] <- list(beta = outbeta, CV.opt = outCV.opt)
  }

  out
}