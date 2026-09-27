CST_MultivarRMSE <- function(
  exp,
  obs,
  weight = NULL,
  memb_dim = 'member',
  dat_dim = 'dataset',
  sdate_dim = 'sdate',
  ftime_dim = 'ftime'
) {
  # s2dv_cube
  if (!is.list(exp) | !is.list(obs)) {
    stop("Parameters 'exp' and 'obs' must be lists of 's2dv_cube' objects")
  }
  if (!(all(sapply(exp, inherits, 's2dv_cube')))) {
    stop(
      "Elements of the list in parameter 'exp' must be of the class ",
      "'s2dv_cube', as output by CSTools::CST_Load."
    )
  }
  if (!(all(sapply(obs, inherits, 's2dv_cube')))) {
    stop(
      "Elements of the list in parameter 'obs' must be of the class ",
      "'s2dv_cube', as output by CSTools::CST_Load."
    )
  }
  if (length(exp) != length(obs)) {
    stop("Parameters 'exp' and 'obs' must be of the same length.")
  }

  nvar <- length(exp)
  if (nvar < 2) {
    stop(
      "Parameters 'exp' and 'obs'  must contain at least two",
      " s2dv objects for two different variables."
    )
  }
  for (j in 1:nvar) {
    if (
      !is.null(names(dim(exp[[j]]$data))) & !is.null(names(dim(obs[[j]]$data)))
    ) {
      if (all(names(dim(exp[[j]]$data)) %in% names(dim(obs[[j]]$data)))) {
        dimnames <- names(dim(exp[[j]]$data))
      } else {
        stop(
          "Dimension names of element 'data' from parameters 'exp'",
          " and 'obs' should be equal."
        )
      }
    } else {
      stop(
        "Element 'data' from parameters 'exp' and 'obs'",
        " should have dimmension names."
      )
    }
  }
  # weight
  if (is.null(weight)) {
    weight <- c(rep(1, nvar))
  } else if (!is.numeric(weight)) {
    stop("Parameter 'weight' must be numeric.")
  } else if (length(weight) != nvar) {
    stop(
      "Parameter 'weight' must have a length equal to the number ",
      "of variables."
    )
  }
  # memb_dim
  if (!is.null(memb_dim)) {
    if (!is.character(memb_dim)) {
      stop("Parameter 'memb_dim' must be a character string.")
    }
    if (
      !memb_dim %in% names(dim(exp[[1]]$data)) |
        !memb_dim %in% names(dim(obs[[1]]$data))
    ) {
      stop("Parameter 'memb_dim' is not found in 'exp' or in 'obs' dimension.")
    }
  } else {
    stop("Parameter 'memb_dim' cannot be NULL.")
  }
  # dat_dim
  if (!is.null(dat_dim)) {
    if (!is.character(dat_dim)) {
      stop("Parameter 'dat_dim' must be a character string.")
    }
    if (
      !dat_dim %in% names(dim(exp[[1]]$data)) |
        !dat_dim %in% names(dim(obs[[1]]$data))
    ) {
      stop("Parameter 'dat_dim' is not found in 'exp' or in 'obs' dimension.")
    }
  }
  # ftime_dim
  if (!is.null(ftime_dim)) {
    if (!is.character(ftime_dim)) {
      stop("Parameter 'ftime_dim' must be a character string.")
    }
    if (
      !ftime_dim %in% names(dim(exp[[1]]$data)) |
        !ftime_dim %in% names(dim(obs[[1]]$data))
    ) {
      stop("Parameter 'ftime_dim' is not found in 'exp' or in 'obs' dimension.")
    }
  } else {
    stop("Parameter 'ftime_dim' cannot be NULL.")
  }
  # sdate_dim
  if (!is.null(sdate_dim)) {
    if (!is.character(sdate_dim)) {
      stop("Parameter 'sdate_dim' must be a character string.")
    }
    if (
      !sdate_dim %in% names(dim(exp[[1]]$data)) |
        !sdate_dim %in% names(dim(obs[[1]]$data))
    ) {
      stop("Parameter 'sdate_dim' is not found in 'exp' or in 'obs' dimension.")
    }
  } else {
    stop("Parameter 'sdate_dim' cannot be NULL.")
  }
  # Variables
  obs_var <- unlist(lapply(exp, function(x) {
    x$attrs$Variable$varName
  }))

  exp_var <- unlist(lapply(exp, function(x) {
    x$attrs$Variable$varName
  }))

  if (all(exp_var != obs_var)) {
    stop("Variables in parameters 'exp' and 'obs' must be in the same order.")
  }

  mvrmse <- 0
  sumweights <- 0

  for (j in 1:nvar) {
    # seasonal average of anomalies
    AvgExp <- MeanDims(exp[[j]]$data, c(memb_dim, ftime_dim), na.rm = TRUE)
    AvgObs <- MeanDims(obs[[j]]$data, c(memb_dim, ftime_dim), na.rm = TRUE)
    # multivariate RMSE (weighted)
    rmse <- RMS(
      AvgExp,
      AvgObs,
      dat_dim = dat_dim,
      time_dim = sdate_dim,
      conf = FALSE
    )$rms
    stdev <- sd(AvgObs)
    mvrmse <- mvrmse + (rmse / stdev * as.numeric(weight[j]))
    sumweights <- sumweights + as.numeric(weight[j])
  }
  mvrmse <- mvrmse / sumweights

  # names(dim(mvrmse)) <- c(dimnames[1], dimnames[1], 'statistics', dimnames[5 : 6])
  exp_Datasets <- unlist(lapply(exp, function(x) {
    x$attrs[[which(names(x$attrs) == 'Datasets')]]
  }))
  exp_source_files <- unlist(lapply(exp, function(x) {
    x$attrs[[which(names(x$attrs) == 'source_files')]]
  }))
  obs_Datasets <- unlist(lapply(obs, function(x) {
    x$attrs[[which(names(x$attrs) == 'Datasets')]]
  }))
  obs_source_files <- unlist(lapply(obs, function(x) {
    x$attrs[[which(names(x$attrs) == 'source_files')]]
  }))

  exp1 <- exp[[1]]
  exp1$data <- mvrmse
  exp1$attrs$Datasets <- c(exp_Datasets, obs_Datasets)
  exp1$attrs$source_files <- c(exp_source_files, obs_source_files)
  exp1$attrs$Variable$varName <- as.character(exp_var)
  exp1$attrs$Variable$metadata <- c(
    exp1$attrs$Variable$metadata,
    exp[[2]]$attrs$Variable$metadata
  )
  return(exp1)
}