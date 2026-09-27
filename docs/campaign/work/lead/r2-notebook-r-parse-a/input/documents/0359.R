CST_Intbc <- function(
  exp,
  obs,
  exp_cor = NULL,
  target_grid,
  bc_method,
  int_method = NULL,
  points = NULL,
  method_point_interp = NULL,
  lat_dim = "lat",
  lon_dim = "lon",
  sdate_dim = "sdate",
  member_dim = "member",
  time_dim = "time",
  region = NULL,
  ncores = NULL,
  loocv = TRUE,
  ...
) {
  if (!inherits(exp, "s2dv_cube")) {
    stop("Parameter 'exp' must be of the class 's2dv_cube'")
  }

  if (!inherits(obs, "s2dv_cube")) {
    stop("Parameter 'obs' must be of the class 's2dv_cube'")
  }

  res <- Intbc(
    exp = exp$data,
    obs = obs$data,
    exp_cor = exp_cor$data,
    exp_lats = exp$coords[[lat_dim]],
    exp_lons = exp$coords[[lon_dim]],
    obs_lats = obs$coords[[lat_dim]],
    obs_lons = obs$coords[[lon_dim]],
    target_grid = target_grid,
    int_method = int_method,
    bc_method = bc_method,
    points = points,
    source_file_exp = exp$attrs$source_files[1],
    source_file_obs = obs$attrs$source_files[1],
    method_point_interp = method_point_interp,
    lat_dim = lat_dim,
    lon_dim = lon_dim,
    sdate_dim = sdate_dim,
    member_dim = member_dim,
    time_dim = time_dim,
    region = region,
    ncores = ncores,
    loocv = loocv,
    ...
  )

  # Modify data, lat and lon in the origina s2dv_cube, adding the downscaled data
  obs$data <- res$obs
  obs$dims <- dim(obs$data)
  obs$coords[[lon_dim]] <- res$lon
  obs$coords[[lat_dim]] <- res$lat

  if (is.null(exp_cor)) {
    exp$data <- res$data
    exp$dims <- dim(exp$data)
    exp$coords[[lon_dim]] <- res$lon
    exp$coords[[lat_dim]] <- res$lat

    res_s2dv <- list(exp = exp, obs = obs)
  } else {
    exp_cor$data <- res$data
    exp_cor$dims <- dim(exp_cor$data)
    exp_cor$coords[[lon_dim]] <- res$lon
    exp_cor$coords[[lat_dim]] <- res$lat

    res_s2dv <- list(exp = exp_cor, obs = obs)
  }

  return(res_s2dv)
}