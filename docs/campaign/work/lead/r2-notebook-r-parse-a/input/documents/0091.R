#' Cluster Micro-Panel (longitudinal) Data employing the CluMP algorithm
#'
#' This function clusters Micro-Panel (longitudinal) Data (or trajectories) to a pre-defined number of clusters by employing Feature-Based Clustering of Micro-Panel (longitudinal) Data algorithm called CluMP (see Reference). Currently, only univariate clustering analysis is available.
#' @param formula A two-sided \code{\link[stats]{formula}} object with a numeric clustering variable (Y) on the left of a ~ separator and the time (numeric) variable on the right. Time is measured from the start of the follow-up period (baseline). Any time units are possible.
#' @param group A grouping factor variable (vector), i.e. single identifier for each individual (trajectory).
#' @param data A data frame containing the variables named in the \code{formula} and \code{group} arguments.
#' @param cl_numb An integer, positive number (scalar) specifying the number of clusters. The \code{\link{OptiNum}} function can be used to determine the optimal number of clusters according to common evaluation criteria (indices).
#' @param base_val Indicates whether include a value at zero time point as an additional clustering variable. Default is \emph{FALSE} and the standard number (7) of clustering parameters is used.
#' @param method A method which use in hierarhical clustering, same as in \code{\link[stats]{hclust}} function, namely "ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid". Default is "ward.D".
#' @keywords CluMP
#' @return Cluster Micro-Panel data. The output is the \code{\link[base]{list}} of 5 components which contain results from clustering.
#' @source Sobisek, L., Stachova, M., Fojtik, J. (2018) Novel Feature-Based Clustering of Micro-Panel Data (CluMP). Working paper version online: www.arxiv.org
#' @export
#' @import data.table
#' @examples
#' data <- GeneratePanel(n = 100, Param = ParamLinear, NbVisit = 10)
#' CluMP(formula = Y ~ Time, group = "ID", data = data, cl_numb = 3,
#' base_val = FALSE, method = "ward.D")
#'
#' CluMP(formula = Y ~ Time, group = "ID", data = data, cl_numb = 3,
#' base_val = TRUE, method = "ward.D")
#'
#'
CluMP <- function(formula, group, data, cl_numb = NA, base_val = FALSE, method = "ward.D"){

  # define global variables
  CluMP_ID = CluMP_X1 = CluMP_Y = ID = Visit = X1 = X1_ann = Y = Y.x = Y.y = Y_ci =
    abs_angle_radian = abs_change = abs_change_ann = angle_radian = best = bestVal =
    cluster = cos_denom = cos_nom  = cosinus = f_up = mean_Time = mean_Y =
    memb_CluMP = nVisit = number = obsah_trojuh = sd_Y = slope =
    slope_first_last = timepoint = value = . = .. = ..colour =
    ..cols = ..cont_vars = ..group = ..scale_cols = Time = NULL


  if(!any(all.vars(formula) %in% names(data))) {
    varErr <- all.vars(formula)[!(all.vars(formula) %in% names(data))]
    if(length(varErr) == 1) {
      stop(paste0('Variable "', varErr, '" does not present in the data.'))
    }else {
      stop(paste0('Variables "', paste(varErr, collapse = ', '), '" do not present in the data.'))
    }
  }

  # Modelframe from formula
  mf <- stats::model.frame(formula = formula, data = data, na.action = NULL)
  variables <- names(mf)
  if(length(variables) > 2) {
    stop('It is not possible to cluster multiple variables. Consider using one of them.')
  }
  if(!any(variables %in% names(data))) {
    varErr <- variables[!(variables %in% names(data))]
    stop(paste0('Variable "', varErr, '" does not present in the data.'))
  }
  names(mf) <- c("CluMP_Y", "CluMP_X1")
  mf <- cbind(mf, setDT(data)[, ..group])
  names(mf)[3] <- "CluMP_ID"
  mf_complete <- mf[stats::complete.cases(mf),]

  # save mf_complete as data table. In order to speed up computations
  mf_complete <- as.data.table(mf_complete)
  mf_complete <- setDT(mf_complete)[, if (.N > 2) .SD, by = CluMP_ID]
  mf_non_complete <- setDT(mf)[, if (.N <= 2) .SD, by = CluMP_ID]
  mf <- mf_complete



  if(nrow(mf_non_complete) > 1) {
    warning('Some of objects has less then 3 observations or contains NA values. Result data set is reduced.', call. = FALSE)
  }

  #Not defined entry
  if (!is.numeric(mf$CluMP_Y)){
    stop('Clustering variable should be numeric')
  }
  if (!is.numeric(mf$CluMP_X1)){
    stop('Time variable should be numeric')
  }
  if (is.na(cl_numb)) {
    warning('Number of clusters defined as 3. If you want different number of clusters define parameter ClNumb', call. = FALSE)
    cl_numb <- 3
  }
  if (!is.numeric(cl_numb) || cl_numb <= 1 || (cl_numb %% 1) != 0){
    stop('Number of clusters argument cl_numb should be an integer greater than 1')
  }
  if (cl_numb > length(unique(mf$CluMP_ID))){
    warning(paste0('Number of clusters cannot exceed the number of groups in data. The number of cluster  set as ', max(data$ID)), call. = FALSE)
    cl_numb <- length(unique(mf$CluMP_ID))
  }


  # CluMP prepare
  cluster_tmp <- setDT(mf)[, Visit := 1:.N, by = CluMP_ID
                           ][order(CluMP_X1)
                             ][max(Visit) >2]
  cluster_tmp <- setDT(cluster_tmp)[, ':='(
    base_val = data.table::first(CluMP_Y),
    obsah_trojuh = c(NA, diff(CluMP_Y)/diff(CluMP_X1)/2),
    slope = (CluMP_Y - data.table::first(CluMP_Y))/(CluMP_X1 - data.table::first(CluMP_X1)),
    slope_first_last = (data.table::last(CluMP_Y) - data.table::first(CluMP_Y))/(last(CluMP_X1) - data.table::first(CluMP_X1)),
    cos_nom = (data.table::last(CluMP_X1) - data.table::first(CluMP_X1))*(CluMP_X1 - data.table::first(CluMP_X1)) + (data.table::last(CluMP_Y) - data.table::first(CluMP_Y))*(CluMP_Y - data.table::first(CluMP_Y)),
    cos_denom = sqrt((data.table::last(CluMP_X1) - data.table::first(CluMP_X1))^2 + (data.table::last(CluMP_Y) - data.table::first(CluMP_Y))^2)*sqrt((CluMP_X1 - data.table::first(CluMP_X1))^2 + (CluMP_Y - data.table::first(CluMP_Y))^2)
    ), by = CluMP_ID]
  cluster_tmp <- setDT(cluster_tmp)[, cosinus := ifelse(cos_nom/cos_denom > 1, 1, cos_nom/cos_denom), ]
  cluster_tmp <- setDT(cluster_tmp)[, abs_angle_radian := acos(cosinus), ]
  cluster_tmp <- setDT(cluster_tmp)[, angle_radian := ifelse(slope_first_last < slope, 1, -1)*abs_angle_radian, ]

  cluster_tmp <- setDT(cluster_tmp)[, ':='(
    base_val = first(base_val),
    mean_triangle = mean(obsah_trojuh, na.rm = T),
    sd_triangle = stats::sd(obsah_trojuh, na.rm = T),
    mean_abs_triangle = mean(abs(obsah_trojuh), na.rm = T),
    sd_abs_triangle = stats::sd(abs(obsah_trojuh), na.rm = T),
    k = stats::cov(CluMP_Y, CluMP_X1)/stats::var(CluMP_X1),
    rel_pos = sum(diff(CluMP_Y) > 0, na.rm = T)/max(sum(diff(CluMP_Y) < 0, na.rm = T), 0.1)
  ), by = CluMP_ID]
  cluster_tmp <- cluster_tmp |>
    group_by(CluMP_ID) |>
    mutate(angle_radian = ifelse(test = all(is.na(angle_radian)),
                                 yes  = NA,
                                 no   = ifelse(test = max(angle_radian, na.rm = T) == max(abs(angle_radian), na.rm = T),
                                               yes  = max(abs(angle_radian), na.rm = T),
                                               no   = -max(abs(angle_radian), na.rm = T))))

  cols <- colnames(cluster_tmp)[c(1,5,13:19)]
  cluster_tmp <- setDT(cluster_tmp)[, ..cols]
  cluster_tmp <- cluster_tmp[!duplicated(cluster_tmp),]

  if (base_val == F) cluster_tmp <- setDT(cluster_tmp)[, base_val := NULL,]

  # Scaling values for clustering
  scale_cols <- colnames(cluster_tmp)[-1]
  cluster_norm_tmp <- setDT(cluster_tmp)[, (scale_cols) := lapply(.SD, scale), .SDcols = scale_cols]

  # Clustering
  tmp_comb <- suppressWarnings(amap::Dist(setDT(cluster_norm_tmp)[, ..scale_cols], method = "euclid"))
  tmp_comb <- stats::hclust(tmp_comb, method = method)
  cluster_norm_tmp$memb_CluMP <- stats::cutree(tmp_comb, k = cl_numb)

  cluster_tmp <- merge(cluster_tmp, cluster_norm_tmp[, c("CluMP_ID", "memb_CluMP")], by = "CluMP_ID")
  setcolorder(cluster_tmp, c("CluMP_ID", scale_cols, "memb_CluMP"))

  names(cluster_tmp)[1] <- group # put back old names

  cols <- c(group, "memb_CluMP")
  data <- merge(data, setDT(cluster_tmp)[, ..cols], by = group)

  data  <- data[order(Time)]

  return(list("CluMP" = cluster_tmp, "data" = data, "formula" = formula, "variables" = variables, "group" = group))
}
