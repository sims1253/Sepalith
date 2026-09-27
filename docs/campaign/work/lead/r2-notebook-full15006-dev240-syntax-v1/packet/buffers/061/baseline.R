#' @title Run a GEE model for data from a crossover experiment
#'
#' @description Provides a GEE model for the data of a crossover design with S
#' sequences of T periods. There must be one observation of each
#' experimental unit in each period.
#'
#'
#' @param response A character string specifying the name of the response variable of the crossover
#' experimental design
#' @param period A character string specifying the name of vector with the
#' observation period of the responses of  the crossover experimental design
#' @param treatment A character string specifying the name of vector with the
#' treatment applied at each observation of the crossover experimental design
#' @param id A  character string specifying the name of vector which identifies
#' the experimental units. The length of ‘id’
#'  should be the same as the number of observations. Data are assumed to be sorted so
#'   that observations on each cluster appear as contiguous rows in data. If data
#'   is not sorted this way, the function will not identify the clusters correctly.
#'   If data is not sorted this way, a warning will be issued.
#' @param carry A vector of  character string specifying the name set of dummy
#'  variables that indicates the treatment applied
#' in the previous period of each experimental unit. They must be 0 in period 1
#' @param covar A vector of  character string specifying the name of possible
#' covariates of the crossover experimental design
#' @param data A data frame with all the variables of the crossover experimental design
#' @param family See corresponding documentation to \code{glm}
#' @param correlation 	a character string specifying the correlation structure.
#'   The following are permitted: "independence", "fixed", "stat_M_dep",
#'  "non_stat_M_dep", "exchangeable", "AR-M" and "unstructured"
#' @param Mv When correlation is "stat_M_dep", "non_stat_M_dep", or "AR-M"
#'   then Mv must be specified.
#' @param formula A formula related the response variable with the explanatory
#'  variables. If it is \code{NULL}, formula
#'  \code{response~period+treatment+carry+covar} will be evaluated.
#'
#' @return A list with:
#' \item{QIC}{The QIC of the model: The model are fitted by \code{geeglm}.}
#' \item{model}{The model fitted by \code{geeglm}.}
#'
#' @references Cruz, N. A., López Pérez, L. A., & Melo, O. O. (2023).
#' Analysis of cross-over experiments with count data in
#' the presence of carry-over effects. Statistica Neerlandica,
#'  77(4), 516-542.
#' @source https://doi.org/10.1111/stan.12295
#' @examples
#' data(Water)
#' model <- CrossGEE(response="LCC", covar=c("Age"), period="Period",
#'                   treatment = "Treatment", id="ID", carry="Carry_Agua",
#'                   family=gaussian(),correlation ="AR-M", Mv=1 ,data=Water)
#'
#' model$QIC
#' model$model
#'
#' ## Aproximate p-values
#' (pvalues <- 2 * pnorm(abs(coef(summary(model$model))[,5]), lower.tail = FALSE))
#'
#'summary(model$model)
#' @export
#' @importFrom dplyr "%>%"
#' @importFrom stats "gaussian"

CrossGEE <- function(response,period,treatment,id,carry, covar=NULL ,data,
                     family=gaussian(), correlation="independence",
                     formula=NULL, Mv=1){
  totalVar <- c(response, period, treatment,carry, covar, id)
  if(sum(totalVar %in% names(data))!=length(totalVar)){
    stop("Some variables are not present in data")
  }
  data <- data %>% dplyr::select_at(totalVar) |>
    dplyr::arrange_at(c(id, period)) %>%
    data.frame() %>% stats::na.exclude()
  data["id"] <- as.numeric(as.factor(data[,id]))
  if(is.null(formula)){
    form1 <- stats::as.formula(paste(response,
                              paste(c(period, treatment,carry, covar),
                                    collapse=" + "), sep=" ~ "))
  }else{
    form1 <- formula}
  model1 <- gee::gee(formula=form1, family=family, corstr = correlation,
                            id=id, data=data, Mv=Mv)
  QICmodels <- data.frame(computeqic(model1))
  names(QICmodels) <- "QICs"
  return(list(QIC =QICmodels, model=model1))
}
