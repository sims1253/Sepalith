#' Parameter estimation based on the Clayton copula for serial dependence and the Clayton copula for dependent censoring with the Weibull distributions
#'
#' Perform two-stage estimation based on the Clayton copula C_theta for serial dependence and the Clayton copula tilde(C)_alpha for dependent censoring with the marginal distributions Weib(scale1, shape1) and Weib(scale2, shape2). The jackknife method estimates the asymptotic covariance matrix. Parametric bootstrap is applied while doing Kolmogorov-Smirnov tests and Cramer-von Mises test. The guide for using this function shall be explained by Huang (2019), and Huang, Wang and Emura (2020).
#'
#' @usage ClaytonClayton.Weibull.MLE(subject, t.event, event, t.death, death, stageI, Weibull.plot,
#'                                   jackknife, plot, GOF, GOF.plot, rep.GOF, digit)
#'
#' @import survival stats graphics
#' @importFrom survival Surv survfit
#' @importFrom stats nlm cor runif
#' @importFrom graphics par axis polygon lines
#'
#' @param subject a vector for numbers of subject
#' @param t.event a vector for event times
#' @param event a vector for event indicator (=1 if recurrent; =0 if censoring)
#' @param t.death a vector for death times
#' @param death a vector for death indicator (=1 if death; =0 if censoring)
#' @param stageI an option to select MLE or LSE method for the 1st-stage optimization
#' @param Weibull.plot if TRUE, show the Weibull probability plot
#' @param jackknife if TRUE, the jackknife method is used for estimate covariance matrix (default = TRUE)
#' @param plot if TRUE, the plots for marginal distributions are shown (default = FALSE)
#' @param GOF if TRUE, show the p-values for KS-test and CvM-test
#' @param GOF.plot if TRUE, show the model diagnostic plot
#' @param rep.GOF repetition number of parametric bootstrap
#' @param digit accurate to some decimal places
#'
#' @return A list with the following elements:
#' \item{Sample_size}{Sample size N}
#' \item{Case}{Count for event occurences}
#' \item{scale1}{Scale parameter for Weib(scale1, shape1)}
#' \item{shape1}{Shape parameter for Weib(scale1, shape1)}
#' \item{scale2}{Scale parameter for Weib(scale2, shape2)}
#' \item{shape2}{Shape parameter for Weib(scale2, shape2)}
#' \item{theta}{Copula parameter for the Clayton copula C_theta}
#' \item{alpha}{Copula parameter for the Clayton copula tilde(C)_alpha}
#' \item{COV}{Asymptotic covariance estimated by the jackknife method}
#' \item{KS}{Kolmogorov-Smirnov test statistics}
#' \item{p.KS}{P-values for Kolmogorov-Smirnov tests}
#' \item{CM}{Cramer-von Mises test statistics}
#' \item{p.CM}{P-values for Cramer-von Mises tests}
#' \item{Convergence}{Convergence results for each stage}
#' \item{Jackknife_error}{Count for error in jackknife repititions}
#' \item{Log_likelihood}{Log-likelihood values}
#'
#' @details When jackknife=FALSE, the corresponding standard error and confidence interval values are shown as NA.
#'
#' @examples
#' data = ClaytonClayton.Weibull.data(N = 30, scale1 = 1, shape1 =0.5, theta = 2,
#'                                    scale2 = 0.45, shape2 = 0.5, alpha = 2, b = 10, l = 300)
#' \donttest{
#' ClaytonClayton.Weibull.MLE(subject = data$Subject,
#'                            t.event = data$T_ij, event = data$delta_ij,
#'                            t.death = data$T_i_star, death = data$delta_i_star,
#'                            jackknife= TRUE, plot = TRUE)
#' }
#'
#' @author Xinwei Huang
#' @references  Huang XW, Wang W, Emura T (2020) A copula-based Markov chain model for serially dependent event times with a dependent terminal event. Japanese Journal of Statistics & Data Science. Accepted.
#'
#' @export


ClaytonClayton.Weibull.MLE = function(subject, t.event, event, t.death, death, stageI = "MLE", Weibull.plot = FALSE,
                                jackknife = TRUE, plot = FALSE, GOF = FALSE, GOF.plot = FALSE,
                                rep.GOF = 200, digit = 5){

  #####Declare variables
  case = NA
  p0_D = p0 = NA
  N = T_i1 = tau_theta_0 = tau_alpha_0 = ni_status = NA
  result.stageI = result.stageII = NA
  result.convergence = c(StageI = NA, StageII = NA)
  #[1]=StageI [2]=StageII
  result.estimate.tilde = result.estimate = result.lowerbound = result.upperbound = result.se = rep(NA, 6)
  #[1]=scale1 [2]=shape1 [3]=theta [4]=alpha [5]=scale2 [6]=shape2
  jackknife.error = 0
  result.cov.tilde = NA
  Theta_k = NA
  KS.D = CM.D = KS.R = CM.R = NA
  KS.D_p = CM.D_p = KS.R_p = CM.R_p = NA
  #####Basic
  N = length(unique(subject))
  for(i in c(1:N)){
    no = which(subject == unique(subject)[i])
    T_i1[i] = t.event[no][1]
    ni_status[i] = ifelse(length(no)>1, 1, 0)
  }
  case =  cbind.data.frame(
    "Terminal event" = c("Death", "Censoring", "Death", "Censoring"),
    "event" = c("1", "1", "> 1", "> 1"),
    "Total" = c(sum(death == 1 & ni_status == 0), sum(death == 0 & ni_status == 0),
                sum(death == 1 & ni_status == 1), sum(death == 0 & ni_status == 1)),
    row.names = c("A", "B", "C", "D")
  )

  #####Begin Stage I

  #Nelson Aalen estimator function
  NelsonAalen = function(time, status){

    temp = as.data.frame(cbind(time, status))

    table = as.data.frame(  matrix(NA, ncol = 3, nrow = 1) )
    colnames(table) = c("t_i", "n_i", "d_i")
    for( t in sort(unique(time)) ){
      table[which(sort(unique(time)) == t), ] =
        c(t, sum(temp$time >= t), sum(temp[which(temp$time == t),2]) )
    }

    table = cbind( table, Hazard_i = cumsum(table$d_i/table$n_i) )

    return(table)
  }

  #LSE
  NA.estimator = NelsonAalen(time = t.event , status = event)
  result.LSE = lm(log(NA.estimator$Hazard_i[NA.estimator$d_i>0])~log(NA.estimator$t_i[NA.estimator$d_i>0]))

  #Initial value
  p0_D = log( c(1/mean(t.death), 1) )

  #Likelihood for margin D
  logL_D = function(par){

    delta_i_star = death
    T_i_star = t.death

    scale2 = exp(par[1])
    shape2 = exp(par[2])

    l = sum(delta_i_star*log(scale2*shape2) + (shape2-1)*delta_i_star*log(T_i_star)-scale2*T_i_star^shape2)

    return(-l)

  }

  #Optimization
  result.stageI = nlm(logL_D, p0_D)

  #Output
  if(stageI == "MLE"){
    result.estimate.tilde[5] = result.stageI$estimate[1]
    result.estimate.tilde[6] = result.stageI$estimate[2]
    result.convergence[1] = ifelse(result.stageI$code==1, 1, 0)
    result.estimate = exp(result.estimate.tilde)
  }

  if(stageI == "LSE"){
    result.estimate.tilde[5] = log(as.numeric(exp(result.LSE$coefficients[1])))
    result.estimate.tilde[6] = log(as.numeric(result.LSE$coefficients[2]))
    result.estimate = exp(result.estimate.tilde)
  }
  if(Weibull.plot == "TRUE"){
    par(mar=c(4,5,2,2))
    plot(log(NA.estimator$Hazard_i[NA.estimator$d_i>0])~log(NA.estimator$t_i[NA.estimator$d_i>0]),
         ylab = expression( paste("log(",hat(Lambda),"(",T[i]^"*","))") ),
         xlab = expression( paste("log(",T[i]^"*",")") ), main = "Weibull probability plot" )
    abline(result.LSE)
    legend("topleft", legend = c("Fitted"), lty = 1, bty = "n")
  }

  #####End Stage I

  #####Begin Stage II
  #Initial value
  tau_theta_0 = cor(t.event[-1], t.event[-length(t.event)], method = "kendall") #sample kendall's tau for Xij
  tau_alpha_0 = cor(T_i1, t.death, method = "kendall") #sample kendall's tau for Xi1 and Di
  p0 = log( c( 1/mean(t.event), 1,
               min(max(2 * tau_theta_0/(1 - tau_theta_0), 0.001), 12),
               min(max(2 * tau_alpha_0/(1 - tau_alpha_0), 0.001), 12)) )

  #Full likelihood
  logL = function(par){

    scale2 = result.estimate[5]
    shape2 = result.estimate[6]

    scale1 = exp(par[1])
    shape1 = exp(par[2])
    theta = max(exp(par[3]),0.00001)
    alpha = max(exp(par[4]),0.00001)

    #corresponding D function
    A_theta = function(s,t){ exp(theta*s) + exp(theta*t) - 1 }
    D_theta_01 = function(s,t){ A_theta(s,t)^(-1/theta-1)*exp(theta*t) }
    D_theta_11 = function(s,t){ (theta+1)*A_theta(s,t)^(-1/theta-2)*exp(theta*s)*exp(theta*t) }
    A.tilde_alpha = function(s,t){ exp(alpha*s) + exp(alpha*t) - 1 }
    D.tilde_alpha = function(s,t){ A.tilde_alpha(s,t)^(-1/alpha) }
    D.tilde_alpha_01 = function(s,t){ A.tilde_alpha(s,t)^(-1/alpha-1)*exp(alpha*t) }
    D.tilde_alpha_10 = function(s,t){ A.tilde_alpha(s,t)^(-1/alpha-1)*exp(alpha*s) }
    D.tilde_alpha_11 = function(s,t){ (alpha+1)*A.tilde_alpha(s,t)^(-1/alpha-2)*exp(alpha*s)*exp(alpha*t) }

    l = 0

    for(i in c(1:N)){

      l_i = 0

      no = which(subject == unique(subject)[i])
      n_i = length(no)
      T_ij = t.event[no]
      delta_ij = event[no]
      T_i_star = t.death[i]
      delta_i_star = death[i]
      #marginal Weibull for Xij
      r.T_ij = scale1*shape1*T_ij^(shape1-1)
      R.T_ij = scale1*T_ij^shape1
      #marginal Weibull for Dij
      lambda.T_i_star = scale2*shape2*T_i_star^(shape2-1)
      Lambda.T_i_star = scale2*T_i_star^shape2
      #likelihood for ni=1
      l_i = delta_i_star*log(lambda.T_i_star) + delta_ij[1]*log(r.T_ij[1]) +
        delta_i_star*delta_ij[1]*log( D.tilde_alpha_11(R.T_ij[1], Lambda.T_i_star) ) +
        delta_i_star*(1-delta_ij[1])*log( D.tilde_alpha_01(R.T_ij[1], Lambda.T_i_star) ) +
        (1 - delta_i_star)*log( D.tilde_alpha_10(R.T_ij[1], Lambda.T_i_star) )+
        (1 - delta_i_star)*(1-delta_ij[1])*log( D.tilde_alpha(R.T_ij[1], Lambda.T_i_star) )
      #likelihood for ni>=2
      if(n_i >= 2){
        l_i = l_i + sum(
          R.T_ij[1:(n_i-1)] + delta_ij[2:n_i]*log(r.T_ij[2:n_i]) +
            delta_ij[2:n_i]*log( D_theta_11(R.T_ij[2:n_i],R.T_ij[1:(n_i-1)]) ) +
            (1 - delta_ij[2:n_i])*log( D_theta_01(R.T_ij[2:n_i],R.T_ij[1:(n_i-1)]) )
        )
      }

      l = l + l_i
    }

    return(-l)

  }

  #Optimization
  result.stageII = nlm(f = logL, p = p0)

  #Output
  result.estimate.tilde[1] = result.stageII$estimate[1]
  result.estimate.tilde[2] = result.stageII$estimate[2]
  result.estimate.tilde[3] = result.stageII$estimate[3]
  result.estimate.tilde[4] = result.stageII$estimate[4]
  result.convergence[2] = ifelse(result.stageII$code==1, 1, 0)
  result.estimate = exp(result.estimate.tilde)
  #####End Stage II

  #####Begin Jackknife
  if(jackknife == TRUE){#do not compute SE for subset

    result.cov.tilde = matrix(0, nrow = 6, ncol = 6)
    for( i in c(1:N)){
      no = which(subject == unique(subject)[i])
      temp = Theta_k = NA
      temp = try(ClaytonClayton.Weibull.MLE(subject = subject[-no], t.event = t.event[-no], event = event[-no],
                                      t.death = t.death[-i], death = death[-i], stageI = stageI, jackknife = FALSE, digit = 10))
      if ("try-error"%in%class(temp)){
        jackknife.error = jackknife.error + 1
        next;
      }else if(temp$theta[1]>15|temp$alpha[1]>15|temp$theta[1]==0|temp$alpha[1]==0){
        jackknife.error = jackknife.error + 1
        next;
      }else{
        Theta_k = (log(c(temp$scale1[1], temp$shape1[1], temp$theta[1], temp$alpha[1], temp$scale2[1], temp$shape2[1])) - result.estimate.tilde)
        result.cov.tilde = result.cov.tilde + Theta_k %*% t(Theta_k)
      }
    }
    result.se = diag(result.cov.tilde)^(1/2)*result.estimate
    result.lowerbound = result.estimate * exp( - 1.96 * diag(result.cov.tilde)^(1/2) )
    result.upperbound = result.estimate * exp( 1.96 * diag(result.cov.tilde)^(1/2) )
    colnames(result.cov.tilde) = c("scale1.tilde", "shape1.tilde", "theta.tilde", "alpha.tilde", "scale2.tilde", "shape2.tilde")
    rownames(result.cov.tilde) = c("scale1.tilde", "shape1.tilde", "theta.tilde", "alpha.tilde", "scale2.tilde", "shape2.tilde")
  }
  #####End Jackknife

  #####Begin Figure
  if(jackknife == TRUE & plot == TRUE){
    scale1 = result.estimate[1]
    shape1 = result.estimate[2]
    V_r = result.cov.tilde[1,1]
    V_shape1 = result.cov.tilde[2,2]
    Cov_r_shape1 = result.cov.tilde[1,2]
    t.event.range = seq(0.9*min(t.event), ceiling(max(t.event)), length.out = 200)
    r.hazard = scale1 * shape1 * t.event.range^(shape1 - 1)
    r.hazard.se = sqrt( ( scale1 * shape1 * t.event.range^(shape1-1) )^2 *
                          ( V_r + (1+log(t.event.range)*shape1)*Cov_r_shape1 + (1+log(t.event.range)*shape1)^2*V_shape1 ) )
    r.hazard.ub = r.hazard + 1.96 * r.hazard.se
    r.hazard.lb = r.hazard - 1.96 * r.hazard.se

    scale2 = result.estimate[5]
    shape2 = result.estimate[6]
    V_lambda = result.cov.tilde[5,5]
    V_shape2 = result.cov.tilde[6,6]
    Cov_lambda_shape2 = result.cov.tilde[5,6]
    t.death.range = seq(0.9*min(t.death), ceiling(max(t.death)), length.out = 200)
    lambda.hazard = scale2 * shape2 * t.death.range^(shape2 - 1)
    lambda.hazard.se = sqrt( ( scale2 * shape2 * t.death.range^(shape2-1) )^2 *
                               ( V_lambda + (1+log(t.death.range)*shape2)*Cov_lambda_shape2 + (1+log(t.death.range)*shape2)^2*V_shape2 ) )
    lambda.hazard.ub = lambda.hazard + 1.96 * lambda.hazard.se
    lambda.hazard.lb = lambda.hazard - 1.96 * lambda.hazard.se

    plot.hazards = function(){
      oldpar <-  par(mfrow = c(2,1))
      on.exit(par(oldpar))
      plot(0, 0, type = "n", ylim = c(0, max(r.hazard.ub)), xlim = c(0.01, ceiling(max(t.event.range))),
           main = "Hazard for event", xlab = "t", ylab = "r(t)", col = "blue", lwd = 2, axes=FALSE, cex = 2)
      axis(1, at = c(0, signif(ceiling(max(t.event.range))/4 * c(1:3), digits = 2), ceiling(max(t.event.range))), col="black", las=1)
      axis(2, at = c(0, signif(max(r.hazard.ub)/3 * c(1:2), digits = 2), signif(max(r.hazard.ub),digits = 2)), col="black", las=1)
      polygon(c(t.event.range,rev(t.event.range)),
              c(r.hazard.lb,rev(r.hazard.ub)), col = "lightblue",border=NA)
      lines(t.event.range, r.hazard, col = "blue", lwd = 2)

      plot(0, 0, type = "n", ylim = c(0, max(lambda.hazard.ub)), xlim = c(0.01, ceiling(max(t.death.range))),
           main = "Hazard for death", xlab = "t", ylab = expression(paste(lambda,"(t)")), col = "blue", lwd = 2, axes=FALSE, cex = 2)
      axis(1, at = c(0, signif(ceiling(max(t.death.range))/4 * c(1:3), digits = 2), ceiling(max(t.death.range))), col="black", las=1)
      axis(2, at = c(0, signif(max(lambda.hazard.ub)/3 * c(1:2), digits = 2), signif(max(lambda.hazard.ub),digits = 2)), col="black", las=1)
      polygon(c(t.death.range,rev(t.death.range)),
              c(lambda.hazard.lb,rev(lambda.hazard.ub)), col = "lightpink",border=NA)
      lines(t.death.range, lambda.hazard, col = "red", lwd = 2)
    }

    plot.hazards()

  }
  #####End Figure

  #####Begin GOF
  SD.KM = Surv(time = t.death, event = death)
  SDn = summary(survfit(SD.KM~1))
  SD = exp(-result.estimate[5]*SDn$time^result.estimate[6])
  SR.KM = Surv(time = t.event, event = event)
  SRn = summary(survfit(SR.KM~1))
  SR = exp(-result.estimate[1]*SRn$time^result.estimate[2])
  KS.D = max(abs(SDn$surv - SD))
  CM.D = sum((SDn$surv - SD)^2)
  KS.R = max(abs(SRn$surv - SR))
  CM.R = sum((SRn$surv - SR)^2)
  KS.D_bootstrap = CM.D_bootstrap = KS.R_bootstrap =  CM.R_bootstrap = rep(NA, rep.GOF)
  if( GOF == TRUE){
    data.sp = NULL
    for(b in c(1:rep.GOF)){
      set.seed(b)
      data_b = ClaytonClayton.Weibull.data(N = N, scale1 = result.estimate[1], shape1 = result.estimate[2], theta = result.estimate[3],
                                         scale2 = result.estimate[5], shape2 = result.estimate[6], alpha = result.estimate[4], b = max(t.death[death==0]), l = 200)
      result.bootstrap.temp = try(ClaytonClayton.Weibull.MLE(subject = data_b$Subject, t.event = data_b$T_ij, event = data_b$delta_ij,
                                                       t.death = data_b$T_i_star, death = data_b$delta_i_star, stageI = stageI,
                                                       jackknife = FALSE, plot = FALSE, GOF = FALSE))
      if("try-error"%in%result.bootstrap.temp){
        next;
      }else{
        KS.D_bootstrap[b] = result.bootstrap.temp$KS[2]
        CM.D_bootstrap[b] = result.bootstrap.temp$CM[2]
        KS.R_bootstrap[b] = result.bootstrap.temp$KS[1]
        CM.R_bootstrap[b] = result.bootstrap.temp$CM[1]
      }
    }
    KS.D_p = mean(KS.D_bootstrap > KS.D, na.rm = TRUE)
    CM.D_p = mean(CM.D_bootstrap > CM.D, na.rm = TRUE)
    KS.R_p = mean(KS.R_bootstrap > KS.R, na.rm = TRUE)
    CM.R_p = mean(CM.R_bootstrap > CM.R, na.rm = TRUE)

    if(GOF.plot == TRUE){
      plot.diagnostic = function(){
        oldpar <-  par(mfrow = c(1,2))
        on.exit(par(oldpar))
        plot(SD~SDn$surv, xlim = c(0,1), ylim = c(0,1), xlab = "S_parametric", ylab = "S_KM", main = "death")
        plot(SR~SRn$surv, xlim = c(0,1), ylim = c(0,1), xlab = "S_parametric", ylab = "S_KM", main = "event")
      }

      plot.diagnostic()
    }

  }
  #####End GOF
  #####Likelihood value
  logL.value = function(par){

    scale1 = par[1]
    shape1 = par[2]
    theta = par[3]
    alpha = par[4]
    scale2 = par[5]
    shape2 = par[6]

    #corresponding D function
    A_theta = function(s,t){ exp(theta*s) + exp(theta*t) - 1 }
    D_theta_01 = function(s,t){ A_theta(s,t)^(-1/theta-1)*exp(theta*t) }
    D_theta_11 = function(s,t){ (theta+1)*A_theta(s,t)^(-1/theta-2)*exp(theta*s)*exp(theta*t) }
    A.tilde_alpha = function(s,t){ exp(alpha*s) + exp(alpha*t) - 1 }
    D.tilde_alpha = function(s,t){ A.tilde_alpha(s,t)^(-1/alpha) }
    D.tilde_alpha_01 = function(s,t){ A.tilde_alpha(s,t)^(-1/alpha-1)*exp(alpha*t) }
    D.tilde_alpha_10 = function(s,t){ A.tilde_alpha(s,t)^(-1/alpha-1)*exp(alpha*s) }
    D.tilde_alpha_11 = function(s,t){ (alpha+1)*A.tilde_alpha(s,t)^(-1/alpha-2)*exp(alpha*s)*exp(alpha*t) }

    l = 0

    for(i in c(1:N)){

      l_i = 0

      no = which(subject == unique(subject)[i])
      n_i = length(no)
      T_ij = t.event[no]
      delta_ij = event[no]
      T_i_star = t.death[i]
      delta_i_star = death[i]
      #marginal Weibull for Xij
      r.T_ij = scale1*shape1*T_ij^(shape1-1)
      R.T_ij = scale1*T_ij^shape1
      #marginal Weibull for Dij
      lambda.T_i_star = scale2*shape2*T_i_star^(shape2-1)
      Lambda.T_i_star = scale2*T_i_star^shape2
      #likelihood for ni=1
      l_i = delta_i_star*log(lambda.T_i_star) + delta_ij[1]*log(r.T_ij[1]) +
        delta_i_star*delta_ij[1]*log( D.tilde_alpha_11(R.T_ij[1], Lambda.T_i_star) ) +
        delta_i_star*(1-delta_ij[1])*log( D.tilde_alpha_01(R.T_ij[1], Lambda.T_i_star) ) +
        (1 - delta_i_star)*log( D.tilde_alpha_10(R.T_ij[1], Lambda.T_i_star) )+
        (1 - delta_i_star)*(1-delta_ij[1])*log( D.tilde_alpha(R.T_ij[1], Lambda.T_i_star) )
      #likelihood for ni>=2
      if(n_i >= 2){
        l_i = l_i + sum(
          R.T_ij[1:(n_i-1)] + delta_ij[2:n_i]*log(r.T_ij[2:n_i]) +
            delta_ij[2:n_i]*log( D_theta_11(R.T_ij[2:n_i],R.T_ij[1:(n_i-1)]) ) +
            (1 - delta_ij[2:n_i])*log( D_theta_01(R.T_ij[2:n_i],R.T_ij[1:(n_i-1)]) )
        )
      }

      l = l + l_i
    }

    return(l)

  }

  #####
  return(
    list(
      Sample_size = N, Case = case,
      scale1 = round(c(Estimate = result.estimate[1], SE = result.se[1], Lower = result.lowerbound[1], Upper = result.upperbound[1]),digit),
      shape1 = round(c(Estimate = result.estimate[2], SE = result.se[2], Lower = result.lowerbound[2], Upper = result.upperbound[2]),digit),
      scale2 = round(c(Estimate = result.estimate[5], SE = result.se[5], Lower = result.lowerbound[5], Upper = result.upperbound[5]),digit),
      shape2 = round(c(Estimate = result.estimate[6], SE = result.se[6], Lower = result.lowerbound[6], Upper = result.upperbound[6]),digit),
      theta = round(c(Estimate = result.estimate[3], SE = result.se[3], Lower = result.lowerbound[3], Upper = result.upperbound[3]),digit),
      alpha = round(c(Estimate = result.estimate[4], SE = result.se[4], Lower = result.lowerbound[4], Upper = result.upperbound[4]),digit),
      COV = result.cov.tilde,
      KS = round(c(Event = KS.R, Death = KS.D),2), p.KS = round(c(Event = KS.R_p, Death = KS.D_p),3),
      CM = round(c(Event = CM.R, Death = CM.D),2), p.CM = round(c(Event = CM.R_p, Death = CM.D_p),3),
      Convergence = result.convergence,
      Jackknife_error = jackknife.error,
      Log_likelihood = logL.value(result.estimate)
    )
  )
}
