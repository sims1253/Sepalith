pval = function(actual.scores, null.scores, method = "exponential"){

  if(length(actual.scores) == 0 || dim(null.scores)[1] == 0){
    stop("actual.scores or null.scores is empty")
  }

  if(!method %in% c("exponential", "gamma", "non_parametric")) {
    stop("Invalid method. Choose from 'exponential', 'gamma', 'non_parametric'")
  }

  pvalue = length(actual.scores)
  for(j in 1:length(actual.scores)){

    if(method == "exponential"){

      # P-value calculation: exponential
      rate = MASS::fitdistr(null.scores[j,], "exponential")$estimate
      pvalue[j] = stats::pexp(actual.scores[j], rate, lower.tail = FALSE)


    }else if(method == "gamma"){

        # P-value calculation: exponential
        param = MASS::fitdistr(null.scores[j,], "gamma")
        shape = param$estimate["shape"]
        rate = param$estimate["rate"]
        pvalue[j] = stats::pgamma(actual.scores[j], shape, rate, lower.tail = FALSE)

    }else if(method == "non_parametric"){

      # P-value calculation: non_parametric
      pvalue[j] = sum(null.scores[j,] >= actual.scores[j]) / ncol(null.scores)
    }
  }
  return(pvalue)
}