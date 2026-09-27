setClass("relout", representation(est="numeric", cov="matrix", pder="numeric", type="character"))
##reqs
adjgpcmmmirt <- function(input){
  ncats <- extract.mirt(input, "K")
  nitem <- length(ncats)
  pdermat <- matrix(0, ncol = sum(ncats), nrow = sum(ncats))
  k <- 1
  for(i in 1:nitem){
    ttpar <- extract.item(input, i)
    tempmat <- matrix(0, nrow = ncats[i], ncol = ncats[i])
    tempmat[1, 1] <- ttpar@par[2+ncats[i]+1] / ttpar@par[1]^2
    for(j in 2:(ncats[i]-1)) tempmat[1, j] <- (ttpar@par[j+ncats[i]+2] - ttpar@par[j+ncats[i]+1]) / ttpar@par[1]^2
    tempmat[1, ncats[i]] <- 1
    
    for(j in 2:(ncats[i]-1)){
      tempmat[j,j-1] <- -1 / ttpar@par[1]
      tempmat[j,j] <- 1 / ttpar@par[1]
    }
    tempmat[ncats[i], ncats[i]-1] <- -1 / ttpar@par[1]
    
    #print(tempmat)
    pdermat[k:(k + ncats[i]-1), k:(k + ncats[i]-1)] <- tempmat
    k <- k + ncats[i]
  }
  pdermat
}

#Function to go to the regular IRT parametrization from the parametrization used in mixirtiw().
adjgpcmmixirt <- function(input){
  ncats <- apply(input$data, 2, function(x) length(unique(x[!is.na(x)])))
  nitem <- length(ncats)
  nparitem <- numeric(nitem)
  for(i in 1:nitem){
	if(input$itemtype[i]==1) nparitem[i] <- 1
	if(input$itemtype[i]==2) nparitem[i] <- 2
	if(input$itemtype[i]==3) nparitem[i] <- 3
	if(input$itemtype[i]==4) nparitem[i] <- ncats[i]
	if(input$itemtype[i]==5) nparitem[i] <- ncats[i] - 1 
  }
  npar <- sum(nparitem)
  pdermat <- matrix(0, ncol = sum(nparitem), nrow = sum(nparitem))
  k <- 1  
  for(i in 1:nitem){
    if(input$itemtype[i]==1){
		#input: (b*)
		#output: (b)
		tempmat <- matrix(0, nrow = nparitem[i], ncol = nparitem[i])
		bpar <- input$par[k]
		tempmat[1,1] <- -1
		pdermat[k:(k + nparitem[i] - 1), k:(k + nparitem[i] - 1)] <- tempmat
		k <- k + nparitem[i]
	}
    if(input$itemtype[i]==2){
		#input: (a, b*)
		#output: (b, a)
		tempmat <- matrix(0, nrow = nparitem[i], ncol = nparitem[i])
		apar <- input$par[k]
		bpar <- input$par[k+1]
		tempmat[1,1] <- bpar / apar^2
		tempmat[1,2] <- 1
		tempmat[2,1] <- - 1 / bpar
		pdermat[k:(k + nparitem[i] - 1), k:(k + nparitem[i] - 1)] <- tempmat
		k <- k + nparitem[i]
	}
  	if(input$itemtype[i]==3){
		#input: (a, b*, c*)
		#output: (c, b, a)
		tempmat <- matrix(0, nrow = nparitem[i], ncol = nparitem[i])
		apar <- input$par[k]
		bpar <- input$par[k+1]
		cpar <- input$par[k+2]
		tempmat[1,2] <- bpar / apar^2
		tempmat[1,3] <- 1
		tempmat[2,2] <- - 1 / apar
		tempmat[3,1] <- exp(cpar) / (1 + exp(cpar))^2
		pdermat[k:(k + nparitem[i] - 1), k:(k + nparitem[i] - 1)] <- tempmat
		k <- k + nparitem[i]
	}
	if(input$itemtype[i]==4){
		#input: (a, b2*, ..., bm*)
		#output: (b2, ..., bm, a)
		ttpar <- input$par[k:(k + nparitem[i] - 1)]
		tempmat <- matrix(0, nrow = nparitem[i], ncol = nparitem[i])
		for(j in 1:(nparitem[i]-1)) tempmat[1, j] <- ttpar[j+1] / ttpar[1]^2
		tempmat[1, nparitem[i]] <- 1
		for(j in 2:(nparitem[i])){
		  tempmat[j,j-1] <- - 1 / ttpar[1]
		}
		pdermat[k:(k + nparitem[i] - 1), k:(k + nparitem[i] - 1)] <- tempmat
		k <- k + nparitem[i]
	}
	if(input$itemtype[i]==5){
		#input: (b2*, ..., bm*)
		#output: (b2, ..., bm)
		tempmat <- matrix(0, nrow = nparitem[i], ncol = nparitem[i])
		for(j in 1:nparitem[i]){
		  tempmat[j,j] <- - 1
		}
		pdermat[k:(k + nparitem[i] - 1), k:(k + nparitem[i] - 1)] <- tempmat
		k <- k + nparitem[i]
	}
  }
  pdermat
}

adjgrmmirt <- function(input){
  ncats <- extract.mirt(input, "K")
  nitem <- length(ncats)
  
  pdermat <- matrix(0, ncol = sum(ncats), nrow = sum(ncats))
  
  k <- 1
  for(i in 1:nitem){
    ttpar <- extract.item(input, i)
    tempmat <- matrix(0, nrow = ncats[i], ncol = ncats[i])
    for(j in 2:ncats[i]) tempmat[j,j] <- -1 / ttpar@par[1]
    tempmat[1,] <- c(1, ttpar@par[-1] / ttpar@par[1]^2)
    pdermat[k:(k + ncats[i]-1), k:(k + ncats[i]-1)] <- tempmat
    k <- k + ncats[i]
  }
  pdermat
}

cmnom <- function(cats, probs, qpoints){
  #cats - no of categories for each item (we assume 0, 1, 2, ... scoring, i.e. cats[i] <- 3 means 0, 1, 2 scoring)
  #probs a list of length(cats) with probabilities to answer in each category on each item for each ability level, i.e. each list entry is a matrix with dim(probs[[i]]) = c(cats[i], length(qpoints))
  #qpoints is a vector of quadrature points
  JX <- length(cats)
  xK <- sum(cats) - JX
  nsprobs <- matrix(0, nrow = xK + 1, ncol = length(qpoints$x))
  nsprobs[1:(cats[1]),] <- probs[[1]]
  
  for(i in 2:JX){
    maxsc <- sum(cats[1:i]) - length(cats[1:i])
    sprobs <- nsprobs
    nsprobs <- matrix(0, nrow = xK + 1, ncol = length(qpoints$x))
    for(j in 1:(maxsc - cats[i] + 2))  nsprobs[j:(cats[i] + j - 1),] <- t(sprobs[j,] * t(probs[[i]])) + nsprobs[j:(cats[i] + j - 1),]
  }
  return(t(t(nsprobs) * (qpoints$w)))
}


ccmnom <- function(cats, probs, qpoints){
  #cats - no of categories for each item (we assume 0, 1, 2, ... scoring, i.e. cats[i] <- 3 means 0, 1, 2 scoring)
  #probs a list of length(cats) with probabilities to answer in each category on each item for each ability level, i.e. each list entry is a matrix with dim(probs[[i]]) = c(cats[i], length(qpoints))
  #qpoints is a vector of quadrature points
  JX <- length(cats)
  xK <- sum(cats) - JX
  nsprobs <- matrix(0, nrow = xK + 1, ncol = length(qpoints))
  nsprobs[1:(cats[1]),] <- probs[[1]]
  
  if(JX>1){
    for(i in 2:JX){
      maxsc <- sum(cats[1:i]) - length(cats[1:i])
      sprobs <- nsprobs
      nsprobs <- matrix(0, nrow = xK + 1, ncol = length(qpoints))
      for(j in 1:(maxsc - cats[i] + 2))  nsprobs[j:(cats[i] + j - 1),] <- t(sprobs[j,] * t(probs[[i]])) + nsprobs[j:(cats[i] + j - 1),]
    }
  }
  nsprobs
}


polyprob <- function(a, b, cats, model, qpoints){
  JX <- length(cats)
  kX <- sum(cats) - JX
  
  probs <- vector("list", JX)
  if(model=="GPCM"){
    for(i in 1:JX){
      out <- matrix(0, nrow = cats[i], ncol = length(qpoints))
      denom <- 0
      for(j in 1:(cats[i] - 1)){
        temp <- 0
        for(l in 1:j) temp <- a[i] * (qpoints - b[[i]][l]) + temp 
        denom <- exp(temp) + denom
      }
      out[1,] <- 1 / (1 + denom)
      for(j in 1:(cats[i] - 1)){
        numer <- exp(j * a[i] * qpoints - a[i] * sum(b[[i]][1:j]))
        out[j + 1,] <- numer / (1 + denom)
      }
      probs[[i]] <- out
    }
  }
  
  if(model=="GRM"){
    for(i in 1:JX){
      out <- matrix(0, nrow = cats[i], ncol = length(qpoints))
      out[1,] <- 1 - 1 / (1 + exp(-a[i] * (qpoints - b[[i]][1])))
      out[cats[i],] <- 1 / (1 + exp(-a[i] * (qpoints - b[[i]][cats[i]-1])))
      if(cats[i]>2){
        for(j in 2:(cats[i] - 1)) out[j,] <- 1 / (1 + exp(-a[i] * (qpoints - b[[i]][j-1]))) - 1 / (1 + exp(-a[i] * (qpoints - b[[i]][j])))
      }
      probs[[i]] <- out
    }
  }
  return(probs)
}


polypderpl <- function(a, b, cats, probs, model, qpoints, ABqpoints){
  JX <- length(cats)
  kX <- sum(cats) - JX
  qp <- qpoints$x
  ABqp <- ABqpoints
  
  cprobs <- vector("list", JX)
  for(i in 1:JX){
    output <- vector("list", cats[i])
    input <- probs
    for(j in 1:cats[i]){
      input[[i]] <- matrix(0, nrow = cats[i], ncol = length(qp))
      input[[i]][j,] <- rep(1, length(qp))
      output[[j]] <- ccmnom(cats, input, ABqp)
    }
    cprobs[[i]] <- output
  }
  pPpalpha <- vector("list", JX)
  
  if(model=="GRM"){
    for(j in 1:JX){
      arr <- vector("list", cats[j])
      equada <- exp(-a[j] * ABqp)
      
      arr[[1]] <- matrix(0, nrow = cats[j], ncol = length(qp))
      arr[[1]][1,] <- -exp(-a[j] * (ABqp - b[[j]][1])) * (ABqp - b[[j]][1]) / (1 + exp(-a[j] * (ABqp - b[[j]][1])))^2
      arr[[1]][2,] <- exp(-a[j] * (ABqp - b[[j]][1])) * a[j] / (1 + exp(-a[j] * (ABqp - b[[j]][1])))^2
      
      for(l in 2:(cats[j]-1)){
        arr[[l]] <- matrix(0, nrow = cats[j], ncol = length(qp)) 
        for(m in 2:(cats[j]-1)){
          if(m == l){
            arr[[l]][m,] <- -exp(-a[j] * (ABqp - b[[j]][m-1])) * a[j] / (1 + exp(-a[j] * (ABqp - b[[j]][m-1])))^2
            arr[[l]][m+1,] <- exp(-a[j] * (ABqp - b[[j]][m])) * a[j] / (1 + exp(-a[j] * (ABqp - b[[j]][m])))^2
          }
        }
        
        arr[[l]][1,] <- exp(-a[j] * (ABqp - b[[j]][l-1])) * (ABqp - b[[j]][l-1]) / (1 + exp(-a[j] * (ABqp - b[[j]][l-1])))^2 - exp(-a[j] * (ABqp - b[[j]][l])) * (ABqp - b[[j]][l]) / (1 + exp(-a[j] * (ABqp - b[[j]][l])))^2
      }
      arr[[cats[j]]] <- matrix(0, nrow = cats[j], ncol = length(qp))
      arr[[cats[j]]][1,] <- exp(-a[j] * (ABqp - b[[j]][cats[j]-1])) * (ABqp - b[[j]][cats[j]-1]) / (1 + exp(-a[j] * (ABqp - b[[j]][cats[j]-1])))^2
      arr[[cats[j]]][cats[j],] <- -exp(-a[j] * (ABqp - b[[j]][cats[j]-1])) * a[j] / (1 + exp(-a[j] * (ABqp - b[[j]][cats[j]-1])))^2
      pPpalpha[[j]] <- arr
    }
    
    prpalpha <- vector("list", kX + 1)
    for(i in 1:(kX + 1)){
      prpalphal <- vector("list", JX)
      for(j in 1:JX){
        tempmat <- matrix(0, nrow = cats[j], ncol = length(qp))
        for(k in 1:cats[j]) tempmat <- t((cprobs[[j]][[k]][i,]) * t(pPpalpha[[j]][[k]]))  + tempmat
        prpalphal[[j]] <- rowSums(t(qpoints$w * t(tempmat)))
      }
      prpalpha[[i]] <- prpalphal
    }
    return(prpalpha)
  }
  
  
  if(model=="GPCM"){
    for(j in 1:JX){
      arr <- vector("list", cats[j])
      #1st: category answer of item j, 2nd: each parameter for item j, 3rd: each quadrature point
      
      ek <- matrix(0, nrow = cats[j], ncol = length(qp))
      ek[1,] <- 1
      
      for(k in 2:cats[j]){
        ek[k,] <- exp(a[j] * (ABqp - b[[j]][k-1]))
      }
      
      s1 <- numeric(length(qp))
      for(k in 2:cats[j]){
        s1 <- apply(ek[1:k,], 2, prod) + s1
      }
      
      s3 <- numeric(length(qp))
      for(k in 2:cats[j]){
        s3 <- apply(ek[1:k,], 2, prod) * ((k - 1) * ABqp - sum(b[[j]][1:(k-1)])) + s3
      }
      
      arr[[1]] <- matrix(0, nrow = cats[j], ncol = length(qp))
      
      for(l in 2:cats[j]){
        temp <- numeric(length(qp))
        for(m in l:(cats[j])){
          temp <- exp(a[j] * ((m - 1) * ABqp - sum(b[[j]][1:(m - 1)]))) +  temp
        }
        s2 <- temp
        arr[[1]][(l - 1),] <- a[j] * s2 / (1 + s1)^2
      }
      arr[[1]][cats[j],] <- -s3 / (1 + s1)^2
      
      
      for(k in 2:cats[j]){
        arr[[k]] <- matrix(0, nrow = cats[j], ncol = length(qp))
        s4 <- apply(ek[1:k,], 2, prod)
        s5 <- (k - 1) * ABqp - sum(b[[j]][1:(k - 1)])
        
        for(l in 2:cats[j]){
          temp <- numeric(length(qp))
          for(m in l:(cats[j])){
            temp <- exp(a[j] * ((m - 1) * ABqp - sum(b[[j]][1:(m - 1)]))) +  temp
          }
          s2 <- temp
          if(l <= k){
            arr[[k]][l-1,] <- -a[j] * (s4 * (1 + s1) - s4 * s2) / (1 + s1)^2
          } else{
            arr[[k]][l-1,] <- a[j] * s2 * s4 / (1 + s1)^2
          }
          
        }
        arr[[k]][cats[j],] <- (s4 * s5 * (1 + s1) - s4 * s3) / (1 + s1)^2 
      }
      pPpalpha[[j]] <- arr
    }
    prpalpha <- vector("list", kX + 1)
    for(i in 1:(kX + 1)){
      prpalphal <- vector("list", JX)
      for(j in 1:JX){
        tempmat <- matrix(0, nrow = cats[j], ncol = length(qp))
        for(k in 1:cats[j]) tempmat <- t((cprobs[[j]][[k]][i,]) * t(pPpalpha[[j]][[k]]))  + tempmat
        prpalphal[[j]] <- rowSums(t(qpoints$w * t(tempmat)))
      }
      prpalpha[[i]] <- prpalphal
    }
    return(prpalpha)
  }
}


polypder <- function(a, b, cats, probs, model, qpoints){
  #NOTE: changed back to previous order, i.e. the b-parameters come first for each item, then the a-par
  JX <- length(cats)
  kX <- sum(cats) - JX
  qp <- qpoints
  #no GRM yet
  
  pPpalpha <- vector("list", JX)
  for(j in 1:JX){
    arr <- vector("list", cats[j])
    #1st: category answer of item j, 2nd: each parameter for item j, 3rd: each quadrature point
    
    ek <- matrix(0, nrow = cats[j], ncol = length(qp))
    ek[1,] <- 1
    
    for(k in 2:cats[j]){
      ek[k,] <- exp(a[j] * (qp - b[[j]][k-1]))
    }
    
    s1 <- numeric(length(qp))
    for(k in 2:cats[j]){
      s1 <- apply(ek[1:k,], 2, prod) + s1
    }
    
    s3 <- numeric(length(qp))
    for(k in 2:cats[j]){
      s3 <- apply(ek[1:k,], 2, prod) * ((k - 1) * qp - sum(b[[j]][1:(k-1)])) + s3
    }
    
    arr[[1]] <- matrix(0, nrow = cats[j], ncol = length(qp))
    #special case category response 1
    
    for(l in 2:cats[j]){
      temp <- numeric(length(qp))
      for(m in l:(cats[j])){
        temp <- exp(a[j] * ((m - 1) * qp - sum(b[[j]][1:(m - 1)]))) +  temp
      }
      s2 <- temp
      arr[[1]][l-1 ,] <- a[j] * s2 / (1 + s1)^2
    }
    arr[[1]][cats[j],] <- -s3 / (1 + s1)^2
    
    
    for(k in 2:cats[j]){
      arr[[k]] <- matrix(0, nrow = cats[j], ncol = length(qp))
      s4 <- apply(ek[1:k,], 2, prod)
      s5 <- (k - 1) * qp - sum(b[[j]][1:(k - 1)])
      
      for(l in 2:cats[j]){
        temp <- numeric(length(qp))
        for(m in l:(cats[j])){
          temp <- exp(a[j] * ((m - 1) * qp - sum(b[[j]][1:(m - 1)]))) +  temp
        }
        s2 <- temp
        if(l <= k){
          arr[[k]][l-1,] <- -a[j] * (s4 * (1 + s1) - s4 * s2) / (1 + s1)^2
        } else{
          arr[[k]][l-1,] <- a[j] * s2 * s4 / (1 + s1)^2
        }
        
      }
      arr[[k]][cats[j],] <- (s4 * s5 * (1 + s1) - s4 * s3) / (1 + s1)^2 
    }
    pPpalpha[[j]] <- arr
  }
  return(pPpalpha)
}


parmirt <- function(P, Q, model, catsX, catsY, catsA, SE = T){
  if(catsX[1]==0) JX <- 0 else JX <- length(catsX)
  if(catsY[1]==0) JY <- 0 else JY <- length(catsX)
  if(catsA[1]==0) JA <- 0 else JA <- length(catsA)
  
  kX <- sum(catsX) - JX
  kY <- sum(catsY) - JY
  kA <- sum(catsA) - JA
  
  if(is.null(Q)){
    aP <- numeric(length(catsX))
    bP <- vector("list", length(catsX))
    if(class(P) == "SingleGroupClass"){
      if(model %in% c("GPCM")){
        for(i in 1:JX){
          ttpar <- extract.item(P, i)
          aP[i] <- ttpar@par[1]
          bP[[i]] <- -(tail(ttpar@par, catsX[i]-1) - c(0, tail(ttpar@par, catsX[i]-1)[-(catsX[i]-1)])) / aP[i]
        }
        if(SE){
          vcovP <- extract.mirt(P, "vcov")
          vcovP <- t(adjgpcmmmirt(P)) %*% vcovP %*% adjgpcmmmirt(P)
          vcovP <- vcovP[1:(sum(catsX)), 1:(sum(catsX))]
        }
      }
      if(model %in% c("GRM")){
        for(i in 1:JX){
          ttpar <- extract.item(P, i)
          aP[i] <- ttpar@par[1]
          bP[[i]] <- -ttpar@par[-1] / aP[i]
        }
        if(SE){
          vcovP <- extract.mirt(P, "vcov")
          vcovP <- t(adjgrmmirt(P)) %*% vcovP %*% adjgrmmirt(P)
          vcovP <- vcovP[1:(sum(catsX)), 1:(sum(catsX))]
        }
      }
      if(SE) outP <- list(aP, bP, vcovP) else outP <- list(aP, bP) 
      return(outP)
    }
  }
  
  if(catsA[1] == 0){
    aP <- numeric(length(catsX))
    aQ <- numeric(length(catsY))
    bP <- vector("list", length(catsX))
    bQ <- vector("list", length(catsY))
    if(class(P) == "SingleGroupClass" && class(Q) == "SingleGroupClass"){
      if(model %in% c("GPCM")){
        for(i in 1:JX){
          ttpar <- extract.item(P, i)
          aP[i] <- ttpar@par[1]
          bP[[i]] <- -(tail(ttpar@par, catsX[i]-1) - c(0, tail(ttpar@par, catsX[i]-1)[-(catsX[i]-1)])) / aP[i]
        }
        if(SE) vcovP <- extract.mirt(P, "vcov")
        for(i in 1:JY){
          ttpar <- extract.item(Q, i)
          aQ[i] <- ttpar@par[1]
          bQ[[i]] <- -(tail(ttpar@par, catsY[i]-1) - c(0, tail(ttpar@par, catsY[i]-1)[-(catsY[i]-1)])) / aQ[i]
        }
        if(SE) vcovQ <- extract.mirt(Q, "vcov")
      }
      
      if(model %in% c("GRM")){
        for(i in 1:JX){
          ttpar <- extract.item(P, i)
          aP[i] <- ttpar@par[1]
          bP[[i]] <- -ttpar@par[-1] / aP[i]
        }
        if(SE) vcovP <- extract.mirt(P, "vcov")
        for(i in 1:JY){
          ttpar <- extract.item(Q, i)
          aQ[i] <- ttpar@par[1]
          bQ[[i]] <- -ttpar@par[-1] / aQ[i]
        }
        if(SE) vcovQ <- extract.mirt(Q, "vcov")
      }
      vcovP <- t(adjgpcmmmirt(P)) %*% vcovP %*% adjgpcmmmirt(P)
      vcovQ <- t(adjgpcmmmirt(Q)) %*% vcovQ %*% adjgpcmmmirt(Q)
      vcovP <- vcovP[1:(sum(catsX)), 1:(sum(catsX))]
      vcovQ <- vcovQ[1:(sum(catsY)), 1:(sum(catsY))]
    }
  } else{
    aP <- aQ <- numeric(length(catsA))
    bP <- bQ <- vector("list", length(catsA))
    if(class(P) == "SingleGroupClass" && class(Q) == "SingleGroupClass"){
      if(model %in% c("GPCM")){
        for(i in 1:JA){
          ttpar <- extract.item(P, i+JX)
          aP[i] <- ttpar@par[1]
          bP[[i]] <- -(tail(ttpar@par, catsA[i]-1) - c(0, tail(ttpar@par, catsA[i]-1)[-(catsA[i]-1)])) / aP[i]
        }
        if(SE) vcovP <- extract.mirt(P, "vcov")
        for(i in 1:JA){
          ttpar <- extract.item(Q, i+JY)
          aQ[i] <- ttpar@par[1]
          bQ[[i]] <- -(tail(ttpar@par, catsA[i]-1) - c(0, tail(ttpar@par, catsA[i]-1)[-(catsA[i]-1)])) / aQ[i]
        }
        if(SE) vcovQ <- extract.mirt(Q, "vcov")
      }
      
      if(model %in% c("GRM")){
        for(i in 1:JA){
          ttpar <- extract.item(P, i+JX)
          aP[i] <- ttpar@par[1]
          bP[[i]] <- -ttpar@par[-1] / aP[i]
        }
        if(SE) vcovP <- extract.mirt(P, "vcov")
        for(i in 1:JA){
          ttpar <- extract.item(Q, i+JY)
          aQ[i] <- ttpar@par[1]
          bQ[[i]] <- -ttpar@par[-1] / aQ[i]
        }
        if(SE) vcovQ <- extract.mirt(Q, "vcov")
      }
      vcovP <- t(adjgpcmmmirt(P)) %*% vcovP %*% adjgpcmmmirt(P)
      vcovQ <- t(adjgpcmmmirt(Q)) %*% vcovQ %*% adjgpcmmmirt(Q)
      if(catsX[1] != 0) vcovP <- vcovP[-(1:(sum(catsX))), -(1:(sum(catsX)))]
      if(catsY[1] != 0) vcovQ <- vcovQ[-(1:(sum(catsY))), -(1:(sum(catsY)))]
    }
  }
  
  outP <- list(aP, bP, vcovP)
  outQ <- list(aQ, bQ, vcovQ)
  
  return(list(outP, outQ))
}

adjltm <- function(mat, pars, design, model){
  if(design=="CE" || design=="PSE"){
    if(model=="2pl"){
      pderltmP <- matrix(0, nrow=2*(length(pars$ax)+length(pars$aa)), ncol=2*(length(pars$ax)+length(pars$aa)))
      
      #print(bx)
      #print(aaP)
      
      for(i in 1:(length(pars$ax))){
        pderltmP[i,i] <- -1/pars$ax[i]
      }
      for(i in (length(pars$ax)+1):(length(pars$ax)+length(pars$aa))){
        pderltmP[i,i] <- -1/pars$aa[i-length(pars$ax)]
      }
      
      for(i in (length(pars$ax)+length(pars$aa)+1):(2*length(pars$ax)+length(pars$aa))){
        pderltmP[i,i-length(pars$aa)-length(pars$ax)] <- pars$bxltm[i-length(pars$ax)-length(pars$aa)]/(pars$ax[i-length(pars$ax)-length(pars$aa)])^2
        pderltmP[i,i] <- 1
      }
      
      for(i in (2*length(pars$ax)+length(pars$aa)+1):(2*(length(pars$ax)+length(pars$aa)))){
        pderltmP[i,i-length(pars$aa)-length(pars$ax)] <- pars$baltm[i-2*length(pars$ax)-length(pars$aa)]/(pars$aa[i-2*length(pars$ax)-length(pars$aa)])^2
        pderltmP[i,i] <- 1
      }
      return(t(pderltmP) %*% mat %*% pderltmP)
    }
    
    if(model=="3pl"){
      pderltmP <- matrix(0, nrow=3*(length(pars$ax)+length(pars$aa)), ncol=3*(length(pars$ax)+length(pars$aa)))
      
      for(i in 1:(length(pars$ax)+length(pars$aa))){
        pderltmP[i,i] <- 1
      }
      
      for(i in (length(pars$ax)+length(pars$aa)+1):(2*length(pars$ax)+length(pars$aa))){
        pderltmP[i,i] <- -1/pars$ax[i-length(pars$ax)-length(pars$aa)]
      }
      for(i in (2*length(pars$ax)+length(pars$aa)+1):(2*length(pars$ax)+2*length(pars$aa))){
        pderltmP[i,i] <- -1/pars$aa[i-2*length(pars$ax)-length(pars$aa)]
      }
      
      for(i in (2*length(pars$ax)+2*length(pars$aa)+1):(3*length(pars$ax)+2*length(pars$aa))){
        pderltmP[i,i-length(pars$ax)-length(pars$aa)] <- pars$bxltm[i-2*length(pars$ax)-2*length(pars$aa)]/(pars$ax[i-2*length(pars$ax)-2*length(pars$aa)])^2
        pderltmP[i,i] <- 1
      }
      
      for(i in (3*length(pars$ax)+2*length(pars$aa)+1):(3*length(pars$ax)+3*length(pars$aa))){
        pderltmP[i,i-length(pars$ax)-length(pars$aa)] <- pars$baltm[i-3*length(pars$ax)-2*length(pars$aa)]/(pars$aa[i-3*length(pars$ax)-2*length(pars$aa)])^2
        pderltmP[i,i] <- 1
      }
      #print(pderltmP)
      #print(aaP)
      return(t(pderltmP) %*% mat %*% pderltmP)
    }  
  }
  if(design=="EG"){
    if(model=="2pl"){
      pderltmP <- matrix(0, nrow=2*length(pars$ax), ncol=2*length(pars$ax))
      for(i in 1:(length(pars$ax))){
        pderltmP[i,i] <- -1/pars$ax[i]
      }
      for(i in (length(pars$ax)+1):(2*length(pars$ax))){
        pderltmP[i,i-length(pars$ax)] <- pars$bxltm[i-length(pars$ax)]/(pars$ax[i-length(pars$ax)])^2
        pderltmP[i,i] <- 1
      }
      return(t(pderltmP) %*% mat %*% pderltmP)
    }
    
    if(model=="3pl"){
      pderltmP <- matrix(0, nrow=3*length(pars$ax), ncol=3*length(pars$ax))
      for(i in 1:(length(pars$ax))){
        pderltmP[i,i] <- 1
      }
      for(i in (length(pars$ax)+1):(2*length(pars$ax))){
        pderltmP[i,i] <- -1/pars$ax[i-length(pars$ax)]
      }
      for(i in (2*length(pars$ax)+1):(3*length(pars$ax))){
        pderltmP[i,i-length(pars$ax)] <- pars$bxltm[i-2*length(pars$ax)]/(pars$ax[i-2*length(pars$ax)])^2
        pderltmP[i,i] <- 1
      }
      return(t(pderltmP) %*% mat %*% pderltmP)
    }
  }  
}

probpl <- function(qpoints, b, model="3pl", a=0, c=0){
  out <- matrix(0, ncol = length(qpoints), nrow = length(b))
  if(model=="1pl"){
    tmat <- matrix(rep(b, length(qpoints)), nrow = length(b), ncol = length(qpoints))
    tmat <- t(t(tmat) - qpoints)
    out <- 1 / (1 + exp(tmat))
  }
  if(model=="2pl"){
    tmat <- matrix(rep(a, length(qpoints)), nrow = length(a), ncol = length(qpoints))
    tmat <- - t(t(tmat) * qpoints) + tmat * b
    out <- 1 / (1 + exp(tmat))
  }
  if(model=="3pl"){
    tmat <- matrix(rep(a, length(qpoints)), nrow = length(a), ncol = length(qpoints))
    tmat <- - t(t(tmat) * qpoints) + tmat * b
    out <- c + (1-c) / (1+exp(tmat))
  }
  return(out)
}


pderpl <- function(irt, qpoints, b, model = "3pl", a = 0, c = 0){
  #irt - the probabilities to answer each item (rows) correctly for the quadrature points (columns) considered
  #qpoints - the quadrature points and weights
  #b - the difficulty parameter for each item
  #model - "1pl", "2pl" or "3pl"
  #a - the discrimination parameter for each item
  #c - the guessing parameter for each item
  cprobs <- array(0, dim = c(length(irt[1,]), length(b), (length(b)+1)))
  probs <- matrix(0, ncol = length(irt[1,]), nrow = (length(b)+1))
  #cprobs - probabilities to achieve each score while answering each question correctly, for each ability level
  #probs - probability to achieve each score for each ability level (local equating score probs)
  
  probs <- LordWWc(irt, qpoints)
  for(j in 1:length(irt[,1])){
    input <- as.matrix(irt)
    input[j,] <- 1
    cprobs[,j,] <- t(LordWWc(input, qpoints$x))
  }
  #print(cprobs)
  if(model=="1pl"){
    pBpalpha <- matrix(0, nrow = length(b), ncol = (length(b)+1))
    Bx <- matrix(0, nrow=length(b), ncol=length(qpoints$x))
    #Bx Matrix of partial derivatives w/ resp to item parameters for each quadrature point
    for(i in 1:(length(b)+1)){
      Bx <- qpoints$w * t(cprobs[,,i] - probs[i,]) * irt
      #Weight each quadrature point under ~N(0,1) assumption
      pBpalpha[,i] <- rowSums(Bx)
      #Put the partial derivatives for each score value in the final matrix
    }
  }
  
  if(model=="2pl"){
    pBpalpha <- matrix(0, nrow = 2*length(b), ncol = (length(b)+1))
    #pBalpha - Matrix of partial derivatives w/ resp to item parameters across all quadrature points, for each score value
    Bx <- matrix(0, nrow=(2*length(b)), ncol=length(qpoints$x))
    #Bx Matrix of partial derivatives w/ resp to item parameters for each quadrature point, first c, then b, last a
    for(i in 1:(length(b)+1)){
      tmat <- cprobs[,,i] - probs[i,]
      if(!is.matrix(tmat)) tmat <- t(as.matrix(tmat))
      Bx[1:length(b),] <- t(tmat) * irt  * (-a)
      Bx[(length(b)+1):(length(b)+length(a)),] <- t(tmat * t(irt) * qpoints$x) - t(tmat) * irt * b
      Bx <- t(qpoints$w * t(Bx))
      #Weight each quadrature point under ~N(0,1) assumption
      pBpalpha[,i] <- rowSums(Bx)
      #Put the partial derivatives for each score value in the final matrix
    }
  }
  
  if(model=="3pl"){
    pBpalpha <- matrix(0, nrow = 3*length(b), ncol = (length(b)+1))
    #pBalpha - Matrix of partial derivatives w/ resp to item parameters across all quadrature points, for each score value
    Bx <- matrix(0, nrow=(3*length(b)), ncol=length(qpoints$x))
    #Bx Matrix of partial derivatives w/ resp to item parameters for each quadrature point, first c, then b, last a
    for(i in 1:(length(b)+1)){
      tmat <- cprobs[,,i] - probs[i,]
      if(!is.matrix(tmat)) tmat <- t(as.matrix(tmat))
      Bx[1:length(c),] <- t(tmat) * (1/(1-c))
      Bx[(length(c)+1):(length(c)+length(b)),] <- t(tmat) * (((irt - c) * (-a)) / (1-c))
      Bx[(length(c)+length(b)+1):(length(c)+length(b)+length(a)),] <- t(tmat * t((irt - c)/ (1-c)) * qpoints$x) + t(tmat) * (((irt - c) * (-b)) / (1-c)) 
      Bx <- t(qpoints$w * t(Bx))
      #Weight each quadrature point under ~N(0,1) assumption
      pBpalpha[,i] <- rowSums(Bx)
      #Put the partial derivatives for each score value in the final matrix
    }
  }
  
  return(pBpalpha)
}

itinfo <- function(a, b, c, theta){
  out <- matrix(0, nrow = length(a), ncol = length(theta))
  for(j in 1:length(a)){
    Pj <- c[j] + (1 - c[j]) / (1 + exp(-a[j] * (theta - b[j])))
    Qj <- 1 - Pj
    out[j,] <- a[j]^2 * Qj^2 * (Pj - c[j])^2 / ((1 - c[j])^2 * Pj) + a[j]^2 * Qj * (Pj - c[j])^2 / (1 - c[j])^2
  }
  return(out)
}

ditinfo <- function(a, b, c, theta){
  out <- matrix(0, nrow = 3 * length(a), ncol = length(theta))
  for(i in 1:length(a)){
    ai <- a[i]
    bi <- b[i]
    ci <- c[i]
    Pi <- ci + (1 - ci) / (1 + exp(-ai * (theta - bi)))
    Qi <- 1 - Pi
    
    Pici1ci <- Pi  / (1 - ci) - ci / (1 - ci)
    dPda <- (theta - bi) * Qi * Pici1ci
    dPdb <- -ai *  Qi * Pici1ci
    dPdc <- Qi / (1 - ci)
    ddPdthetada <- Qi * Pici1ci + ai * Qi * dPda / (1 - ci) - ai * dPda * Pici1ci
    ddPdthetadb <- ai * Qi * dPdb / (1 - ci) - ai * dPdb * Pici1ci
    ddPdthetadc <- - ai * dPdc * Pici1ci + ai * Qi * ((dPdc - 1) * (1 - ci) + (Pi - ci)) / (1 - ci)^2
    dinfoall1 <- 2 * ai * Pici1ci  / Pi
    dinfoall2 <- (1 - 2 * Pi) * ai^2 * Pici1ci^2 / Pi^2
    
    out[2 * length(a) + i, ] <- ddPdthetada * dinfoall1 - dPda * dinfoall2
    out[length(a) + i, ] <- ddPdthetadb * dinfoall1 - dPdb * dinfoall2
    out[i, ] <- ddPdthetadc * dinfoall1 - dPdc * dinfoall2
  }
  out
}

LordWWc <- function(p, qpoints){
  if(!is.matrix(p))
    p<-as.matrix(p)
  n<-dim(p)[1]
  N<-dim(p)[2]
  test<-array(0, c(n+1, N, n))
  q<-1-p
  l<-1
  test[1,,l]<-q[1,]
  test[2,,l]<-p[1,]
  testR<-test[1,,l]
  testR2<-test[2,,l]
  
  for(i in 2:n){
    for(r in 1:(i+1)){
      if(r==1)
        testR<-testR*q[i,]
      if(r>1 && r<(i+1))
        test[r,,l+1]<-test[r,,l]*q[i,]+test[r-1,,l]*p[i,]
      if(r==(i+1)){
        testR2<-testR2*p[i,]
        test[r,,l+1]<-testR2
      }
    }
    test[1,,l+1]<-testR
    l<-l+1
  }
  return(as.matrix(test[,,n]))
}


irtinput <- function(P, x, a, robust, model, SE = T, catsX = 0, catsA = 0){
  covP <- NULL
  if(model == "2pl"){
    nX <- length(x) - 1
    nA <- length(a) - 1
    if("ltm" %in% class(P)){
      bx <- as.numeric(coef.ltm(P)[,1])[1:nX]
      baP <- as.numeric(coef.ltm(P)[,1])[(nX + 1):(nX + nA)]
      ax <- as.numeric(coef.ltm(P)[,2])[1:nX]
      aaP <- as.numeric(coef.ltm(P)[,2])[(nX + 1):(nX + nA)]
      if(P$IRT.param){
        bxltm <- -ax*bx
        baPltm <- -aaP*baP
      } else{
        bxltm <- bx
        baPltm <- baP
        bx <- -bxltm/ax
        baP <- -baPltm/aaP
      }
      N <- dim(P$X)[1]
      ltmP <- P
      P <- ltmP$X
      if(SE) covP <- vcov.ltm(ltmP, robust = robust)
    } else if(class(P) %in% c("SingleGroupClass", "ConfirmatoryClass")){
      myspP <- extract.mirt(P, "parvec")
      b <- myspP[seq(2, 2 * (nX + nA), by = 2)]
      a <- myspP[seq(1, 2 * (nX + nA), by = 2)]
      ax <- a[1:nX]
      bx <- b[1:nX]
      baP <- b[(nX + 1):(nX + nA)]
      aaP <- a[(nX + 1):(nX + nA)]
      if(SE){
        mycovP <- extract.mirt(P, "vcov")
        upmat <- cbind(mycovP[seq(2, 2*(nX + nA), by = 2), seq(2, 2*(nX + nA), by = 2)], mycovP[seq(2, 2*(nX + nA), by = 2), seq(1, 2*(nX + nA), by = 2)])
        lowmat <- cbind(mycovP[seq(2, 2*(nX + nA), by = 2), seq(1, 2*(nX + nA), by = 2)], mycovP[seq(1, 2*(nX + nA), by = 2), seq(1, 2*(nX + nA), by = 2)])
        covP <- rbind(upmat, lowmat)
      }
      bxltm <- bx
      baPltm <- baP
      bx <- -bxltm/ax
      baP <- -baPltm/aaP
      dataP <- extract.mirt(P, "tabdata")
      N <- nrow(dataP)
      ltmP <- P
      P <- dataP
    } else if(is.matrix(P)){
      if((length(a)+length(x)-2) != ncol(P))
        return("Unsupported input. Input matrices must have rows denoting individuals and columns denoting items.")
      ltmP <- ltm(P ~ z1, IRT.param=FALSE)
      bx <- as.numeric(coef.ltm(ltmP)[,1])[1:nX]
      baP <- as.numeric(coef.ltm(ltmP)[,1])[(nX + 1):(nX + nA)]
      ax <- as.numeric(coef.ltm(ltmP)[,2])[1:nX]
      aaP <- as.numeric(coef.ltm(ltmP)[,2])[(nX + 1):(nX + nA)]
      bxltm <- bx
      baPltm <- baP
      bx <- -bxltm/ax
      baP <- -baPltm/aaP
      N <- dim(P)[1]
      if(SE) covP <- vcov.ltm(ltmP, robust = robust)
    } else if(is.list(P)){
      if(P$IRT.param == T){
        ax <- P$pars[1:nX,1]
        bx <- P$pars[1:nX,2]
        aaP <- P$pars[(nX + 1):(nX + nA),1]
        baP <- P$pars[(nX + 1):(nX + nA),2]
        baPltm <- -ax*bx
        bxltm <- -aaP*baP
        N <- P$N
        if(SE) covP <- P$cov
        ltmP <- NULL
        P <- matrix(0)
      } else{  
        bxltm <- P$pars[1:nX,2]
        baPltm <- P$pars[(nX + 1):(nX + nA),2]
        ax <- P$pars[1:nX,1]
        aaP <- P$pars[(nX + 1):(nX + nA),1]
        bx <- -bxltm/ax
        baP <- -baPltm/aaP
        N <- P$N      
        if(SE) covP <- P$cov
        ltmP <- NULL
        P <- matrix(0)
      }
    } else return("Unsupported input. P must be either an object created by the packages ltm or mirt, a matrix of responses or a list with parameters, covariance matrix and sample size.")
    
    return(list(bxltm=bxltm, baPltm=baPltm, ax=ax, aaP=aaP, bx=bx, baP=baP, N=N, covP=covP, ltmP=ltmP, P=P, IRT.param=T))
  }
  if(model=="3pl"){
    nX <- length(x) - 1
    nA <- length(a) - 1
    if("tpm" %in% class(P)){
      cx <- as.numeric(coef.tpm(P)[,1])[1:(length(x)-1)]
      caP <- as.numeric(coef.tpm(P)[,1])[(length(x)):((length(x)+length(a)-2))]
      bx <- as.numeric(coef.tpm(P)[,2])[1:(length(x)-1)]
      baP <-  as.numeric(coef.tpm(P)[,2])[(length(x)):((length(x)+length(a)-2))]
      ax <- as.numeric(coef.tpm(P)[,3])[1:(length(x)-1)]
      aaP <- as.numeric(coef.tpm(P)[,3])[(length(x)):((length(x)+length(a)-2))]
      if(P$IRT.param){
        bxltm <- -ax*bx
        baPltm <- -aaP*baP
      } else{
        bxltm <- bx
        baPltm <- baP
        bx <- -bxltm/ax
        baP <- -baPltm/aaP
      }
      N <- dim(P$X)[1]
      ltmP <- P
      P <- ltmP$X
      if(SE) covP <- vcov.tpm(ltmP)
    } else if(class(P) %in% c("SingleGroupClass", "ConfirmatoryClass")){
      myspP <- extract.mirt(P, "parvec")
      a <- myspP[seq(1, 3 * (nX + nA), by = 3)]
      b <- myspP[seq(2, 3 * (nX + nA), by = 3)]
      cpar <- myspP[seq(3, 3 * (nX + nA), by = 3)]
      ax <- a[1:nX]
      bx <- b[1:nX]
      cx <- cpar[1:nX]
      aaP <- a[(nX + 1):(nX + nA)]
      baP <- b[(nX + 1):(nX + nA)]
      caP <- cpar[(nX + 1):(nX + nA)]
      if(SE){
        mycovP <- extract.mirt(P, "vcov")
        upmat <- cbind(mycovP[seq(3, 3*(nX + nA), by = 3), seq(3, 3*(nX + nA), by = 3)], mycovP[seq(3, 3*(nX + nA), by = 3), seq(2, 3*(nX + nA), by = 3)], mycovP[seq(3, 3*(nX + nA), by = 3), seq(1, 3*(nX + nA), by = 3)])
        midmat <- cbind(mycovP[seq(2, 3*(nX + nA), by = 3), seq(3, 3*(nX + nA), by = 3)], mycovP[seq(2, 3*(nX + nA), by = 3), seq(2, 3*(nX + nA), by = 3)], mycovP[seq(2, 3*(nX + nA), by = 3), seq(1, 3*(nX + nA), by = 3)])
        lowmat <- cbind(mycovP[seq(1, 3*(nX + nA), by = 3), seq(3, 3*(nX + nA), by = 3)], mycovP[seq(1, 3*(nX + nA), by = 3), seq(2, 3*(nX + nA), by = 3)], mycovP[seq(1, 3*(nX + nA), by = 3), seq(1, 3*(nX + nA), by = 3)])
        covP <- rbind(upmat, midmat, lowmat)
      }
      bxltm <- bx
      baPltm <- baP
      bx <- -bxltm/ax
      baP <- -baPltm/aaP
      dataP <- extract.mirt(P, "tabdata")
      N <- nrow(dataP)
      ltmP <- P
      P <- dataP
    } else return("Unsupported input. P must be either an object created by the packages ltm or mirt, a matrix of responses or a list with parameters, covariance matrix and sample size.")
    
    return(list(bxltm=bxltm, baPltm=baPltm, ax=ax, aaP=aaP, bx=bx, baP=baP, cx=cx, caP=caP, N=N, covP=covP, ltmP=ltmP, P=P, IRT.param=T))
  }
  
  
  if(model %in% c("GPCM")){
    JX <- length(catsX)
    JA <- length(catsA)
    
    ax <- numeric(JX)
    bx <- vector("list", JX)
    aaP <- numeric(JA)
    baP <- vector("list", JA)
    
    if(class(P) == "SingleGroupClass"){
      for(i in 1:JX){
        ttpar <- extract.item(P, i)
        ax[i] <- ttpar@par[1]
        bx[[i]] <- -(tail(ttpar@par, catsX[i]-1) - c(0, tail(ttpar@par, catsX[i]-1)[-(catsX[i]-1)])) / ax[i]
      }
      for(i in 1:JA){
        ttpar <- extract.item(P, i+JX)
        aaP[i] <- ttpar@par[1]
        baP[[i]] <- -(tail(ttpar@par, catsA[i]-1) - c(0, tail(ttpar@par, catsA[i]-1)[-(catsA[i]-1)])) / aaP[i]
      }
      dataP <- extract.mirt(P, "tabdata")
      N <- nrow(dataP)
      if(SE) covP <- extract.mirt(P, "vcov")
      ltmP <- P
      P <- dataP
    }
    return(list(ax=ax, aaP=aaP, bx=bx, baP=baP, N=N, covP=covP, ltmP=ltmP, P=P, IRT.param=F))
  }
  
  if(model %in% c("GRM")){
    JX <- length(catsX)
    JA <- length(catsA)
    
    ax <- numeric(JX)
    bx <- vector("list", JX)
    aaP <- numeric(JA)
    baP <- vector("list", JA)
    
    if(class(P) == "SingleGroupClass"){
      for(i in 1:JX){
        ttpar <- extract.item(P, i)
        ax[i] <- ttpar@par[1]
        bx[[i]] <- -ttpar@par[-1] / ax[i]
      }
      for(i in 1:JA){
        ttpar <- extract.item(P, i+JX)
        aaP[i] <- ttpar@par[1]
        baP[[i]] <- -ttpar@par[-1] / aaP[i]
      }
      dataP <- extract.mirt(P, "tabdata")
      N <- nrow(dataP)
      if(SE) covP <- extract.mirt(P, "vcov")
      ltmP <- P
      P <- dataP
    }
    
    return(list(ax=ax, aaP=aaP, bx=bx, baP=baP, N=N, covP=covP, ltmP=ltmP, P=P, IRT.param=F))
  }
}


#marginal reliability
irtmrcbare <- function(aP, bP, model, cats, qpoints, cP = 0, type = "mirt", SE = T){
  if(model == "GPCM"){
    J <- length(cats)
    K <- sum(cats) - J
    P1 <- polyprob(aP, bP, cats, model, qpoints$x)
    info <- numeric(length(qpoints$x))
    for(j in 1:J){
      tempRF1 <- numeric(length(qpoints$x))
      dtempRF1 <- numeric(length(qpoints$x))
      for(k in 1:(cats[j])) tempRF1 <- k * P1[[j]][k,] + tempRF1
      for(k in 1:(cats[j])) dtempRF1 <- P1[[j]][k,] * aP[j]^2 * (k - tempRF1)^2 + dtempRF1
      info <- dtempRF1 + info
    }
  }
  if(model %in% c("1-PL", "2-PL", "3-PL")){
    if(model == "1-PL"){
      bpar <- bP
      apar <- rep(0, length(bpar))
      cpar <- rep(0, length(bpar)) 		
    }
    if(model == "2-PL"){
      apar <- aP
      bpar <- bP
      cpar <- rep(0, length(bpar))
    }
    if(model == "3-PL"){
      apar <- aP
      bpar <- bP
      cpar <- cP
    }
    info <- colSums(itinfo(apar, bpar, cpar, qpoints$x))
  }
  if(SE){
    if(model == "GPCM"){
      pderX <- polypder(aP, bP, cats, P1, model, qpoints$x)
      dinfodalpha <- matrix(0, nrow = sum(cats), ncol = length(qpoints$x))
      for(j in 1:J){
        sumcP1 <- numeric(length(qpoints$x))
        sumdPdab <- matrix(0, nrow = cats[j], ncol = length(qpoints$x))
        for(i in 1:(cats[j])){
          sumcP1 <- i * P1[[j]][i,] + sumcP1
          sumdPdab <- i * pderX[[j]][[i]] + sumdPdab
        }
        dinfojdalpha <- matrix(0, nrow = cats[j], ncol = length(qpoints$x))
        for(kj in 1:(cats[j])){
          dPda <- P1[[j]][kj,] * (kj - sumcP1) +  pderX[[j]][[kj]][cats[j], ] * aP[j] * (kj - sumcP1) - aP[j] * P1[[j]][kj,] * sumdPdab[cats[j],]
          dinfojdalpha[cats[j],] <- aP[j] * (kj - sumcP1) * (2 * dPda  - aP[j] * (kj - sumcP1) * pderX[[j]][[kj]][cats[j],] ) + dinfojdalpha[cats[j],]
          for(k in 2:(cats[j])){
            dPdb <- aP[j] * pderX[[j]][[kj]][k-1, ] * (kj - sumcP1) - aP[j] * P1[[j]][kj,] * sumdPdab[k-1,]
            dinfojdalpha[k-1,] <- aP[j] * (kj - sumcP1) * (2 * dPdb  - aP[j] * (kj - sumcP1) * pderX[[j]][[kj]][k-1,]) + dinfojdalpha[k-1,]
          }
        }
        if(j == 1) dinfodalpha[1:(cats[1]),] <- dinfojdalpha else dinfodalpha[(sum(cats[1:(j-1)])+1):(sum(cats[1:j])),] <- dinfojdalpha
      }
    }
    if(model == "3-PL") dinfodalpha <- ditinfo(apar, bpar, cpar, qpoints$x)
    if(type == "Green"){
      mymat <- t(t(dinfodalpha) * (1 / info^2))
      dwdalpha <- colSums(t(mymat) *  qpoints$w)
      coef <- 1 - sum((1 / info) * qpoints$w)
      return(list(coef = coef, pder = dwdalpha, type = "Green"))
    }
    if(type == "mirt"){
      mymat <- t(t(dinfodalpha) * (1 / (info + 1) - info / (info + 1)^2))
      dwdalpha <- colSums(t(mymat) * qpoints$w)
      coef <- sum(info / (1 + info) * qpoints$w)
      return(list(coef = coef, pder = dwdalpha, type = "mirt"))
    }
  } else{
    if(type == "Green") return(list(coef = 1 - sum((1 / info) / (1 + 1/info) * qpoints$w), type = "Green"))
    if(type == "mirt") return(list(coef = sum(info / (1 + info) * qpoints$w), type = "mirt"))
  }
}

#test reliability

irttrcbare <- function(aP, bP, model, cats, qpoints, cP = 0, SE = T){
  if(model == "GPCM"){
    J <- length(cats)
    K <- sum(cats) - J
    irtr <- polyprob(aP, bP, cats, model, qpoints$x)
    res <- 0
    for(i in 1:J) res <- colSums((irtr[[i]]) * (0:(cats[i]-1))^2) - colSums((irtr[[i]]) * (0:(cats[i]-1)))^2 + res
    vare <- sum(res  * qpoints$w)
    rP <- rowSums(cmnom(cats, irtr, qpoints))
    scores <- 0:K
    varX <- as.vector(rP %*% scores^2) - as.vector(rP %*% scores)^2
    sumWpderX <- matrix(0, nrow = sum(cats), ncol = length(qpoints$x))
    sumW2pderX <- matrix(0, nrow = sum(cats), ncol = length(qpoints$x))
    if(SE){
      pderX <- polypder(aP, bP, cats, irtr, model, qpoints$x)
      
      for(k in 2:cats[1]){
        sumWpderX[1:cats[1],] <- pderX[[1]][[k]] * (k-1) + sumWpderX[1:cats[1],]
        sumW2pderX[1:cats[1],] <- pderX[[1]][[k]] * (k-1)^2 + sumW2pderX[1:cats[1],]
      }
      sumWpderX[1:cats[1],] <- t(colSums((irtr[[1]]) * (0:(cats[1]-1))) * t(sumWpderX[1:cats[1],]))
      for(j in 2:J){
        for(k in 2:cats[j]){
          sumWpderX[(sum(cats[1:(j-1)])+1):(sum(cats[1:j])),] <- pderX[[j]][[k]] * (k-1) + sumWpderX[(sum(cats[1:(j-1)])+1):(sum(cats[1:j])),]
          sumW2pderX[(sum(cats[1:(j-1)])+1):(sum(cats[1:j])),] <- pderX[[j]][[k]] * (k-1)^2 + sumW2pderX[(sum(cats[1:(j-1)])+1):(sum(cats[1:j])),]
        }
        sumWpderX[(sum(cats[1:(j-1)])+1):(sum(cats[1:j])),] <- t(colSums((irtr[[j]]) * (0:(cats[j]-1))) * t(sumWpderX[(sum(cats[1:(j-1)])+1):(sum(cats[1:j])),]))
      }
      pdervare <- numeric(sum(cats))
      for(i in 1:sum(cats)) pdervare[i] <-  sum((sumW2pderX[i,] - 2 * sumWpderX[i,]) * qpoints$w)
      
      pderX <- polypderpl(aP, bP, cats, irtr, model, qpoints, qpoints$x)
      pdermatX <- matrix(0, nrow = K + 1, ncol = sum(cats))
      for(i in 1:(K + 1)){
        pdermatX[i, 1:cats[1]] <- pderX[[i]][[1]]
        for(j in 2:J){
          pdermatX[i, (sum(cats[1:(j - 1)]) + 1):sum(cats[1:j])] <- pderX[[i]][[j]]
        }
      }
      pdervarX1 <- pdervarX2 <- numeric(sum(cats))
      for(i in 2:(K + 1)) {
        pdervarX1 <- pdermatX[i,] * (i-1)^2 + pdervarX1
      }
      for(i in 2:(K + 1)) pdervarX2 <- 2 * as.vector((rP %*% scores)) * pdermatX[i,] * (i-1) + pdervarX2
      pdervarX <- pdervarX1 - pdervarX2
      pderrho <- (1 / varX) * ((vare / varX) * pdervarX - pdervare)
      return(list(coef = as.numeric(1 - vare / varX), pder = pderrho))
    } else(return(list(coef = as.numeric(1 - vare / varX))))
  }
  if(model=="3-PL"){
    J <- length(cats)
    K <- sum(cats) - J
    irtx <- probpl(qpoints$x, bP, a = aP, c = cP) 
    rP <- colSums(t(LordWWc(irtx, qpoints$x)) * qpoints$w)
    res <- 0
    for(i in 1:J) res <- irtx[i,] - irtx[i,]^2 + res
    vare <- sum(res  * qpoints$w)
    scores <- 0:K
    varX <- as.vector(rP %*% (scores^2)) - as.vector((rP %*% scores)^2)
    
    if(SE){
      pdervare <- numeric(sum(cats))
      sumWpderX <- matrix(0, nrow = 3 * length(aP), ncol = length(qpoints$x))
      sumW2pderX <- matrix(0, nrow = 3 * length(aP), ncol = length(qpoints$x))
      k <- 0
      for(i in 1:length(aP)){
        ai <- aP[i]
        bi <- bP[i]
        ci <- cP[i]
        Pi <- ci + (1 - ci) / (1 + exp(-ai * (qpoints$x - bi)))
        Qi <- 1 - Pi
        
        Pici1ci <- Pi  / (1 - ci) - ci / (1 - ci)
        dPda <- (qpoints$x - bi) * Qi * Pici1ci
        dPdb <- -ai *  Qi * Pici1ci
        dPdc <- Qi / (1 - ci)
        
        sumWpderX[i,] <- Pi * dPdc
        sumW2pderX[i,] <- dPdc
        sumWpderX[length(aP) + i,] <- Pi * dPdb
        sumW2pderX[length(aP) + i,] <- dPdb
        sumWpderX[2 * length(aP) + i,] <- Pi * dPda
        sumW2pderX[2 * length(aP) + i,] <- dPda
      }		
      for(i in 1:(3 * length(aP))) pdervare[i] <-  sum((sumW2pderX[i,] - 2 * sumWpderX[i,]) * qpoints$w)
      pderX <- pderpl(irtx, qpoints, bP, model = "3pl", a = aP, c = cP)
      pdervarX1 <- pdervarX2 <- numeric(3 * length(aP))
      for(i in 2:(K + 1)) {
        pdervarX1 <- pderX[,i] * (i-1)^2 + pdervarX1
      }
      for(i in 2:(K + 1)) pdervarX2 <- 2 * as.vector((rP %*% scores)) * pderX[,i] * (i-1) + pdervarX2
      pdervarX <- pdervarX1 - pdervarX2
      pderrho <- (1 / varX) * ((vare / varX) * pdervarX - pdervare)
      
      return(list(coef = as.numeric(1 - vare / varX), pder = pderrho))
    } else(return(list(coef = as.numeric(1 - vare / varX))))
  }
}

calpha <- function(mymat){
  K <- ncol(mymat)
  varY <- apply(mymat, 2, var)
  return(K / (K - 1) * (1 - sum(varY) / var(rowSums(mymat))))
}

#Wrapper function for CTT reliability
ctt.reliability <- function(input, relcoef = "alpha", SE = FALSE){
  if(!SE){
    if(relcoef=="alpha") return(new("relout", est = calpha(input), type = "alpha"))
  }
}

#Wrapper function for IRT reliability
irtreliability <- function(input, model, cats, relcoef = "trc", nquad = 49, SE = TRUE){
  qpoints <- gaussHermiteData(nquad)
  qpoints$x <- -sqrt(2) * (qpoints$x)
  qpoints$w <- qpoints$w / sqrt(pi)

  if(model == "GPCM"){
    J <- length(cats)
    K <- sum(cats) - J
    
    if("SingleGroupClass" %in% class(input)){
		P <- parmirt(input, NULL, model, cats, 0, 0, SE)
		aP <- P[[1]][1:J]
		bP <- vector("list", J)
		for(i in 1:J) bP[[i]] <- P[[2]][[i]]
		if(SE) vcovP <- P[[3]]
	}
    # if(mixirt){
		# aP <- numeric(J)
		# bP <- vector("list", J)
		# k <- 1
		# for(i in 1:J){
			# aP[i] <- input$par[k]
			# bP[[i]] <- -input$par[(k+1):(k + cats[i] - 1)] / aP[i]
			# k <- k + cats[i]
		# }
		# if(SE){
			# if(robust) vcovP <- t(adjgpcmmixirt(input)) %*% solve(input$Amat) %*% input$Bmat %*% solve(input$Amat) %*% adjgpcmmixirt(input) else vcovP <- t(adjgpcmmixirt(input)) %*% solve(input$Amat) %*% adjgpcmmixirt(input)
		# }
	# }
	
    if(relcoef == "mrc") rcout <- irtmrcbare(aP, bP, model, cats, qpoints, SE = SE)
    if(relcoef == "trc") rcout <- irttrcbare(aP, bP, model, cats, qpoints, SE = SE)
    if(SE) return(new("relout", est = rcout$coef, cov = t(rcout$pder) %*% vcovP %*% rcout$pder, pder = rcout$pder, type = relcoef)) else return(new("relout", est = rcout$coef, type = relcoef))
  }
  if(model == "3-PL"){
    J <- length(cats)
    K <- sum(cats) - J
    
    if("SingleGroupClass" %in% class(input)){
      if(SE) parvect <- irtinput(input, 0:K, 0, F, "3pl", SE = T) else parvect <- irtinput(input, 0:K, 0, F, "3pl", SE = F)
      
      aP <- parvect$ax
      bP <- parvect$bx
      cP <- exp(parvect$cx) / (1 + exp(parvect$cx))
      covalpha <- parvect$covP
      adjcovalpha <- adjltm(covalpha, pars=list(ax=aP, bxltm=parvect$bxltm), design = "EG", model="3pl")
      
      vectadj <- rep(1, 3 * length(aP))
      vectadj[1:length(aP)] <- exp(parvect$cx) / (1 + exp(parvect$cx))^2
      vcovP <- diag(vectadj) %*% adjcovalpha %*% t(diag(vectadj))
    }
	
	# if(mixirt){
		# aP <- bP <- cP <- bstar <- cstar <- numeric(J)
		# k <- 1
		# for(i in 1:J){
			# aP[i] <- input$par[k]
			# bP[i] <- -input$par[k+1] / aP[i]
			# cP[i] <- exp(input$par[k+2]) / (1 + exp(input$par[k+2]))
			# bstar[i] <- input$par[k+1]
			# cstar[i] <- input$par[k+2]
			# k <- k + 3
		# }
		# if(SE){
			# if(robust){
				# vcovP <- t(adjgpcmmixirt(input)) %*% solve(input$Amat) %*% input$Bmat %*% solve(input$Amat) %*% (adjgpcmmixirt(input))
			# } else{
				# if(is.null(input$Amat)) vcovP <- t(adjgpcmmixirt(input)) %*% solve(input$Bmat) %*% (adjgpcmmixirt(input)) else vcovP <- t(adjgpcmmixirt(input)) %*% solve(input$Amat) %*% (adjgpcmmixirt(input))
			# }
			# #if(robust) vcovP <- solve(input$Amat) %*% input$Bmat %*% solve(input$Amat) else vcovP <- solve(input$Amat)
			# upmat <- cbind(vcovP[seq(1, 3*J, by = 3), seq(1, 3*J, by = 3)], vcovP[seq(1, 3*J, by = 3), seq(2, 3*J, by = 3)], vcovP[seq(1, 3*J, by = 3), seq(3, 3*J, by = 3)])
			# midmat <- cbind(vcovP[seq(2, 3*J, by = 3), seq(1, 3*J, by = 3)], vcovP[seq(2, 3*J, by = 3), seq(2, 3*J, by = 3)], vcovP[seq(2, 3*J, by = 3), seq(3, 3*J, by = 3)])
			# lowmat <- cbind(vcovP[seq(3, 3*J, by = 3), seq(1, 3*J, by = 3)], vcovP[seq(3, 3*J, by = 3), seq(2, 3*J, by = 3)], vcovP[seq(3, 3*J, by = 3), seq(3, 3*J, by = 3)])
			# vcovP <- rbind(upmat, midmat, lowmat)
			# #covalpha <- rbind(upmat, midmat, lowmat)
			# #adjcovalpha <- adjltm(covalpha, pars=list(ax=aP, bxltm=bstar), design = "EG", model="3pl")
			# #vectadj <- rep(1, 3 * J)
			# #vectadj[1:length(aP)] <- exp(cstar) / (1 + exp(cstar))^2
			# #vcovP <- diag(vectadj) %*% adjcovalpha %*% t(diag(vectadj))
		# }
	# }
    # if("list" %in% class(input)){
      # aP <- input$a
      # bP <- input$b
      # cP <- exp(input$c) / (1 + exp(input$c))
      
      # covalpha <- input$cov
      # adjcovalpha <- adjltm(covalpha, pars=list(ax=aP, bxltm=input$bltm), design = "EG", model="3pl")
      # vectadj <- rep(1, 3 * length(aP))
      # vectadj[1:length(aP)] <- exp(input$c) / (1 + exp(input$c))^2
      # vcovP <- diag(vectadj) %*% adjcovalpha %*% t(diag(vectadj))
      
    # }

    if(relcoef == "mrc") rcout <- irtmrcbare(aP, bP, model, cats, qpoints, cP = cP, SE = SE)
    if(relcoef == "trc") rcout <- irttrcbare(aP, bP, model, cats, qpoints, cP = cP, SE = SE)
    if(SE) return(new("relout", est = rcout$coef, cov = t(rcout$pder) %*% vcovP %*% rcout$pder, pder = rcout$pder, type = relcoef)) else return(new("relout", est = rcout$coef, type = relcoef))
  }
}

