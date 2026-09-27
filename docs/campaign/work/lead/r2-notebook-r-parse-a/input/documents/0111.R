MsplineBasis <-
  function(x, df = NULL, knots = NULL, degree = 3, intercept = FALSE,
           Boundary.knots = range(x), periodic = FALSE){
    
    if(periodic){
      if(!is.null(df) & !intercept) df <- df + 1L
      X <- MsplineBasis(x = x, df = df, knots = knots, degree = degree, 
                        Boundary.knots = Boundary.knots)
      X <- X[,1:(ncol(X)-1)]
      if(intercept) X <- cbind(1, X)
      attr(X, "intercept") <- intercept
      attr(X, "periodic") <- periodic
      return(X)
    }
    
    # check degree and order
    degree <- as.integer(degree)
    ord <- degree + 1L
    if(ord < 1L) stop("Input 'degree' must be an integer >= 0L.")
    
    # check Boundary.knots
    if(min(x) < Boundary.knots[1]) warning("Minimum 'x' is outside of 'Boundary.knots'.")
    if(max(x) > Boundary.knots[2]) warning("Maximum 'x' is outside of 'Boundary.knots'.")
    maxid <- which(x == Boundary.knots[2])
    nmax <- length(maxid)
    
    # define knots
    if(!is.null(df) && is.null(knots)){
      nIknots <- df - ord + !intercept
      if(nIknots < 0L){
        nIknots <- 0L
        warning("Input 'df' was too small and has been reset to minimum possible df.")
      }
      knots <- if(nIknots > 0L){ 
        knots <- seq(0, 1, length.out = nIknots + 2L)[-c(1L, nIknots+2L)]
        unname(quantile(x, knots))
      }
    }
    Aknots <- sort(c(rep(Boundary.knots, ord), knots))
    
    # get number of observations and knots
    nx <- length(x)
    nk <- length(Aknots)
    
    # build up design for given degree
    knotdif <- Aknots[2:nk] - Aknots[1:(nk-1)]
    if(any(knotdif == 0)) knotdif[knotdif==0] <- 1
    Xmat <- (outer(x, Aknots[1:(nk-1)], FUN=">=") & outer(x, Aknots[2:nk], FUN="<")) * matrix(1/knotdif, nrow=nx, ncol=nk-1, byrow=TRUE)
    if(nmax > 0L) Xmat[maxid,] <- (outer(rep(Boundary.knots[2],nmax), Aknots[1:(nk-1)], FUN=">=") & outer(rep(Boundary.knots[2],nmax), Aknots[2:nk], FUN="<=")) * matrix(1/knotdif, nrow=nmax, ncol=nk-1, byrow=TRUE)
    if(ord > 1L){
      for(k in 2:ord){
        X1new <- outer(x, Aknots[1:(nk-k)], FUN="-") * Xmat[,1:(nk-k)]
        X2new <- outer(-x, Aknots[(1+k):nk], FUN="+") * Xmat[,2:(nk-k+1)]
        knotdif <- Aknots[(1+k):nk] - Aknots[1:(nk-k)]
        if(any(knotdif == 0)) knotdif[knotdif==0] <- 1
        Xmat <- (k/(k-1)) * (X1new + X2new) * matrix(1/knotdif, nrow=nx, ncol=nk-k, byrow=TRUE)
      } # end for(j in 2:ord)
    } # end if(ord > 1L)
    
    if(!intercept) Xmat <- Xmat[,-1]
    dimnames(Xmat) <- list(1:nx, 1:ncol(Xmat))
    a <- list(degree = degree, knots = knots, Boundary.knots = Boundary.knots, 
              intercept = intercept, periodic = periodic)
    attributes(Xmat) <- c(attributes(Xmat), a)
    return(Xmat)
    #return(list(X=Xmat, degree=degree, knots=knots, Boundary.knots=Boundary.knots, intercept=intercept))
    
  }