redux2ArAr <- function(x,irr,fract=NULL,ca=NULL,
                       k=NULL,format=1,file=NULL){
    Cl <- corrections(x,irr,fract=fract,ca=ca,k=k)
    Y <- getABCDEF(Cl)
    ni <- length(Y$intercepts)
    out <- list()
    class(out) <- "ArAr"
    R <- get4039(Cl,irr)
    JJ <- getJfactors(R)
    Jfact <- subset(JJ,labels="J:")
    ns <- nruns(Jfact)
    if (ns==1) {
        out$J <- c(Jfact$intercepts,sqrt(Jfact$covmat))
    } else if (diff(range(Jfact$pos))==0) {
        out$J <- c(Jfact$intercepts[1],sqrt(Jfact$covmat[1,1]))
    } else {
        stop("The dataset contains more than one J-factor.")
    }
    labels <- rep(c('Ar39Ar40','Ar36Ar40','Ar39Ar36','Ar40Ar36'),ns)
    out$x <- rep(0,4*ns)
    out$covmat <- matrix(0,4*ns,4*ns)
    J <- matrix(0,nrow=4*ns,ncol=ni)
    hasKglass <- "K-glass" %in% Cl$labels
    hasCasalt <- "Ca-salt" %in% Cl$labels
    for (i in 1:ns){
        ri <- (i-1)*4 # row index (39/40,36/40,39/36,40/36)
        ci <- (i-1)*6 # column index (A,B,C,D,E,F)
        label <- Y$labels[i]
        AA <- Y$intercepts[getindices(Y,label,num='A')]
        CC <- Y$intercepts[getindices(Y,label,num='C')]
        EE <- Y$intercepts[getindices(Y,label,num='E')]
        if (hasKglass) {
            DD <- Y$intercepts[getindices(Y,label,num='D')]
        } else {
            DD <- 0
        }
        if (!hasCasalt | expired(irr[[Y$irr[i]]],Y$thedate[i],Y$param$l7)) {
            BB <- 0
            FF <- 0
        } else {
            BB <- Y$intercepts[getindices(Y,label,num='B')]
            FF <- Y$intercepts[getindices(Y,label,num='F')]
        }
        out$x[ri+1] <- (EE-FF)/(1-DD)        # X1=Ar39/Ar40
        out$x[ri+2] <- (AA-BB-CC)/(1-DD)     # Y1=Ar36/Ar40
        out$x[ri+3] <- (EE-FF)/(AA-BB-CC)    # X2=Ar39/Ar36
        out$x[ri+4] <- (1-DD)/(AA-BB-CC)     # Y2=Ar40/Ar36
        J[ri+1,ci+4] <- -(EE-FF)/(1-DD)^2    # dX1dD
        J[ri+1,ci+5] <- 1/(1-DD)             # dX1dE
        J[ri+1,ci+6] <- -1/(1-DD)            # dX1dF
        J[ri+2,ci+1] <- 1/(1-DD)             # dY1dA
        J[ri+2,ci+2] <- -1/(1-DD)            # dY1dB
        J[ri+2,ci+3] <- -1/(1-DD)            # dY1dC
        J[ri+2,ci+4] <- -(AA-BB-CC)/(1-DD)^2 # dY1dD
        J[ri+3,ci+1] <- (FF-EE)/(AA-BB-CC)^2 # dX2dA
        J[ri+3,ci+2] <- (EE-FF)/(AA-BB-CC)^2 # dX2dB
        J[ri+3,ci+3] <- (EE-FF)/(AA-BB-CC)^2 # dX2dC
        J[ri+3,ci+5] <- 1/(AA-BB-CC)         # dX2dE
        J[ri+3,ci+6] <- -1/(AA-BB-CC)        # dX2dF
        J[ri+4,ci+1] <- -(1-DD)/(AA-BB-CC)^2 # dY2dA
        J[ri+4,ci+2] <- (1-DD)/(AA-BB-CC)^2  # dY2dB
        J[ri+4,ci+3] <- (1-DD)/(AA-BB-CC)^2  # dY2dC
        J[ri+4,ci+4] <- -1/(AA-BB-CC)        # dY2dD
    }
    out$covmat <- J %*% Y$covmat %*% t(J)
    names(out$x) <- labels
    rownames(out$covmat) <- labels
    colnames(out$covmat) <- labels
    out
}