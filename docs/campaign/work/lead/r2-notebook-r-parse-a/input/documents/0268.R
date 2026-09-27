resid.corr.test=function(obj, lag.cor=1, alpha=0.95, dig1=5, dig2=3){
  # Residual correlation test
  # Tests the residuals of the country VECM models for serial autocorrelation
  # Input arguments:
  #                  x......GVAR object
  #                  lag....the degree of serial autocorrelation that should be tested for
  #                  alpha..significance level
  #                  dig1 / dig2...nr. of digits for the test statistic / p-value
  if(!inherits(obj, "bgvar")) {stop("Please provide a `bgvar` object.")}
  # get data and arguments - note each second column of V has sign switched -> does not impact on results of F test so keep it as it is
  xglobal  <- obj$xglobal    
  res      <- obj$cc.results$res
  lags     <- obj$args$lags
  pmax     <- max(lags)
  bigT     <- nrow(xglobal)-pmax
  pidx     <- 1:lag.cor
  varNames <- colnames(xglobal)
  cN       <- unique(sapply(strsplit(varNames,".",fixed=TRUE),function(x) x[1]))
  vars     <- unique(sapply(strsplit(varNames,".",fixed=TRUE),function(x) x[2]))
  
  # helper function to construct W projector matrix
  w.t<-function(x,lag=1){
    x.n<-c(rep(0,lag),rev(rev(x)[-c(1:lag)]))
    return(x.n)
  }
  
  # Calculate F-Statistic in a loop
  Fstat<-critL<-pL<-dofL<-list() # list objects since not for every country some nr. of regressors
  for(cc in 1:length(cN)){
    idx   <- grep(paste("^",cN[cc],".",sep=""),varNames) 
    X.dat <- xglobal[-c(1:pmax),idx,drop=FALSE]
    r.dat <- res[[cN[cc]]]
    ki    <- ncol(X.dat)
    dof   <- (bigT-ki-lag.cor)
    M     <- diag(bigT)-tcrossprod(X.dat%*%chol2inv(chol(crossprod(X.dat))),X.dat)
    # construct W matrix
    w.array <- array(0,dim=c(bigT,ki,lag.cor))
    faux<-critV<-pV<-NULL
    for(j in 1:ki){ # for all variables
      w<-NULL
      for(p in 1:lag.cor){
        w<-cbind(w,w.t(r.dat[,j],lag=p))
      }
      aux  <- bigT*((crossprod(r.dat[,j],w)%*%solve(t(w)%*%M%*%w)%*%(t(w)%*%r.dat[,j]))/crossprod(r.dat[,j]))
      faux <- c(faux,(dof/lag.cor)*(aux/(bigT-aux)))
      pV   <- c(pV,c(1-pf(aux,lag.cor,dof)))
    }
    names(faux) <- sapply(strsplit(colnames(X.dat),".",fixed=TRUE),function(x) x[[2]])
    Fstat[[cc]] <- faux
    critL[[cc]] <- critV <-c(critV,qf(alpha, lag.cor,dof))
    pL[[cc]]    <- pV
    dofL[[cc]]  <- dof
  }
  names(Fstat) <- names(pL) <- cN
  
  # Generate Output
  resTest<-array("-",dim=c(length(cN)*2,(length(vars)+3)))
  colnames(resTest)<-c("Country","DoF",paste("F-crit."," (",alpha,")",sep=""),vars)
  arrayIdx<-(1:(length(cN)*2))[c(TRUE, FALSE)]
  for(i in 1:length(arrayIdx)){
    resTest[arrayIdx[i],1:3]<-c(cN[i],paste("F(",lag.cor," ,",dofL[[i]],")",sep=""),format(round(critL[[i]],dig1)))
    resTest[arrayIdx[i],names(Fstat[[i]])]<-c(format(round(Fstat[[i]],dig1)))
    resTest[arrayIdx[i]+1,names(Fstat[[i]])]<-paste("(",format(round(pL[[i]],dig2)),")",sep="")
  }
  # Generate p-table
  pp<-unlist(pL);K<-length(pp)
  p.res<-matrix(0,nrow=4,ncol=2)
  p.res[1,]<-c(length(which(pp>0.10)),paste(round((length(which(pp>0.10))/K)*100,2),"%",sep=""))
  p.res[2,]<-c(length(which(pp>0.05& pp<=0.1)),paste(round((length(which(pp>0.05& pp<=0.1))/K)*100,2),"%",sep=""))
  p.res[3,]<-c(length(which(pp>0.01& pp<=0.05)),paste(round((length(which(pp>0.01& pp<=0.05))/K)*100,2),"%",sep=""))
  p.res[4,]<-c(length(which(pp<=0.01)),paste(round((length(which(pp<=0.01))/K)*100,2),"%",sep=""))
  rownames(p.res)<-c(">0.1","0.05-0.1","0.01-0.05","<0.01")
  colnames(p.res)<-c("# p-values", "in %")
  
  res<-list(Fstat=Fstat,resTest=resTest,p.res=p.res,pL=pL)
  return(res)
}