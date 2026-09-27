avg.pair.cc=function(object, digits=3){
  if(inherits(object, "bgvar")){
    lags <- object$args$lags
    pmax <- max(lags)
    dat  <- object$xglobal[-c(1:pmax),]
    res  <- do.call("cbind",object$cc.results$res)
    res  <- res[,colnames(dat)] 
  }
  if(inherits(object, "bgvar.resid")){
    dat    <- object$Data
    res    <- apply(object$country,c(2,3),mean)  
    res.g  <- apply(object$global,c(2,3),mean) 
  }
  bigT     <- nrow(res)
  varNames <- colnames(dat)
  cN   <- unique(sapply(strsplit(varNames,".",fixed=TRUE),function(x) x[1]))
  vars <- unique(sapply(strsplit(varNames,".",fixed=TRUE),function(x) x[2]))
  
  idx<-lapply(as.list(1:length(vars)),function(x) grep(paste(".",vars[x],sep=""),colnames(dat)))
  names(idx)<-vars
  # kick out exo variables
  exo<-which(sapply(idx,length)==1)
  if(length(exo)>0){
    idx<-idx[-exo]
  }
  
  datL<-resL<-resL.g<-matrix("-",nrow=length(cN),ncol=length(idx))
  rownames(datL)<-rownames(resL)<-rownames(resL.g)<-cN
  colnames(datL)<-colnames(resL)<-colnames(resL.g)<-names(idx)
  
  if(inherits(object,"bgvar")){
    for(i in 1:length(idx)){
      aux.dat <- cor(dat[,idx[[i]]])
      aux.res <- cor(res[,idx[[i]]])
      diag(aux.dat)<-diag(aux.res)<-NA
      ii <- sapply(strsplit(rownames(aux.dat),".",fixed=TRUE),function(x) x[1]) 
      aux.dat <- round(rowMeans(aux.dat,na.rm=TRUE),digits=digits)
      aux.res <-round(rowMeans(aux.res,na.rm=TRUE),digits=digits)
      datL[ii,i]<-aux.dat
      resL[ii,i]<-aux.res
    }
  }
  if(inherits(object,"bgvar.resid")){  # include analysis based on residuals of the global model as well
    for(i in 1:length(idx)){
      aux.dat <- cor(dat[,idx[[i]]])
      aux.res <- cor(res[,idx[[i]]])
      aux.rg  <- cor(res.g[,idx[[i]]])
      diag(aux.dat)<-diag(aux.res)<-diag(aux.rg)<-NA
      ii <- sapply(strsplit(rownames(aux.dat),".",fixed=TRUE),function(x) x[1]) #should be the same for u
      aux.dat <- round(rowMeans(aux.dat,na.rm=TRUE),digits=digits)
      aux.res <- round(rowMeans(aux.res,na.rm=TRUE),digits=digits)
      aux.rg  <- round(rowMeans(aux.rg,na.rm=TRUE),digits=digits)
      datL[ii,i]   <- aux.dat
      resL[ii,i]   <- aux.res
      resL.g[ii,i] <- aux.rg
    }
  }
  
  # Generate summary-table
  pp<-suppressWarnings(apply(datL,2,as.numeric))
  rr<-suppressWarnings(apply(resL,2,as.numeric))
  rg<-suppressWarnings(apply(resL.g,2,as.numeric))
  dat.res<-res.res<-res.resG<-matrix(0,nrow=4,ncol=ncol(datL))
  for(i in 1:ncol(datL)){
    aux<-pp[,i];aux<-abs(aux[which(!is.na(aux))]);K<-length(aux)
    aux2<-rr[,i];aux2<-abs(aux2[which(!is.na(aux2))]);K2<-length(aux2)
    aux3<-rg[,i];aux3<-abs(aux3[which(!is.na(aux3))]);K3<-length(aux3)
    
    dat.res[1,i]<-paste(length(which(aux<=0.1))," (",round((length(which(aux<=0.1))/K)*100,2),"%)",sep="")
    res.res[1,i]<-paste(length(which(aux2<=0.1))," (",round((length(which(aux2<=0.1))/K2)*100,2),"%)",sep="")
    temp<-round((length(which(aux3<=0.1))/K3)*100,2)
    res.resG[1,i]<-paste(length(which(aux3<=0.1))," (",ifelse(is.nan(temp),0,1),"%)",sep="")
    
    dat.res[2,i]<-paste(length(which(aux>0.1&aux<=0.2))," (",round((length(which(aux>0.1&aux<=0.2))/K)*100,2),"%)",sep="")
    res.res[2,i]<-paste(length(which(aux2>0.1&aux2<=0.2))," (",round((length(which(aux2>0.1&aux2<=0.2))/K2)*100,2),"%)",sep="")
    temp<-round((length(which(aux3>0.1&aux3<=0.2))/K3)*100,2)
    res.resG[2,i]<-paste(length(which(aux3>0.1&aux2<=0.2))," (",ifelse(is.nan(temp),0,1),"%)",sep="")
    
    dat.res[3,i]<-paste(length(which(aux>0.2&aux<=0.5))," (",round((length(which(aux>0.2&aux<=0.5))/K)*100,2),"%)",sep="")
    res.res[3,i]<-paste(length(which(aux2>0.2&aux2<=0.5))," (",round((length(which(aux2>0.2&aux2<=0.5))/K2)*100,2),"%)",sep="")
    temp<-round((length(which(aux3>0.2&aux3<=0.5))/K3)*100,2)
    res.resG[3,i]<-paste(length(which(aux3>0.2&aux3<=0.5))," (",ifelse(is.nan(temp),0,temp),"%)",sep="")
    
    
    dat.res[4,i]<-paste(length(which(aux>0.5&aux<=1))," (",round((length(which(aux>0.5&aux<=1))/K)*100,2),"%)",sep="")
    res.res[4,i]<-paste(length(which(aux2>0.5&aux2<=1))," (",round((length(which(aux2>0.5&aux2<=1))/K2)*100,2),"%)",sep="")
    temp<-round((length(which(aux3>0.5&aux3<=1))/K3)*100,2)
    res.resG[4,i]<-paste(length(which(aux3>0.5&aux3<=1))," (",ifelse(is.nan(temp),0,temp),"%)",sep="")
    
  }
  colnames(dat.res) <- colnames(res.res) <- colnames(res.resG) <- colnames(datL)
  rownames(dat.res) <- rownames(res.res) <- rownames(res.resG) <- c("<0.1","0.1-0.2","0.2-0.5",">0.5")
  #dat.res  <- rbind(c("",colnames(datL)),cbind(c("<0.1","0.1-0.2","0.2-0.5",">0.5"),dat.res))
  #res.res  <- rbind(c("",colnames(datL)),cbind(c("<0.1","0.1-0.2","0.2-0.5",">0.5"),res.res))
  #res.resG <- rbind(c("",colnames(datL)),cbind(c("<0.1","0.1-0.2","0.2-0.5",">0.5"),res.resG))
  
  avg.cc<-list(data.cor=datL,resid.cor=resL,resid.corG=resL.g,dat.res=dat.res,res.res=res.res,res.resG=res.resG)
  return(avg.cc)
}