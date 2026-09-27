TPDC<- function(data=NULL,  k=2, ini="kmedoids",nr=5,iter=100,epsilon=0.001) {
  # Cluster the data whit pd-clustering Algoritmh
  #
  #
  #%%%%%%INPUT%%%%%%%%%
  #data=input data
  #k=number of cluster
  #
  #
  #%%%%%%OUTPUT%%%%%%%%
  #cnew=cluster's center  
  #l=class label
  #p=nxk matrix
  #probability to belong to each class 
  #JDF  join distance function
  #cont=number of iterations until convergence
  method=ini
  if((!is.double(k))&(!is.integer(k))){stop("The number of clusters (k) must take an integer value.")}
  if(k<2){stop("The number of clusters (k) must be greater than one.");}
  if((k-round(k)!=0)){stop("The number of clusters (k) must be a whole number.");}
  data=as.matrix(data)
  if(!is.double(data)){stop("All elements of data must have type double.");}
  n=nrow(data)
  J=ncol(data)
  temp.center<-list()
  JDFini=matrix(0,5,1)
  s=list()
  for(i in 1:k){
    s[[i]]=cov(data)#diag(J)
  }
  if(method=="random"){
    for(t in 1:nr){
      x<-vector()
      for(i in 1:J)
      {
        x<-c(x,runif(k,min(data[,i]), max(data[,i])))
      }
      center<-matrix(x,k,J)
      temp.center[[t]]<-center
      update=corePDT(data,k,center,s,n=nrow(data),J=ncol(data),iter=4)
      JDFini[t]=update$JDF}
    center= temp.center[[which(JDFini==min(JDFini))[1]]]}
  else if(method=="PDclust"){
    ini=PDC(data,k)
    center=ini$centers
    l=ini$label
  }
  else{center=pam(data,k)$medoids}
  cnew=center
  update=corePDT(data,k,cnew,s,n=nrow(data),J=ncol(data),iter=iter,epsilon=epsilon)
  #check classification
  class<-apply(update$probability,1,which.max)
  #output 
  out<-list(label=class,centers=update$centers,sigma=update$sigma,df=update$df,probability=update$probability,JDF=update$JDF, iter=update$cont,data=data)
  class(out) <- "FPDclustering"
  return(out)
}