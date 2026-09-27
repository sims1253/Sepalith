mle.nu <- function(p,J,mah,mah1,mah2,df,k){
  #p is the probability matrix, k is the cluster, J is the # of params, mah is the mahalanobis
  df=as.numeric(df)
  p2=sum(p^2)
  #	val <- (uniroot.all(function( z ) ((-digamma((z+J)/2) + digamma(z/2) +.5*J/z)*p2+ (.5*mah1-.5*((z+J)/z)*mah2)),lower=2,upper=200,maxiter=30))
  val <- (uniroot.all(function( z ) (1+(digamma((z+J)/2) - digamma(z/2) )*2+log(z)- sum(p^2*(log(df+mah)))/p2-(z+J)*sum(p^2*(df+mah)^(-1))/p2),lower=2,upper=200,maxiter=10))
  
  
  if(length(val)!=1)val=val[1]
  #if(!is.numeric(val))val=df
  #	if(val > 200) val <- 200
  #	if(val < 2) val <- 2
  return(val)
}