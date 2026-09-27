CPoptim<-function(rFunction, lower, upper,maxFE,sampleSize)
{
	if( missing(rFunction) || missing(lower) ||		missing(upper) )
		return(NULL)

  forestSize=1
  minimalX<-lower
  maximalX<-upper

	if(!inherits(rFunction,"function"))
		return (NULL)
	if(!inherits(rFunction(c(minimalX+maximalX)/2),"numeric"))
		return (NULL)

	dimension<-length(minimalX)
	argInt<-integer(4)
	argInt[1]<-dimension
	if(missing(sampleSize))
		sampleSize<-1000
	argInt[2]<-sampleSize
	if(missing(maxFE))
		maxFE<-argInt[1]*5000
	argInt[3]<-maxFE
	argInt[4]<-forestSize

	minimalSampleInAGroup=10
	partitions=maxFE/minimalSampleInAGroup+1;

	aPriori=double(partitions)
  minCondP=double(partitions)
  hatMu=double(partitions)
  hatSD=double(partitions)
  minX=double(partitions*dimension)
  maxX=double(partitions*dimension)
  X=double(maxFE*dimension)
  Y=double(maxFE)
  ID=integer(maxFE)
  N=integer(partitions)

	#dyn.load("CPoptim.so")
	obj<-as.numeric(.Call("CPoptim",as.integer(argInt),
		as.double(minimalX),as.double(maximalX),
		rFunction,environment(),aPriori,minCondP,
		hatMu,hatSD,minX,maxX,X,Y,ID,N,PACKAGE = "CPoptim"))
	curPart<-obj[1]
	#dyn.unload("CPoptim.so")


	post<-minCondP[1:curPart]*aPriori[1:curPart]

	subsets<-list(	aPriori=aPriori[1:curPart], aPosteriori= post/sum(post),
					mean=hatMu[1:curPart],
					std=hatSD[1:curPart],
				  lowerBound=matrix(minX,byrow=TRUE,ncol=dimension),
				  upperBound=matrix(maxX,byrow=TRUE,ncol=dimension) )

	sample<-list(	x=matrix(X,byrow=TRUE,nrow=maxFE),
					y=matrix(Y,byrow=TRUE,nrow=maxFE),
					subsetID=1+ID)


	return(list(subsets=subsets,sample=sample))
}