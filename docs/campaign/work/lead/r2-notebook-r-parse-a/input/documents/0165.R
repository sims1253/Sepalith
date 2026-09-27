predictBatchData<-function(batch)
{
	if(missing(batch))
	{
		stop("ERROR:please specify the input batch data");
	}
	if(!is(batch,"elisa_batch"))
	{
		stop("ERROR:please specify the data as elisa_batch");
	}
	
	#now get the model
	pars<-batch@pars
	batchNormFac<-batch@normFactor;
	
	#predict.mean<-list();
	#now go through each 
	count<-0;
	for(i in 1:batch@num.runs)
	{
		for(j in 1:batch@runs[[i]]@num.plates)
		{
			count<-count+1;
			#get the model
			pars.plate<-pars+c(0,0,-1*batch@runs[[i]]@plates[[j]]@normFactor,0,0)
			
			#for doing unknows
			if(!is.null(batch@runs[[i]]@plates[[j]]@data.unknown)&& dim(batch@runs[[i]]@plates[[j]]@data.unknown)[1]!=0){
				x.expect<-inv.f5pl(pars.plate, batch@runs[[i]]@plates[[j]]@data.unknown$OD);
				batch@runs[[i]]@plates[[j]]@data.unknown$conc_pred<-x.expect;
				batch@runs[[i]]@plates[[j]]@data.unknown$conc_pred.bc<-x.expect*exp(batchNormFac);
				
				#doing 
				dF<-aggregate(batch@runs[[i]]@plates[[j]]@data.unknown$OD, 
						by=list(group=batch@runs[[i]]@plates[[j]]@data.unknown$ID), FUN=mean);
				
				x.expect<-inv.f5pl(pars.plate, dF$x);
				dF$conc_pred<-x.expect
				dF$conc_pred.bc<-x.expect*exp(batchNormFac);
				dF<-dF[,c("group","x","conc_pred","conc_pred.bc")]
				names(dF)<-c("ID","OD","conc_pred","conc_pred.bc")
				batch@runs[[i]]@plates[[j]]@mdata.unknown<-dF;
			}
			#--now add the predications for standard table
			stdx.expect<-inv.f5pl(pars.plate, batch@runs[[i]]@plates[[j]]@data.std$OD);
			batch@runs[[i]]@plates[[j]]@data.std$conc_pred<-stdx.expect;
			batch@runs[[i]]@plates[[j]]@data.std$conc_pred.bc<-stdx.expect*exp(batchNormFac);
			dF<-aggregate(batch@runs[[i]]@plates[[j]]@data.std$OD, 
					by=list(group=batch@runs[[i]]@plates[[j]]@data.std$ID), FUN=mean);
			dF.conc<-aggregate(batch@runs[[i]]@plates[[j]]@data.std$conc, 
					by=list(group=batch@runs[[i]]@plates[[j]]@data.std$ID), FUN=mean);
			names(dF.conc)<-c("ID","conc");
			stdx.expect<-inv.f5pl(pars.plate, dF$x);
			dF$conc_pred<-stdx.expect
			dF$conc_pred.bc<-stdx.expect*exp(batchNormFac);
			dF<-dF[,c("group","x","conc_pred","conc_pred.bc")]
			names(dF)<-c("ID","OD","conc_pred","conc_pred.bc")
			dF.conc<-cbind(dF.conc,dF[,c("OD","conc_pred","conc_pred.bc")]);
			batch@runs[[i]]@plates[[j]]@mdata.std<-dF.conc;
			
		}
	}
	return (batch)
}