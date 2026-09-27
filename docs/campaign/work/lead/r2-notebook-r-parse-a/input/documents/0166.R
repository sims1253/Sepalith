predictAll<-function(batches)
{
	if(missing(batches))
	{
		stop("ERROR:please specify the input batch data");
	}
	if(!is.list(batches))
	{
		stop("ERROR:please specify the data as a list of elisa_batch object");
	}
	
	#now go through the data to get everything done
	
	for(i in 1:length(batches))
	{
		batch<-batches[[i]]
		batches[[i]]<-predictBatchData(batch)
	}
	return(batches)
}