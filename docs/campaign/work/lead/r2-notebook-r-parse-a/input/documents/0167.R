combineData<-function(eb1, eb2)
{
		if(missing(eb1)||missing(eb2))
		{
			stop("please specify the input data")
		}
		#let's do merge sort kind of combining.
		#first get the batch ids
		eb1.id<-names(eb1);
		eb2.id<-names(eb2);
		
		#sort them
		eb1.ids<-sort(eb1.id)
		eb2.ids<-sort(eb2.id)
		
		#now go through to combine then
		len1<-length(eb1.ids)
		len2<-length(eb2.ids)
		idx1<-1;
		idx2<-1;
		count<-0;
		#flag<-TRUE
		batch<-list();
		while(TRUE){
			#check for orders
			if(idx1>len1 || idx2>len2)
			{
				#flag<-FALSE;
				break;
			}
			count<-count+1;
			if(eb1.ids[idx1]==eb2.ids[idx2])
			{
				#combine batch together.
				batch[[count]]<-combineBatch(eb1[[eb1.ids[idx1] ]],eb2[[eb2.ids[idx2] ]]);
				idx1<-idx1+1;
				idx2<-idx2+1;
				#next;
			} else if(eb1.ids[idx1]>eb1.ids[idx2]){
				batch[[count]]<-eb1[[ eb1.ids[idx1] ]];
				batch[[count]]<-resetElisaBatchAnalysis(batch[[count]]);
				idx1<-idx1+1
			} else { #the case where eb1.ids[idx]<eb1.ids[idx2]
				batch[[count]]<-eb2[[ eb2.ids[idx2] ]];
				batch[[count]]<-resetElisaBatchAnalysis(batch[[count]]);
				idx2<-idx2+1
			}
		}#end of merge combine.
		
		#now need to copy over the left-behind
		if(idx1<=len1)
		{
			for(i in idx1:len1)
			{
				count<-count+1;
				batch[[count]]<-eb1[[ eb1.ids[i] ]]
			}
		}
		if(idx2<=len2)
		{
			for(i in idx2:len2)
			{
				count<-count+1;
				batch[[count]]<-eb2[[ eb2.ids[i] ]];
			}
		}
		return(batch)
}