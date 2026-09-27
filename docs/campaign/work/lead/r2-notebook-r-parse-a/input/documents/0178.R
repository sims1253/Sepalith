split_df<-function(AbundantDF,numberSplitDF_2=2){
  dt_list<-list()
  if(ncol(AbundantDF) %% numberSplitDF_2 !=0){
    SplitFirstParts<-floor(ncol(AbundantDF)/numberSplitDF_2)
    start<-1 #Start to slice is position 1 of data frame
    end<-SplitFirstParts #
    i=1
    while(end <= SplitFirstParts*numberSplitDF_2){
      dt_list[[i]]<-AbundantDF[,start:end]
      start<-start+SplitFirstParts
      end<-start+SplitFirstParts-1
      i<-i+1
    }
    start<-SplitFirstParts*numberSplitDF_2+1
    dt_list[[i]]<-AbundantDF[,start:ncol(AbundantDF),drop=FALSE] #Avoid losing column name when is a single column that is left
  }else{
    split_size<-ncol(AbundantDF)/numberSplitDF_2
    start<-1 #Start to slice is position 1 of data frame
    end<-split_size
    i=1
    while(end <= ncol(AbundantDF)){
      dt_list[[i]]<-AbundantDF[,start:end]
      start<-start+split_size
      end<-start+split_size-1
      i<-i+1
    }
  }
  dt_list
}