matrix_to_list <- function(datamat){
  if(any(is.na(datamat))){
    stop("The data you have submitted contains NAs. Please check the data.")
  }
  if(!all(grepl("\\.",colnames(datamat)))){
    stop("Please seperate country- and variable names with a point.")
  }
  cN <- unique(unlist(lapply(strsplit(colnames(datamat),".",fixed=TRUE),function(l) l[1])))
  N  <- length(cN)
  if(!all(nchar(cN)>1)){
    stop("Please provide entity names with minimal two characters.")
  }
  datalist <- list()
  for(cc in 1:N){
    datalist[[cN[cc]]] <- datamat[,grepl(cN[cc],colnames(datamat)),drop=FALSE]
    colnames(datalist[[cN[cc]]]) <- unlist(lapply(strsplit(colnames(datalist[[cN[cc]]]),".",fixed=TRUE),function(l)l[2]))
  }
  return(datalist)
}