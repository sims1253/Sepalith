ccf_ts <- 
function(bints1, bints2, lagmax=NULL, ylima=-1, ylimb=1, rmltrd="N", RedL=T,  
         device="screen", Hfig, Wfig, Hpdf, Wpdf, resfig, ofilename) { 
 
 #:: Checking the input data 
 if( dim(bints1)[2] !=2 | dim(bints2)[2] != 2) 
   stop ("There is a problem with the input data. The input data should 
    be a couple of vectors of dimension N x 2 (rows x columns). Please, 
    use in the R's command line: dim(ts'x')[2] to verify the number of 
    columns. Thank you for using our BINCOR package.") 

 if( dim(bints1)[1] != dim(bints2)[1] ) 
  stop ("The binned time series under analysis do not have the same 
   number of elements. Thank you for using our BINCOR package.") 

 #:: Devices options: png, jpg & pdf! 
 if (device=="png") {
  fileout <- file.path(
    tempdir(),
    paste0("ccf_", ofilename, ".png")
  )
  png(fileout, height=Hfig, width=Wfig, res=resfig) 
 }

 if (device=="jpeg" || device=="jpg") {
  fileout <- file.path(
    tempdir(),
    paste0("ccf_", ofilename, ".jpg")
  )
  jpeg(fileout, height=Hfig, width=Wfig, res=resfig)
 }

 if (device=="pdf") {
  fileout <- file.path(
    tempdir(),
    paste0("ccf_", ofilename, ".pdf")
  ) 
  pdf(fileout, height=Hpdf, width=Wpdf)
 }

 if (rmltrd == "N" || rmltrd == "n") 
 ccf.12 <- ccf(bints1[,2], bints2[,2], main="",   
  	      ylab="Corr. coef.", lag.max=lagmax, ylim=c(ylima,ylimb), las=1)
 if (rmltrd == "Y" || rmltrd == "y") 
 #:: The linear trend is removed -the R pack. "pracma" is required!
 ccf.12 <- ccf(c(detrend(bints1[,2])), c(detrend(bints2[,2])), main="",   
  	      ylab="Corr. coef.", lag.max=lagmax, ylim=c(ylima,ylimb), las=1)

 if (RedL == TRUE | RedL == T) { 
  idmidp <- which(ccf.12$lag == 0)
  abline(h=ccf.12$acf[idmidp], col="red") 
 } 


 if (device != "screen") 

 dev.off()

 return(ccf.12) 

}