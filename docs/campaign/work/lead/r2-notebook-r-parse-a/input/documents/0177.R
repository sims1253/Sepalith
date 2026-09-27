writeTable <- function(results_write,num_cores,outputDir,iteration) {
  out <- tryCatch(
    {
      suppressMessages(fwrite(results_write, paste(outputDir,"CoNIOutputSplit",iteration,".csv",sep=""),nThread=num_cores))
    },
    error=function(cond) {
      message('fwrite failed')
      message("Here's the original error message:")
      message(cond)
      # Choose a return value in case of error
      return('NA')
    }
  )
  return(out)
}