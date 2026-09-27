myTable <- function(x) { 
myDF <- data.frame( table(x) ) 
myDF$Prop <- prop.table( myDF$Freq ) 
myDF$CumProp <- cumsum( myDF$Prop ) 
myDF 
}