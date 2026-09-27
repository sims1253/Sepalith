ObsNumbs <- function(ID, TS, BY = c("P", "D", "PD")) {
  if (BY == "P") {
    ObsNumb_out <- ave(TS, ID, FUN = seq_along)
  }
  if (BY == "D") {
    PromptDate <- anytime::anydate(TS)
    ObsNumb_out <- ave(as.character(PromptDate), ID, FUN = function(x) {
      as.numeric(factor(x))
    })
  }
  if (BY == "PD") {
    PromptDate <- anytime::anydate(TS)
    ObsNumb_out <- ave(TS, paste(ID, PromptDate), FUN = seq_along)
  }
  return(ObsNumb_out)
}