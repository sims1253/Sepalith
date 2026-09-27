data_preparation<-function(Outcomes, Control, Treatment=NULL){

  predictions<-c(unlist(Control),unlist(Treatment))
  if( any(predictions<= 0 | predictions>= 1) ){stop('All predictions should be strictly between 0 and 1.')}
  if(!all(Outcomes %in% 0:1)){stop('All outcomes should be binary.')}

  N=length(Outcomes)

  dd<-list()
  for(n in 1:N){
    dd[[n]] <- list(outcome = Outcomes[[n]],
                    control_probits = qnorm(Control[[n]]),
                    treatment_probits = qnorm(as.numeric(Treatment[[n]]))
                    )
  }
  data = lapply(list(dd), summariser)
  return(data)
}