actual_score = function(graph, initial.score, damping = 0.7){

  if (length(initial.score)==0){
    stop("initial.score is an empty vector")
  }

  actual_score = igraph::page_rank(
      graph,
      directed = FALSE,
      damping = damping,
      personalized = initial.score)$vector

  return(actual_score)
}