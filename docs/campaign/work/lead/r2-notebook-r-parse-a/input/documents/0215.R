graph_extract_components <- function( x, directed = TRUE, bipartite_proj=FALSE, num_proj=1){

  if(!("igraph" %in% class(x)|| "network" %in% class(x))) stop("The input is not an igraph or a
                                                           network object")

  if(is_igraph(x)){

    if( bipartite_proj){

      if(is_bipartite(x)){

        x<-bipartite.projection(x)[[num_proj]]

        if (!is_simple(x))   x<-simplify(x)

        cl <- clusters(x)

        graph.splitting <- function(k, x, cl){
          induced.subgraph(x, cl$membership == k)
        }

        components<-sapply(1:max(cl$membership), graph.splitting, x = x, cl = cl, simplify = FALSE)

      }

    }

    else{

      if (!is_simple(x))   x<-simplify(x)

      cl <- clusters(x)

      graph_splitting <- function(k, x, cl){
        induced.subgraph(x, cl$membership == k)
      }

      components<-sapply(1:max(cl$membership), graph_splitting, x = x, cl = cl, simplify = FALSE)

    }

  }

  if( is.network(x)){

    edgelist<-as.edgelist(x)

    x<-graph_from_edgelist(edgelist, directed = TRUE)

    if (!is_simple(x))  x<-simplify(x)

    cl <- clusters(x)

    graph_splitting <- function(k, x, cl){
      induced.subgraph(x, cl$membership == k)
    }

    components<-sapply(1:max(cl$membership), graph_splitting, x = x, cl = cl, simplify = FALSE)

  }

  return(components)

}