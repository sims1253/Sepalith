find_linked_edges <- function(tree) {
  # Create matrix of linked edges (all starting as unlinked, 0):
  edge_link_matrix <- matrix(
    0,
    nrow = nrow(tree$edge),
    ncol = nrow(tree$edge),
    dimnames = list(seq_len(nrow(tree$edge)), seq_len(nrow(tree$edge)))
  )

  # For each edge:
  for (i in seq_len(nrow(tree$edge))) {
    # Find linked edges:
    links <- setdiff(
      x = union(
        which(x = rowSums(tree$edge == tree$edge[i, 1]) == 1),
        which(x = rowSums(tree$edge == tree$edge[i, 2]) == 1)
      ),
      y = i
    )

    # Code 1 (linked) for linked edges in matrix:
    edge_link_matrix[i, links] <- edge_link_matrix[links, i] <- 1
  }

  # Return edge links matrix:
  edge_link_matrix
}