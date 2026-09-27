mod_random_clust_sim <- function (x, p) {
  # A: make percolated raster, where proportion of filled cells = p_cluster
  x_sim <- terra::rast(nrows = nrow(x), ncols = ncol(x), extent = terra::ext(x), crs = terra::crs(x))
  for (i in terra::cells(x)) {
    if (stats::runif(1) <= p) {x_sim[i] = 1} else {x_sim[i] = 0}
  }

  # B: find clusters in raster
  x_sim <- terra::patches(x_sim, zeroAsNA = TRUE)

  # C: assign clusters to categories
  prop <- terra::freq(x, digits = 2) %>% # proportions of cells in category
    as.data.frame %>%
    tibble::add_column(., p = .$count/sum(.$count)) %>%
    .[order(.$p),c(2, 4)]
  clump_selection_order <- sample(terra::unique(x_sim)[-1,])
  areas <- terra::cellSize(x_sim)
  total_area <- terra::expanse(x_sim)
  
  row = 1
  for (i in clump_selection_order) {
    assignment = prop[row, 1]; max_prop = prop[row, 2]
    terra::values(x_sim)[terra::values(x_sim) == i] <- assignment
    
    current_prop <- which(terra::values(x_sim) == assignment) %>%
      areas[.] %>%
      sum(.)/total_area %>%
      .[,2]
    if (current_prop >= max_prop) {row = row + 1}
    
    if (row > nrow(prop)) {break}
  }
  
  # D: fill in the rest of the cells
  for (i in setdiff(terra::cells(x), terra::cells(x_sim))) {
    choices <- terra::adjacent(x_sim, i) %>%
      as.vector %>%
      x_sim[.] %>%
      subset(patches != 0) %>%
      table %>%
      as.data.frame

    if (length(choices != 0)) {
      choices <- as.vector(choices[choices$Freq == max(choices$Freq), 1])
      if (length(choices > 1)) {x_sim[i] = as.numeric(sample(choices, 1))} else {x_sim[1] = as.numeric(choices)}
    } else {
      x_sim[i] <- sample(prop[,1], 1, prob = prop[,2])
    }
  }
  
  return(x_sim)
  
}