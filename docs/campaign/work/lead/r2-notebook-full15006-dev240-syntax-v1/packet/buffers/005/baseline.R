
contour3d <- function(
  griddata,
  level,
  x,
  y,
  z
) {
  # Run the marching cubes algorithm
  result <- marching_cubes(
    data = griddata,
    x = x,
    y = y,
    z = z,
    iso = level
  )

  # Set the vectors and coordinates
  list(
    triangles = result$triangles + 1,
    vertices = result$vertices,
    normals = result$normals
  )
}