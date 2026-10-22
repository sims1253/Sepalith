makeRaster <- function(nr = 10, nc = 12) {
  r <- terra::rast(nrows = nr, ncols = nc, xmin = 0, xmax = nc * 10,
                   ymin = 0, ymax = nr * 10, crs = "EPSG:25832")
  terra::values(r) <- seq_len(nr * nc)
  r
}

test_that("tileScheme() covers every row and column exactly once", {
  r <- makeRaster()
  scheme <- tileScheme(r, tileRows = 4, tileCols = 5)
  expect_equal(nrow(scheme), 3 * 3)
  rows <- unique(scheme[, c("rowStart", "rowEnd")])
  cols <- unique(scheme[, c("colStart", "colEnd")])
  expect_equal(sum(rows$rowEnd - rows$rowStart + 1), 10)
  expect_equal(sum(cols$colEnd - cols$colStart + 1), 12)
  expect_false(anyDuplicated(scheme$tileId) > 0)
})

test_that("tileScheme() returns a single tile for small rasters", {
  scheme <- tileScheme(makeRaster(), tileRows = 100, tileCols = 100)
  expect_equal(nrow(scheme), 1)
  expect_equal(scheme$rowEnd, 10)
  expect_equal(scheme$colEnd, 12)
})

test_that("tileScheme() rejects non-positive tile sizes", {
  expect_error(tileScheme(makeRaster(), tileRows = 0), "must be positive")
  expect_error(tileScheme(makeRaster(), tileCols = -1), "must be positive")
})

test_that("tileExtent() matches cell boundaries", {
  r <- makeRaster()
  tile <- tileScheme(r, tileRows = 4, tileCols = 5)[1, ]
  e <- as.vector(tileExtent(r, tile))
  expect_equal(e, c(xmin = 0, xmax = 50, ymin = 60, ymax = 100))
})

test_that("tileExtent() clamps buffers at the raster edge", {
  r <- makeRaster()
  tile <- tileScheme(r, tileRows = 4, tileCols = 5)[1, ]
  e <- as.vector(tileExtent(r, tile, buffer = 2))
  expect_equal(e, c(xmin = 0, xmax = 70, ymin = 40, ymax = 100))
})

test_that("applyByTile() reproduces a cell-wise operation", {
  r <- makeRaster()
  out <- applyByTile(r, function(x) x * 2, tileRows = 4, tileCols = 5)
  expect_true(terra::compareGeom(out, r))
  expect_equal(terra::values(out, mat = FALSE), terra::values(r, mat = FALSE) * 2)
})

test_that("writeTiles() and mosaicTiles() round-trip a raster", {
  r <- makeRaster()
  dir <- withr::local_tempdir()
  paths <- writeTiles(r, dir, tileRows = 4, tileCols = 5)
  expect_length(paths, 9)
  expect_true(all(file.exists(paths)))
  back <- mosaicTiles(dir)
  expect_equal(terra::values(back, mat = FALSE), terra::values(r, mat = FALSE))
})

test_that("tileValidCounts() counts non-missing cells per tile", {
  r <- makeRaster()
  r[1:2, ] <- NA
  counts <- tileValidCounts(r, tileRows = 4, tileCols = 5)
  expect_equal(sum(counts$validCells), 10 * 12 - 2 * 12)
  expect_equal(counts$validCells[1], 2 * 5)
})
