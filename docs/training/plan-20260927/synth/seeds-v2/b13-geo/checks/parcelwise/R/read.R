# skipped: needs sf (read_layer_gpkg, read_layer_shp, read_layer_geojson,
#   read_layer_csv, read_layer, finalize_layer, list_gpkg_layers)

# detect_layer_format()
stopifnot(
  identical(detect_layer_format("parcels.GPKG"), "gpkg"),
  identical(detect_layer_format("a/b/zones.json"), "geojson"),
  identical(detect_layer_format("sites.txt"), "csv"),
  inherits(try(detect_layer_format("raster.tif"), silent = TRUE), "try-error")
)

# check_layer_path()
tmp <- tempfile(fileext = ".csv")
writeLines("lon,lat", tmp)
stopifnot(
  identical(check_layer_path(tmp, c("csv", "txt")), tmp),
  inherits(try(check_layer_path(tmp, "gpkg"), silent = TRUE), "try-error"),
  inherits(try(check_layer_path(c(tmp, tmp), "csv"), silent = TRUE), "try-error"),
  inherits(try(check_layer_path(NA_character_, "csv"), silent = TRUE), "try-error")
)
unlink(tmp)
