# Prepare district, settlement and clinic layers for the access analysis.
library(sf)
library(dplyr)
library(readr)

raw_dir <- "data/raw"
out_dir <- "data/derived"
target_crs <- 32736

read_districts <- function(path) {
  st_read(path, quiet = TRUE) |>
    rename_with(tolower) |>
    select(district_id = adm2_pcode, district = adm2_en, region = adm1_en) |>
    st_make_valid() |>
    st_transform(target_crs)
}

read_settlements <- function(path) {
  st_read(path, quiet = TRUE) |>
    rename_with(tolower) |>
    filter(!is.na(population), population > 0) |>
    select(settlement_id = osm_id, name, population) |>
    st_transform(target_crs)
}

read_clinics <- function(path) {
  read_csv(path, show_col_types = FALSE) |>
    filter(!is.na(longitude), !is.na(latitude)) |>
    mutate(level = factor(level, levels = c("dispensary", "health_centre", "hospital"))) |>
    st_as_sf(coords = c("longitude", "latitude"), crs = 4326) |>
    st_transform(target_crs)
}

drop_outside <- function(points, districts, label) {
  inside <- lengths(st_intersects(points, districts)) > 0
  if (any(!inside)) {
    message(label, ": dropping ", sum(!inside), " feature(s) outside the study area")
  }
  points[inside, ]
}

dedupe_clinics <- function(clinics, tolerance = 50) {
  near <- st_is_within_distance(clinics, clinics, dist = tolerance)
  keep <- rep(TRUE, nrow(clinics))
  for (i in seq_len(nrow(clinics))) {
    if (!keep[i]) {
      next
    }
    dupes <- setdiff(near[[i]], i)
    keep[dupes[dupes > i]] <- FALSE
  }
  message("Removed ", sum(!keep), " duplicate clinic(s)")
  clinics[keep, ]
}

attach_district <- function(points, districts) {
  points |>
    st_join(select(districts, district_id), join = st_within, left = TRUE) |>
    filter(!is.na(district_id))
}

layer_report <- function(...) {
  layers <- list(...)
  tibble(
    layer = names(layers),
    features = vapply(layers, nrow, integer(1)),
    crs = vapply(layers, function(x) st_crs(x)$epsg, integer(1))
  )
}

districts <- read_districts(file.path(raw_dir, "admin2.gpkg"))
settlements <- read_settlements(file.path(raw_dir, "settlements.geojson")) |>
  drop_outside(districts, "settlements") |>
  attach_district(districts)
clinics <- read_clinics(file.path(raw_dir, "facility_register.csv")) |>
  drop_outside(districts, "clinics") |>
  dedupe_clinics(tolerance = 50) |>
  attach_district(districts)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
st_write(districts, file.path(out_dir, "layers.gpkg"), layer = "districts",
         delete_layer = TRUE, quiet = TRUE)
st_write(settlements, file.path(out_dir, "layers.gpkg"), layer = "settlements",
         delete_layer = TRUE, quiet = TRUE)
st_write(clinics, file.path(out_dir, "layers.gpkg"), layer = "clinics",
         delete_layer = TRUE, quiet = TRUE)

print(layer_report(districts = districts, settlements = settlements, clinics = clinics))
