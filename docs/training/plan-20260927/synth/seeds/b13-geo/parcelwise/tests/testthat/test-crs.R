make_points <- function(crs) {
  pts <- sf::st_sfc(sf::st_point(c(9.18, 48.78)), sf::st_point(c(9.20, 48.80)),
                    crs = 4326)
  if (!is.na(crs) && crs != 4326) {
    pts <- sf::st_transform(pts, crs)
  }
  if (is.na(crs)) {
    sf::st_crs(pts) <- NA
  }
  sf::st_sf(id = 1:2, geometry = pts)
}

test_that("has_crs() detects missing coordinate reference systems", {
  expect_true(has_crs(make_points(4326)))
  expect_true(has_crs(make_points(25832)))
  expect_false(has_crs(make_points(NA)))
})

test_that("is_geographic() and is_projected() are mutually exclusive", {
  geo <- make_points(4326)
  utm <- make_points(25832)
  expect_true(is_geographic(geo))
  expect_false(is_projected(geo))
  expect_true(is_projected(utm))
  expect_false(is_geographic(utm))
})

test_that("assert_projected() rejects geographic and missing CRS", {
  expect_error(assert_projected(make_points(4326), "parcels"), "`parcels` must use")
  expect_error(assert_projected(make_points(NA), "parcels"), "no coordinate reference")
  expect_invisible(assert_projected(make_points(25832)))
})

test_that("assert_same_crs() reports both CRS labels on mismatch", {
  geo <- make_points(4326)
  utm <- make_points(25832)
  expect_error(assert_same_crs(geo, utm), "EPSG:4326 vs EPSG:25832")
  expect_invisible(assert_same_crs(utm, utm))
  expect_error(assert_same_crs(geo, make_points(NA)), "Both layers")
})

test_that("crs_label() falls back for layers without a CRS", {
  expect_equal(crs_label(make_points(4326)), "EPSG:4326")
  expect_equal(crs_label(make_points(NA)), "<no CRS>")
})

test_that("utm_epsg_for() picks the zone of the layer centroid", {
  expect_equal(utm_epsg_for(make_points(4326)), 32632L)
  south <- sf::st_sf(id = 1, geometry = sf::st_sfc(sf::st_point(c(36.8, -1.3)), crs = 4326))
  expect_equal(utm_epsg_for(south), 32737L)
})

test_that("ensure_projected() leaves projected layers alone", {
  utm <- make_points(25832)
  expect_identical(ensure_projected(utm), utm)
  expect_no_message(ensure_projected(utm))
})

test_that("ensure_projected() transforms geographic layers to UTM", {
  geo <- make_points(4326)
  expect_message(out <- ensure_projected(geo), "to EPSG:32632")
  expect_true(is_projected(out))
  expect_equal(sf::st_crs(out)$epsg, 32632L)
  expect_no_message(ensure_projected(geo, target = 25832, quiet = TRUE))
  quiet <- ensure_projected(geo, target = 25832, quiet = TRUE)
  expect_equal(sf::st_crs(quiet)$epsg, 25832L)
})
