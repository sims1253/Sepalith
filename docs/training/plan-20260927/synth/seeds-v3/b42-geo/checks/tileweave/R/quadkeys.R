x = c(3, 0, 550)
y = c(5, 1, 335)
z = c(3, 1, 10)
stopifnot(identical(encodeTileKey(3, 5, 3, "quadkey"), "213"))
stopifnot(identical(encodeTileKey(3, 5, 3, "path"), "3/3/5"))
stopifnot(identical(encodeTileKey(3, 5, 3, "tms_path"), "3/3/2"))
stopifnot(identical(encodeTileKey(3, 5, 3, "morton"), "3:39"))
stopifnot(identical(encodeTileKey(3, 5, 3, "hex"), "03000003000005"))
for (f in c("quadkey", "path", "tms_path", "morton", "hex")) {
  k = encodeTileKey(x, y, z, f)
  stopifnot(all(isTileKey(k, f)))
  d = decodeTileKey(k, f)
  stopifnot(identical(d$x, as.integer(x)), identical(d$y, as.integer(y)),
    identical(d$zoom, as.integer(z)))
}

stopifnot(identical(isTileKey(c("0123", "0124", NA), "quadkey"), c(TRUE, FALSE, FALSE)))
stopifnot(identical(isTileKey(c("3/1/2", "3-1-2"), "path"), c(TRUE, FALSE)))
stopifnot(!isTileKey("3/1", "tms_path"), !isTileKey("3:x", "morton"))
stopifnot(!isTileKey("03ZZ", "hex"))
stopifnot(identical(isTileKey("12", error = TRUE), "12"))
e = try(isTileKey("9", "quadkey", error = TRUE), silent = TRUE)
stopifnot(inherits(e, "try-error"), grepl("not a valid quadkey key", e))

stopifnot(identical(convertTileKey("213", "quadkey", "path"), "3/3/5"))
stopifnot(identical(convertTileKey("3/3/2", "tms", "quad"), "213"))
stopifnot(identical(convertTileKey("x", "hex", "hex"), "x"))
stopifnot(inherits(try(convertTileKey("bad", "path", "hex"), silent = TRUE), "try-error"))

dk = describeTileKeys(c("213", "1"))
stopifnot(identical(names(dk), c("key", "x", "y", "zoom", "parent", "quadkey", "path")))
stopifnot(identical(dk$parent, c("21", "")), identical(dk$path, c("3/3/5", "1/1/0")))
dp = describeTileKeys(c("0/0/0", "2/3/1"), "path")
stopifnot(is.na(dp$parent[1]), dp$parent[2] == "1/1/0", dp$quadkey[2] == "13")
