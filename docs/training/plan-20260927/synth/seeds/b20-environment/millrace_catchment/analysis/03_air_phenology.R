# Low-cost PM sensor calibration, daily air-quality index, and growing
# degree-day phenology for the orchard plots near the mill station.

library(data.table)
library(aerocal)

sensor_path = "data/raw/pm_sensors_hourly.csv"
reference_path = "data/raw/pm_reference_hourly.csv"
stations_path = "data/clean/stations_daily.csv"

apple_stages = c(
  green_tip = 80,
  tight_cluster = 160,
  pink = 230,
  full_bloom = 300,
  petal_fall = 380
)

loadCollocation = function(sensorPath, referencePath) {
  sensors = fread(sensorPath, na.strings = c("", "NA", "-999"))
  reference = fread(referencePath, na.strings = c("", "NA", "-999"))
  sensors[, time := as.POSIXct(time, tz = "UTC")]
  reference[, time := as.POSIXct(time, tz = "UTC")]
  setnames(reference, "pm25", "pm25_ref")
  merged = merge(sensors, reference[, .(time, pm25_ref)], by = "time", all.x = TRUE)
  setorder(merged, sensor_id, time)
  merged
}

fitSensorModels = function(dt, trainEnd) {
  ids = unique(dt$sensor_id)
  models = vector("list", length(ids))
  names(models) = ids
  for (id in ids) {
    train = dt[sensor_id == id & time < trainEnd & !is.na(pm25_ref)]
    lin = tryCatch(
      fitLinearCal(train$pm25_raw, train$pm25_ref),
      error = function(e) NULL
    )
    hum = tryCatch(
      fitHumidityCal(train$pm25_raw, train$pm25_ref, train$rh),
      error = function(e) NULL
    )
    models[[id]] = list(linear = lin, humidity = hum)
  }
  models
}

scoreSensorModels = function(dt, models, trainEnd) {
  rows = list()
  for (id in names(models)) {
    test = dt[sensor_id == id & time >= trainEnd & !is.na(pm25_ref)]
    for (type in names(models[[id]])) {
      cal = models[[id]][[type]]
      if (is.null(cal) || nrow(test) == 0) {
        next
      }
      est = applyCal(cal, test$pm25_raw, rh = test$rh)
      m = calMetrics(est, test$pm25_ref)
      rows[[length(rows) + 1]] = data.table(sensor_id = id, model = type, t(m))
    }
  }
  rbindlist(rows)
}

applyBestModel = function(dt, models, scores) {
  best = scores[order(rmse), .SD[1], by = sensor_id]
  out = copy(dt)
  out[, pm25_cal := NA_real_]
  for (i in seq_len(nrow(best))) {
    id = best$sensor_id[i]
    cal = models[[id]][[best$model[i]]]
    rows = out$sensor_id == id
    out[rows, pm25_cal := applyCal(cal, pm25_raw, rh = rh)]
  }
  out
}

dailyIndex = function(dt, minHours = 18L) {
  daily = dt[, .(
    pm25 = mean(pm25_cal, na.rm = TRUE),
    nHours = sum(!is.na(pm25_cal))
  ), by = .(site = sensor_id, date = as.IDate(time))]
  daily[nHours < minHours, pm25 := NA_real_]
  daily[is.nan(pm25), pm25 := NA_real_]
  as.data.table(overallIndex(daily, pollutants = "pm25"))
}

exceedanceEpisodes = function(daily, minIndex = 101L, maxGap = 1L) {
  setorder(daily, site, date)
  episodes = list()
  for (s in unique(daily$site)) {
    d = daily[site == s]
    high = !is.na(d$aqi) & d$aqi >= minIndex
    if (!any(high)) {
      next
    }
    idx = which(high)
    startI = idx[1]
    endI = idx[1]
    for (j in idx[-1]) {
      if (j - endI - 1L <= maxGap) {
        endI = j
      } else {
        episodes[[length(episodes) + 1L]] = data.table(
          site = s, start = d$date[startI], end = d$date[endI],
          days = endI - startI + 1L, peak = max(d$aqi[startI:endI], na.rm = TRUE)
        )
        startI = j
        endI = j
      }
    }
    episodes[[length(episodes) + 1L]] = data.table(
      site = s, start = d$date[startI], end = d$date[endI],
      days = endI - startI + 1L, peak = max(d$aqi[startI:endI], na.rm = TRUE)
    )
  }
  if (length(episodes) == 0) {
    return(data.table(site = character(), start = as.IDate(character()),
                      end = as.IDate(character()), days = integer(), peak = integer()))
  }
  rbindlist(episodes)
}

compareDegreeDayMethods = function(stations, base = 5, upper = 30) {
  methods = c("average", "triangle", "sine")
  out = vector("list", length(methods))
  for (k in seq_along(methods)) {
    dd = stations[, .(
      date,
      dd = degreeDays(tmin, tmax, method = methods[k], base = base, upper = upper)
    ), by = station]
    acc = dd[, accumulateDegreeDays(date, dd, biofix = "01-01"), by = station]
    acc[, method := methods[k]]
    out[[k]] = acc
  }
  rbindlist(out)
}

stageCalendar = function(acc, stages, ddMethod = "sine") {
  sub = acc[method == ddMethod]
  onsets = as.data.table(stageOnsets(sub[, .(site = station, season, date, cumDd)], stages))
  onsets[, doy := as.integer(format(onset, "%j"))]
  dcast(onsets, site + season ~ stage, value.var = "doy")
}

# Sensor calibration ----------------------------------------------------------

colo = loadCollocation(sensor_path, reference_path)
train_end = as.POSIXct("2026-05-01", tz = "UTC")
models = fitSensorModels(colo, train_end)
scores = scoreSensorModels(colo, models, train_end)
fwrite(scores, "output/tables/sensor_calibration_scores.csv")

calibrated = applyBestModel(colo, models, scores)
aqi = dailyIndex(calibrated)
fwrite(aqi, "output/tables/daily_aqi.csv")
fwrite(as.data.table(categoryDays(aqi)), "output/tables/aqi_category_days.csv")
fwrite(exceedanceEpisodes(aqi), "output/tables/aqi_episodes.csv")

# Degree-days and phenology ---------------------------------------------------

stations = fread(stations_path)[, date := as.IDate(date)]
acc = compareDegreeDayMethods(stations)
fwrite(acc, "output/tables/gdd_methods.csv")

calendar = stageCalendar(acc, apple_stages, ddMethod = "sine")
fwrite(calendar, "output/tables/apple_stage_calendar.csv")
print(calendar)
