newton.raphson <- function(dose1, omega, slope1, slopeU) {
  dta <- function(id, cycle, course) {
    # ..which stands for 'Dose Titration Algorithm' of course!
    # The data frame 'course' is assumed to have columns: 'cycle','id','scaled.dose','CircMin'
    # and in general may or may not have been subsetted to just a single id.
    dose <- dose1
    if (cycle > 1) {
      lag_1 <- which(course$cycle == (cycle - 1) & course$id == id)
      CircMin_1 <- course[lag_1, 'CircMin']
      dose_1 <- course[lag_1, 'dose']
      if (cycle == 2) {
        slope <- slope1
      } else {
        # cycle >= 3 so we also have lag -2 to look back at
        lag_2 <- which(course$cycle == (cycle - 2) & course$id == id)
        CircMin_2 <- course[lag_2, 'CircMin']
        dY <- log(CircMin_1 / CircMin_2)
        dose_2 <- course[lag_2, 'dose']
        dX <- scaled(dose_1) - scaled(dose_2)
        slope <- dY / dX
        slope <- min(slope, slopeU) # bound dY/dX away from zero in face of possible hysteresis
      }
      delta.scaleddose <- log(500 / CircMin_1) / slope
      # For safety's sake, we (asymmetrically) apply relaxation factor 'omega' to any proposed dose increase:
      delta.safer <- ifelse(
        delta.scaleddose > 0,
        omega * delta.scaleddose,
        delta.scaleddose
      )
      new.scaleddose <- scaled(dose_1) + delta.safer
      dose <- uniroot(function(y) scaled(y) - new.scaleddose, c(0, 100000))$root
    }
    return(dose)
  }
  dta
}