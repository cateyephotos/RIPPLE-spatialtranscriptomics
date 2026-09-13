# The bundled `ripple_mock_data` deliberately has three sections sharing one
# coordinate frame, which is what real per-section Xenium and Visium output
# looks like. 54.7% of its cells would take a cross-sample nearest neighbour
# under a pooled search, so run_ripple() correctly reports a severe overlap
# every time it touches that dataset.
#
# For tests where coordinate frames are the subject, that warning should be
# asserted on. For tests that merely need a working multi-sample fixture it is
# incidental noise, and left unhandled it accumulates into the same
# alarm-fatigue problem the graded warning exists to avoid: a wall of expected
# W markers is exactly where a genuinely new warning goes unnoticed.
#
# So muffle that one warning by message, never with a blanket
# suppressWarnings(), which would also hide the warnings these tests do want
# to surface (too few valid samples, distance-cap clamping, dropped cells with
# no query cell in their sample).
without_frame_warning <- function(expr) {
  withCallingHandlers(
    expr,
    warning = function(w) {
      if (grepl("overlapping coordinate bounding boxes",
        conditionMessage(w),
        fixed = TRUE
      )) {
        invokeRestart("muffleWarning")
      }
    }
  )
}
