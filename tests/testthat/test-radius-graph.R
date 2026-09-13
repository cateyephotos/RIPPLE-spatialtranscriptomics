radius_builders <- function() {
  # Exercise the shipped standalone helper without loading unrelated scripts.
  code <- parse(system.file("scripts/utils.R", package = "ripple"))
  env <- new.env(parent = baseenv())
  for (expr in code) {
    if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
        identical(expr[[2]], as.name("build_radius_graph"))) eval(expr, env)
  }
  list(package = build_radius_graph, standalone = env$build_radius_graph)
}

test_that("radius searches return every neighbor in dense regions", {
  # Global density is low, but each clustered cell has nine neighbors in range.
  xy <- rbind(cbind(seq(0, 0.09, length.out = 10), rep(0, 10)), c(1000, 1000))
  for (build in radius_builders()) {
    graph <- suppressMessages(build(xy, radius = 1, sample_ids = rep("s", 11)))
    expect_equal(lengths(graph), c(rep(9L, 10), 0L))
    expect_identical(sort(graph[[1]]), 2:10)
  }
})

test_that("radius graphs match direct distances within interleaved samples", {
  set.seed(481)
  xy <- matrix(runif(120), ncol = 2)
  sid <- rep(c("a", "b", "c"), 20)
  sid[59] <- "singleton"
  sid[60] <- NA_character_
  distance <- as.matrix(stats::dist(xy))
  expected <- lapply(seq_len(nrow(xy)), function(i) {
    unname(which(!is.na(sid) & !is.na(sid[i]) & sid == sid[i] &
                   seq_along(sid) != i & distance[i, ] <= 0.3))
  })
  for (build in radius_builders()) {
    graph <- suppressMessages(build(xy, radius = 0.3, sample_ids = sid))
    expect_identical(lapply(graph, sort), expected)
  }
})

test_that("radius boundaries are inclusive and samples stay separate", {
  xy <- rbind(c(0, 0), c(3, 4), c(0, 0), c(3, 4), c(20, 20))
  for (build in radius_builders()) {
    graph <- suppressMessages(build(xy, radius = 5,
                                    sample_ids = c("a", "a", "b", "b", "a")))
    expect_identical(graph, list(2L, 1L, 4L, 3L, integer(0)))
  }
})
