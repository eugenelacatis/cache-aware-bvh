#pragma once

#include <cstdint>
#include <vector>

#include "bvh_format.h"

// Isotropic uniform random lines that cross the bounding sphere of `box`
// (the measure behind T2 and the SAH). Each ray starts on the sphere, points
// inward, and has tmin = 0, tmax = sphere diameter, so it covers the full chord.
std::vector<Ray> isotropicLines(const Box& box, uint32_t count, uint64_t seed);
