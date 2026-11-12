#pragma once

#include "bvh_io.h"

// Builds a 2-wide SAH tree with surface-area visit probabilities. On failure,
// sets error and leaves out unchanged.
bool buildBinarySah(const std::vector<Triangle>& triangles, TreeFile& out, std::string& error);
