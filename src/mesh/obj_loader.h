#pragma once

#include <string>
#include <vector>

#include "bvh_format.h"

// Reads positions and triangular faces from an OBJ. Triangle IDs follow face order.
// On failure, sets error and leaves out unchanged.
bool loadObj(const std::string& path, std::vector<Triangle>& out, std::string& error);
