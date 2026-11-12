#include "obj_loader.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <string_view>

namespace {

constexpr std::string_view kWhitespace = " \t\r\v\f";

std::string_view nextToken(std::string_view& remaining) {
    const auto start = remaining.find_first_not_of(kWhitespace);
    if (start == std::string_view::npos) {
        remaining = {};
        return {};
    }
    remaining.remove_prefix(start);
    const auto token = remaining.substr(0, remaining.find_first_of(kWhitespace));
    remaining.remove_prefix(token.size());
    return token;
}

template <typename Number>
bool parseNumber(std::string_view token, Number& value) {
    if (token.empty()) return false;
    const char* end = token.data() + token.size();
    const auto result = std::from_chars(token.data(), end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parseCoordinate(std::string_view token, float& value) {
    if (!token.empty() && token.front() == '+') {
        token.remove_prefix(1);
        if (!token.empty() && token.front() == '-') return false;
    }
    return parseNumber(token, value) && std::isfinite(value);
}

bool parsePositionIndex(std::string_view token, std::size_t vertexCount, std::size_t& index) {
    // Faces may also carry texture and normal indices after the position.
    token = token.substr(0, token.find('/'));
    if (token.empty()) return false;
    const bool relative = token.front() == '-';
    if (relative || token.front() == '+') token.remove_prefix(1);

    std::size_t ordinal = 0;
    if (!parseNumber(token, ordinal) || ordinal == 0 || ordinal > vertexCount) return false;
    index = relative ? vertexCount - ordinal : ordinal - 1;
    return true;
}

}  // namespace

bool loadObj(const std::string& path, std::vector<Triangle>& out, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open OBJ file";
        return false;
    }

    std::vector<std::array<float, 3>> vertices;
    std::vector<Triangle> triangles;
    std::size_t lineNumber = 0;
    const auto failAtLine = [&](const char* message) {
        error = "line " + std::to_string(lineNumber) + ": " + message;
        return false;
    };

    std::string line;
    while (std::getline(input, line)) {
        ++lineNumber;
        std::string_view fields(line);
        fields = fields.substr(0, fields.find('#'));
        const auto type = nextToken(fields);
        if (type == "v") {
            std::array<float, 3> position;
            for (float& coordinate : position) {
                if (!parseCoordinate(nextToken(fields), coordinate))
                    return failAtLine("expected three finite vertex coordinates");
            }
            vertices.push_back(position);
        } else if (type == "f") {
            Triangle triangle{};
            for (float* vertex : {triangle.v0, triangle.v1, triangle.v2}) {
                const auto reference = nextToken(fields);
                if (reference.empty()) return failAtLine("expected a triangular face");
                std::size_t index = 0;
                if (!parsePositionIndex(reference, vertices.size(), index))
                    return failAtLine("invalid face position index");
                std::copy(vertices[index].begin(), vertices[index].end(), vertex);
            }
            if (!nextToken(fields).empty()) return failAtLine("expected a triangular face");
            if (triangles.size() == std::numeric_limits<uint32_t>::max())
                return failAtLine("too many triangles for the BVH format");
            triangle.id = static_cast<uint32_t>(triangles.size());
            triangles.push_back(triangle);
        }
    }
    if (input.bad() || !input.eof()) {
        error = "could not read OBJ file";
        return false;
    }
    if (triangles.empty()) {
        error = "OBJ file contains no triangular faces";
        return false;
    }

    out.swap(triangles);
    error.clear();
    return true;
}
