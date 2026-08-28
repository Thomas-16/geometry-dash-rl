// Decodes a level file and prints object stats. Doubles as a smoke test for
// the decode/parse pipeline.

#include "core/level.hpp"

#include <algorithm>
#include <cstdio>
#include <map>

static const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : kDefaultLevel;
    gd::Level level;
    try {
        level = gd::Level::loadFromFile(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::map<int, int> counts;
    float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
    for (const auto& o : level.objects) {
        counts[o.id]++;
        minX = std::min(minX, o.x);
        maxX = std::max(maxX, o.x);
        minY = std::min(minY, o.y);
        maxY = std::max(maxY, o.y);
    }

    std::printf("file:            %s\n", path);
    std::printf("header (%zu chars): %.80s...\n", level.header.size(),
                level.header.c_str());
    std::printf("objects:         %zu\n", level.objects.size());
    std::printf("distinct ids:    %zu\n", counts.size());
    std::printf("x range:         %.1f .. %.1f\n", minX, maxX);
    std::printf("y range:         %.1f .. %.1f\n", minY, maxY);
    std::printf("\nid counts (desc):\n");
    std::vector<std::pair<int, int>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    for (auto& [id, n] : sorted) std::printf("  id %4d  x%d\n", id, n);
    return 0;
}
