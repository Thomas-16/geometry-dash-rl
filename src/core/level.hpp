#pragma once

#include <string>
#include <utility>
#include <vector>

namespace gd {

// One object from the inner level string. Coordinates are native GD units
// (30 units = 1 block), y-up, (x, y) is the object's center.
struct Object {
    int id = 0;
    float x = 0.0f;
    float y = 0.0f;
    float rotation = 0.0f; // degrees, clockwise as seen in game
    bool flipX = false;
    bool flipY = false;
    // Every key/value pair as found in the level string (including the ones
    // decoded above), so nothing is lost for later sim work.
    std::vector<std::pair<int, std::string>> props;

    const std::string* prop(int key) const;
};

struct Level {
    std::string header; // first ';' segment (kA/kS settings), kept raw
    std::vector<Object> objects;

    // Reads an official-level file (Resources/levels/<id>.txt):
    // URL-safe base64 of gzip data.
    static Level loadFromFile(const std::string& path);

    // Parses an already-decoded inner level string.
    static Level parse(const std::string& decoded);
};

// URL-safe base64 decode -> gzip inflate. Throws std::runtime_error on
// malformed input.
std::string decodeLevelString(const std::string& raw);

} // namespace gd
