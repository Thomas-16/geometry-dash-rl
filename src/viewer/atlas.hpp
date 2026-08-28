// Cocos2d/TexturePacker format-3 sprite atlas loader (plist + png).
// Viewer-only: reads sheets from the GD install at runtime, never bundled.
#pragma once

#include "raylib.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace gdv {

struct AtlasFrame {
    int sheet = 0;         // index into Atlas sheet textures
    Rectangle rect{};      // textureRect; width/height = logical (unrotated) size, px
    bool rotated = false;  // frame packed rotated 90° CW in the sheet
    Vector2 offset{};      // spriteOffset px, +Y up: trimmed center - untrimmed center
    Vector2 sourceSize{};  // spriteSourceSize px (untrimmed)
};

class Atlas {
public:
    // Loads <stem>.plist and the sibling <stem>.png. Needs a GL context
    // (call after InitWindow). Returns false on any failure.
    bool loadSheet(const std::string& plistPath);

    // Exact frame key including the ".png" suffix; nullptr if absent.
    const AtlasFrame* find(const std::string& name) const;

    const Texture2D& texture(int sheet) const { return sheets_[sheet]; }
    size_t frameCount() const { return frames_.size(); }
    void unloadAll();

private:
    std::vector<Texture2D> sheets_;
    std::unordered_map<std::string, AtlasFrame> frames_;
};

}  // namespace gdv
