#include "atlas.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gdv {

namespace {

// Returns the text between "<key>" and "</key>" starting at or after pos,
// and advances pos past the closing tag. Empty string when no key remains.
std::string nextKey(const std::string& s, size_t& pos) {
    size_t k = s.find("<key>", pos);
    if (k == std::string::npos) return {};
    k += 5;
    size_t e = s.find("</key>", k);
    if (e == std::string::npos) return {};
    pos = e + 6;
    return s.substr(k, e - k);
}

// Value of the first <string>...</string> after pos, bounded by end.
std::string stringValueAfter(const std::string& s, size_t pos, size_t end) {
    size_t k = s.find("<string>", pos);
    if (k == std::string::npos || k >= end) return {};
    k += 8;
    size_t e = s.find("</string>", k);
    if (e == std::string::npos || e > end) return {};
    return s.substr(k, e - k);
}

bool parseVec2(const std::string& v, Vector2& out) {
    return std::sscanf(v.c_str(), "{%f,%f}", &out.x, &out.y) == 2;
}

bool parseRect(const std::string& v, Rectangle& out) {
    return std::sscanf(v.c_str(), "{{%f,%f},{%f,%f}}", &out.x, &out.y, &out.width, &out.height) == 4;
}

}  // namespace

bool Atlas::loadSheet(const std::string& plistPath) {
    std::ifstream in(plistPath, std::ios::binary);
    if (!in) {
        TraceLog(LOG_WARNING, "atlas: cannot open %s", plistPath.c_str());
        return false;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string s = buf.str();

    size_t pos = s.find("<key>frames</key>");
    if (pos == std::string::npos) {
        TraceLog(LOG_WARNING, "atlas: no frames dict in %s", plistPath.c_str());
        return false;
    }
    // Step past the "frames" key itself, or nextKey() returns it as a frame
    // name and its fake body swallows the first real frame of the sheet.
    pos += std::strlen("<key>frames</key>");

    const int sheetIndex = (int)sheets_.size();
    size_t added = 0;
    for (;;) {
        std::string name = nextKey(s, pos);
        if (name.empty() || name == "metadata") break;

        // Frame dicts contain no nested <dict> (aliases is an empty <array/>),
        // so the body ends at the first </dict>.
        size_t bodyStart = s.find("<dict>", pos);
        size_t bodyEnd = s.find("</dict>", bodyStart);
        if (bodyStart == std::string::npos || bodyEnd == std::string::npos) break;

        AtlasFrame f;
        f.sheet = sheetIndex;
        Vector2 spriteSize{};
        bool ok = true;
        size_t p = bodyStart;
        for (;;) {
            std::string key = nextKey(s, p);
            if (key.empty() || p > bodyEnd) break;
            if (key == "spriteOffset") {
                ok &= parseVec2(stringValueAfter(s, p, bodyEnd), f.offset);
            } else if (key == "spriteSize") {
                ok &= parseVec2(stringValueAfter(s, p, bodyEnd), spriteSize);
            } else if (key == "spriteSourceSize") {
                ok &= parseVec2(stringValueAfter(s, p, bodyEnd), f.sourceSize);
            } else if (key == "textureRect") {
                ok &= parseRect(stringValueAfter(s, p, bodyEnd), f.rect);
            } else if (key == "textureRotated") {
                size_t t = s.find("<true/>", p);
                size_t nk = s.find("<key>", p);
                f.rotated = t != std::string::npos && t < bodyEnd &&
                            (nk == std::string::npos || t < nk);
            }
        }
        pos = bodyEnd + 7;
        if (!ok) {
            TraceLog(LOG_WARNING, "atlas: bad frame %s in %s", name.c_str(), plistPath.c_str());
            continue;
        }
        if (spriteSize.x != f.rect.width || spriteSize.y != f.rect.height)
            TraceLog(LOG_WARNING, "atlas: %s spriteSize != textureRect size", name.c_str());
        frames_[name] = f;
        ++added;
    }

    if (added == 0) {
        TraceLog(LOG_WARNING, "atlas: no frames parsed from %s", plistPath.c_str());
        return false;
    }

    std::string pngPath = plistPath;
    size_t dot = pngPath.rfind(".plist");
    if (dot != std::string::npos) pngPath.replace(dot, 6, ".png");
    Texture2D tex = LoadTexture(pngPath.c_str());
    if (tex.id == 0) {
        TraceLog(LOG_WARNING, "atlas: failed to load %s", pngPath.c_str());
        return false;
    }
    GenTextureMipmaps(&tex);
    SetTextureFilter(tex, TEXTURE_FILTER_TRILINEAR);
    sheets_.push_back(tex);
    TraceLog(LOG_INFO, "atlas: %s — %zu frames, %dx%d", plistPath.c_str(), added, tex.width,
             tex.height);
    return true;
}

const AtlasFrame* Atlas::find(const std::string& name) const {
    auto it = frames_.find(name);
    return it == frames_.end() ? nullptr : &it->second;
}

void Atlas::unloadAll() {
    for (auto& t : sheets_) UnloadTexture(t);
    sheets_.clear();
    frames_.clear();
}

}  // namespace gdv
