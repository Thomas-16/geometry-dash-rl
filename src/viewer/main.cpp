// Level viewer: renders a decoded level with the game's own sprites, loaded
// at runtime from the GD install (never bundled). Falls back to colored
// shapes when the atlases are unavailable. All tables here are
// presentation-only — collision/hitbox data comes later from the binary,
// per gd-replica-notes.md.
//
// Usage: viewer [level.txt] [--res <ResourcesDir>] [--camx N] [--zoom Z]
//               [--shot out.png]
//   --shot renders a few frames, saves a screenshot, and exits (for
//   automated visual checks).
// Keys: T = textured/shapes toggle, D = debug markers (triggers, unknowns)

#include "core/level.hpp"
#include "viewer/atlas.hpp"
#include "viewer/gd_frames.hpp"

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

static constexpr float kUnitsPerPx = 0.25f;   // uhd sheets: 4 px per GD unit
static constexpr float kCubeSpeed = 311.58f;  // GD 1x speed, units/sec (approx)

// Classification of the object ids that appear in Stereo Madness, derived
// empirically from position patterns in the level data (see repo notes).
enum class Shape { Block, Spike, SmallSpike, GroundSpikes, Portal, Coin, Trigger, Deco, Unknown };

static Shape classify(int id) {
    switch (id) {
        case 1: case 2: case 3: case 4: case 5: case 6: case 7:
            return Shape::Block;
        case 8:
            return Shape::Spike;
        case 39: case 103:
            return Shape::SmallSpike;
        case 9:
            return Shape::GroundSpikes;
        case 12: case 13:
            return Shape::Portal;
        case 142:
            return Shape::Coin;
        case 29: case 30:
            return Shape::Trigger;
        case 15: case 16: case 17: case 18: case 19: case 20: case 21:
        case 22: case 23: case 27: case 40: case 41: case 54: case 62: case 65:
            return Shape::Deco;
        default:
            return Shape::Unknown;
    }
}

// Object-local offset (GD units, y-up) -> screen point, applying flip,
// clockwise rotation, and the world y-flip.
static Vector2 localToScreen(const gd::Object& o, float dx, float dy) {
    if (o.flipX) dx = -dx;
    if (o.flipY) dy = -dy;
    float t = o.rotation * DEG2RAD; // clockwise in y-up coords
    float rx = dx * std::cos(t) + dy * std::sin(t);
    float ry = -dx * std::sin(t) + dy * std::cos(t);
    return {o.x + rx, -(o.y + ry)};
}

// DrawTriangle needs counterclockwise winding in screen space.
static void drawTriCCW(Vector2 a, Vector2 b, Vector2 c, Color col) {
    float cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (cross > 0) std::swap(b, c);
    DrawTriangle(a, b, c, col);
}

static void drawSpikeTri(const gd::Object& o, float halfW, float baseY,
                         float apexY, Color col) {
    drawTriCCW(localToScreen(o, -halfW, baseY), localToScreen(o, halfW, baseY),
               localToScreen(o, 0, apexY), col);
}

static void drawObjectShapes(const gd::Object& o) {
    switch (classify(o.id)) {
        case Shape::Block: {
            Rectangle r{o.x, -o.y, 30, 30};
            DrawRectanglePro(r, {15, 15}, o.rotation, Color{200, 200, 210, 255});
            // outline: rotated rect via 4 corners
            Vector2 p[4] = {localToScreen(o, -15, 15), localToScreen(o, 15, 15),
                            localToScreen(o, 15, -15), localToScreen(o, -15, -15)};
            for (int i = 0; i < 4; ++i)
                DrawLineEx(p[i], p[(i + 1) % 4], 1.5f, Color{60, 60, 80, 255});
            break;
        }
        case Shape::Spike:
            drawSpikeTri(o, 14, -15, 15, Color{230, 60, 60, 255});
            break;
        case Shape::SmallSpike:
            drawSpikeTri(o, 14, -7, 8, Color{230, 60, 60, 255});
            break;
        case Shape::GroundSpikes:
            // strip of 3 small teeth across one block width
            for (int i = -1; i <= 1; ++i)
                drawTriCCW(localToScreen(o, i * 10.f - 5.f, -4),
                           localToScreen(o, i * 10.f + 5.f, -4),
                           localToScreen(o, i * 10.f, 5), Color{230, 60, 60, 255});
            break;
        case Shape::Portal: {
            Color col = (o.id == 13) ? Color{240, 120, 220, 255}   // ship-side
                                     : Color{110, 240, 110, 255};  // cube-side
            Rectangle r{o.x, -o.y, 16, 85};
            DrawRectanglePro(r, {8, 42.5f}, o.rotation, Fade(col, 0.75f));
            DrawText(TextFormat("%d", o.id), (int)o.x - 8, (int)-o.y - 60, 10, col);
            break;
        }
        case Shape::Coin:
            DrawCircleV({o.x, -o.y}, 12, Fade(YELLOW, 0.4f));
            DrawCircleLinesV({o.x, -o.y}, 12, YELLOW);
            break;
        case Shape::Trigger: {
            // 1.0 color triggers: keys 7,8,9 hold the target RGB
            unsigned char rr = 128, gg = 128, bb = 128;
            if (auto* v = o.prop(7)) rr = (unsigned char)std::atoi(v->c_str());
            if (auto* v = o.prop(8)) gg = (unsigned char)std::atoi(v->c_str());
            if (auto* v = o.prop(9)) bb = (unsigned char)std::atoi(v->c_str());
            Vector2 c{o.x, -o.y};
            Vector2 pts[4] = {{c.x, c.y - 10}, {c.x + 10, c.y}, {c.x, c.y + 10}, {c.x - 10, c.y}};
            drawTriCCW(pts[0], pts[1], pts[3], Color{rr, gg, bb, 255});
            drawTriCCW(pts[1], pts[2], pts[3], Color{rr, gg, bb, 255});
            DrawText(o.id == 29 ? "BG" : "G", (int)c.x - 6, (int)c.y - 4, 8, WHITE);
            break;
        }
        case Shape::Deco: {
            Vector2 p[4] = {localToScreen(o, -14, 14), localToScreen(o, 14, 14),
                            localToScreen(o, 14, -14), localToScreen(o, -14, -14)};
            for (int i = 0; i < 4; ++i)
                DrawLineEx(p[i], p[(i + 1) % 4], 1.0f, Color{110, 110, 130, 90});
            break;
        }
        case Shape::Unknown: {
            Vector2 p[4] = {localToScreen(o, -15, 15), localToScreen(o, 15, 15),
                            localToScreen(o, 15, -15), localToScreen(o, -15, -15)};
            for (int i = 0; i < 4; ++i)
                DrawLineEx(p[i], p[(i + 1) % 4], 2.0f, MAGENTA);
            DrawText(TextFormat("%d", o.id), (int)o.x - 10, (int)-o.y - 5, 10, MAGENTA);
            break;
        }
    }
}

static float propFloat(const gd::Object& o, int key, float def) {
    auto* v = o.prop(key);
    return v ? std::strtof(v->c_str(), nullptr) : def;
}

static int propInt(const gd::Object& o, int key, int def) {
    auto* v = o.prop(key);
    return v ? std::atoi(v->c_str()) : def;
}

// Draws one atlas frame at the object's transform. Frames flagged rotated are
// packed 90° CW in the sheet: swap source/dest dims and compensate with -90°.
// spriteOffset is a center-relative local offset (+Y up), so it composes with
// flip/rotation/y-negation exactly like any local offset via localToScreen.
static void drawFrame(const gdv::Atlas& atlas, const gdv::AtlasFrame& f,
                      const gd::Object& o, float scale, Color tint) {
    Vector2 c = localToScreen(o, f.offset.x * kUnitsPerPx * scale,
                              f.offset.y * kUnitsPerPx * scale);
    float w = f.rect.width * kUnitsPerPx * scale;
    float h = f.rect.height * kUnitsPerPx * scale;
    const Texture2D& tex = atlas.texture(f.sheet);
    if (!f.rotated) {
        Rectangle src{f.rect.x, f.rect.y, f.rect.width * (o.flipX ? -1.f : 1.f),
                      f.rect.height * (o.flipY ? -1.f : 1.f)};
        DrawTexturePro(tex, src, {c.x, c.y, w, h}, {w * 0.5f, h * 0.5f}, o.rotation, tint);
    } else {
        // Physical (swapped) region; flip axes swap under the 90° CW packing.
        Rectangle src{f.rect.x, f.rect.y, f.rect.height, f.rect.width};
        if (o.flipY) src.width = -src.width;
        if (o.flipX) src.height = -src.height;
        DrawTexturePro(tex, src, {c.x, c.y, h, w}, {h * 0.5f, w * 0.5f}, o.rotation - 90.0f,
                       tint);
    }
}

// True if the object was fully handled by the textured path (including
// invisible objects with an empty layer list).
static bool drawTexturedObject(const gdv::Atlas& atlas, const gd::Object& o) {
    const gdv::RenderSpec* spec = gdv::renderSpec(o.id);
    if (!spec) return false;
    float scale = propFloat(o, 32, 1.0f);
    bool drewAny = false, missing = false;
    for (const char* name : spec->layers) {
        if (!name) continue;
        const gdv::AtlasFrame* f = atlas.find(name);
        if (!f) { missing = true; continue; }
        drawFrame(atlas, *f, o, scale, WHITE);
        drewAny = true;
    }
    bool hasLayers = spec->layers[0] != nullptr;
    return !hasLayers || (drewAny && !missing);
}

// Piecewise color timeline built from the header color + 1.0 color triggers.
struct ColorTrack {
    struct Key {
        float x;
        Color target;
        float durUnits;  // trigger duration converted to world-x units
    };
    Color initial{};
    std::vector<Key> keys;

    Color eval(float camX) const {
        Color cur = initial;
        for (const auto& k : keys) {
            if (camX < k.x) break;
            if (k.durUnits <= 0 || camX >= k.x + k.durUnits) {
                cur = k.target;
            } else {
                float t = (camX - k.x) / k.durUnits;
                cur = Color{(unsigned char)(cur.r + (k.target.r - cur.r) * t),
                            (unsigned char)(cur.g + (k.target.g - cur.g) * t),
                            (unsigned char)(cur.b + (k.target.b - cur.b) * t), 255};
            }
        }
        return cur;
    }
};

// Parses "1_R_2_G_3_B_..." after "<key>," in the level header (kS29/kS30/...).
static Color headerColor(const std::string& header, const char* key, Color fallback) {
    size_t p = header.find(std::string(key) + ",");
    if (p == std::string::npos) return fallback;
    p += std::strlen(key) + 1;
    size_t end = header.find(',', p);
    std::string s = header.substr(p, end == std::string::npos ? std::string::npos : end - p);
    int vals[6] = {0, 0, 0, 0, 0, 0};  // key/value alternating: 1 R 2 G 3 B ...
    Color c = fallback;
    int k = -1, i = 0;
    for (size_t j = 0; j <= s.size(); ++j) {
        if (j == s.size() || s[j] == '_') {
            int v = std::atoi(s.c_str() + i);
            if (k < 0) {
                k = v;
            } else {
                if (k == 1) c.r = (unsigned char)v;
                if (k == 2) c.g = (unsigned char)v;
                if (k == 3) c.b = (unsigned char)v;
                k = -1;
            }
            i = (int)j + 1;
        }
    }
    c.a = 255;
    return c;
}

static ColorTrack buildTrack(const gd::Level& level, int triggerId, Color initial) {
    ColorTrack tr;
    tr.initial = initial;
    for (const auto& o : level.objects) {
        if (o.id != triggerId) continue;
        Color target{(unsigned char)propInt(o, 7, 128), (unsigned char)propInt(o, 8, 128),
                     (unsigned char)propInt(o, 9, 128), 255};
        tr.keys.push_back({o.x, target, propFloat(o, 10, 0) * kCubeSpeed});
    }
    std::sort(tr.keys.begin(), tr.keys.end(),
              [](const auto& a, const auto& b) { return a.x < b.x; });
    return tr;
}

int main(int argc, char** argv) {
    std::string levelPath = kDefaultLevel;
    std::string shotPath, resDirOverride;
    float camX = 0.0f, zoom = 1.5f;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc) shotPath = argv[++i];
        else if (!std::strcmp(argv[i], "--camx") && i + 1 < argc) camX = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--res") && i + 1 < argc) resDirOverride = argv[++i];
        else levelPath = argv[i];
    }

    gd::Level level;
    try {
        level = gd::Level::loadFromFile(levelPath);
    } catch (const std::exception& e) {
        TraceLog(LOG_ERROR, "failed to load level: %s", e.what());
        return 1;
    }
    float maxX = 0;
    for (const auto& o : level.objects) maxX = std::max(maxX, o.x);

    // Draw order: z-layer (key 24, else per-id default), then z-order (key 25);
    // stable sort keeps file order on ties.
    std::vector<const gd::Object*> drawOrder;
    drawOrder.reserve(level.objects.size());
    for (const auto& o : level.objects) drawOrder.push_back(&o);
    auto zLayer = [](const gd::Object& o) {
        const gdv::RenderSpec* s = gdv::renderSpec(o.id);
        return propInt(o, 24, s ? s->defaultZ : 0);
    };
    std::stable_sort(drawOrder.begin(), drawOrder.end(),
                     [&](const gd::Object* a, const gd::Object* b) {
                         int za = zLayer(*a), zb = zLayer(*b);
                         if (za != zb) return za < zb;
                         return propInt(*a, 25, 0) < propInt(*b, 25, 0);
                     });

    ColorTrack bgTrack = buildTrack(level, 29, headerColor(level.header, "kS29", {40, 62, 255, 255}));
    ColorTrack groundTrack =
        buildTrack(level, 30, headerColor(level.header, "kS30", {0, 19, 200, 255}));
    Color lineColor = headerColor(level.header, "kS31", WHITE);

    const int W = 1600, H = 900;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(W, H, "gd-rl viewer — Stereo Madness");
    SetTargetFPS(60);

    // Resources dir: explicit --res, else derived from the level path
    // (Resources/levels/1.txt -> Resources).
    std::filesystem::path resDir = resDirOverride.empty()
                                       ? std::filesystem::path(levelPath).parent_path().parent_path()
                                       : std::filesystem::path(resDirOverride);
    gdv::Atlas atlas;
    bool textured = atlas.loadSheet((resDir / "GJ_GameSheet-uhd.plist").string()) &&
                    atlas.loadSheet((resDir / "GJ_GameSheet02-uhd.plist").string());
    if (textured) {
        // Validate the whole id->frame table up front so mapping mistakes are
        // loud instead of silently falling back per object.
        for (int id : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                       22, 23, 27, 29, 30, 39, 40, 41, 54, 62, 65, 103, 142}) {
            const gdv::RenderSpec* s = gdv::renderSpec(id);
            if (!s) continue;
            for (const char* name : s->layers)
                if (name && !atlas.find(name))
                    TraceLog(LOG_WARNING, "no atlas frame %s for id %d", name, id);
        }
    } else {
        TraceLog(LOG_WARNING, "atlases unavailable — falling back to shape rendering");
        atlas.unloadAll();
    }

    Texture2D bgTex{}, groundTex{};
    if (textured) {
        bgTex = LoadTexture((resDir / "game_bg_01_001-uhd.png").string().c_str());
        groundTex = LoadTexture((resDir / "groundSquare_01_001-uhd.png").string().c_str());
        if (bgTex.id) SetTextureWrap(bgTex, TEXTURE_WRAP_REPEAT);
        if (groundTex.id) SetTextureWrap(groundTex, TEXTURE_WRAP_REPEAT);
    }

    Camera2D cam{};
    cam.zoom = zoom;
    cam.target = {camX, -180.0f};
    bool autoScroll = false, useTextures = textured, debugMarkers = false;

    int frame = 0;
    while (!WindowShouldClose()) {
        cam.offset = {GetScreenWidth() * 0.25f, GetScreenHeight() * 0.6f};

        if (IsKeyPressed(KEY_SPACE)) autoScroll = !autoScroll;
        if (IsKeyPressed(KEY_T) && textured) useTextures = !useTextures;
        if (IsKeyPressed(KEY_D)) debugMarkers = !debugMarkers;
        if (IsKeyPressed(KEY_R)) { cam.target = {0, -180.0f}; cam.zoom = 1.5f; autoScroll = false; }
        if (autoScroll) cam.target.x += kCubeSpeed * GetFrameTime();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 d = GetMouseDelta();
            cam.target.x -= d.x / cam.zoom;
            cam.target.y -= d.y / cam.zoom;
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            Vector2 before = GetScreenToWorld2D(GetMousePosition(), cam);
            cam.zoom = std::clamp(cam.zoom * (1.0f + 0.1f * wheel), 0.05f, 10.0f);
            Vector2 after = GetScreenToWorld2D(GetMousePosition(), cam);
            cam.target.x += before.x - after.x;
            cam.target.y += before.y - after.y;
        }

        Color bgCol = bgTrack.eval(cam.target.x);
        Color groundCol = groundTrack.eval(cam.target.x);

        BeginDrawing();
        ClearBackground(useTextures ? Color{(unsigned char)(bgCol.r / 2),
                                            (unsigned char)(bgCol.g / 2),
                                            (unsigned char)(bgCol.b / 2), 255}
                                    : Color{25, 28, 48, 255});

        // Background: screen space so it fills the window at any zoom, with a
        // 0.5x parallax scroll like the game.
        if (useTextures && bgTex.id) {
            float sw = (float)GetScreenWidth(), sh = (float)GetScreenHeight();
            float srcH = (float)bgTex.height;
            float srcW = sw * srcH / sh;
            float srcX = cam.target.x * 0.5f * (srcH / sh);
            DrawTexturePro(bgTex, {srcX, 0, srcW, srcH}, {0, 0, sw, sh}, {0, 0}, 0, bgCol);
        }

        BeginMode2D(cam);
        {
            // visible world x-range for grid lines
            Vector2 tl = GetScreenToWorld2D({0, 0}, cam);
            Vector2 br = GetScreenToWorld2D({(float)GetScreenWidth(), (float)GetScreenHeight()}, cam);
            // ground: floor surface is world y=0 (screen y=0), drawn solid below
            if (useTextures && groundTex.id) {
                // 512 px = 128 units at uhd scale; scrolls 1:1 with the world.
                float gh = groundTex.height * kUnitsPerPx;
                DrawTexturePro(groundTex,
                               {tl.x / kUnitsPerPx, 0, (br.x - tl.x) / kUnitsPerPx,
                                (float)groundTex.height},
                               {tl.x, 0, br.x - tl.x, gh}, {0, 0}, 0, groundCol);
                if (br.y > gh)
                    DrawRectangleV({tl.x, gh}, {br.x - tl.x, br.y - gh},
                                   Color{(unsigned char)(groundCol.r / 2),
                                         (unsigned char)(groundCol.g / 2),
                                         (unsigned char)(groundCol.b / 2), 255});
            } else {
                DrawRectangleV({tl.x, 0}, {br.x - tl.x, std::max(0.0f, br.y)},
                               Color{40, 44, 80, 255});
            }
            // grid (30-unit spacing), faint
            for (float gx = std::floor(tl.x / 30) * 30; gx <= br.x; gx += 30)
                DrawLineEx({gx, tl.y}, {gx, br.y}, 1.0f / cam.zoom, Color{255, 255, 255, 10});
            for (float gy = std::floor(tl.y / 30) * 30; gy <= br.y; gy += 30)
                DrawLineEx({tl.x, gy}, {br.x, gy}, 1.0f / cam.zoom, Color{255, 255, 255, 10});
            DrawLineEx({tl.x, 0}, {br.x, 0}, 2.0f / cam.zoom, lineColor);

            for (const gd::Object* o : drawOrder) {
                if (useTextures) {
                    bool handled = drawTexturedObject(atlas, *o);
                    if (!handled) drawObjectShapes(*o);
                    else if (debugMarkers && !gdv::renderSpec(o->id)->layers[0])
                        drawObjectShapes(*o);  // invisible objects: show marker
                } else {
                    drawObjectShapes(*o);
                }
            }
        }
        EndMode2D();

        DrawText(TextFormat("objects: %d   cam.x: %.0f / %.0f   zoom: %.2f   %d fps%s",
                            (int)level.objects.size(), cam.target.x, maxX,
                            cam.zoom, GetFPS(), useTextures ? "" : "   [shapes]"),
                 12, 10, 20, RAYWHITE);
        DrawText("drag: pan   wheel: zoom   space: auto-scroll   R: reset   T: textures   D: debug",
                 12, 34, 16, Color{170, 175, 200, 255});
        EndDrawing();

        if (!shotPath.empty() && ++frame >= 5) {
            TakeScreenshot(shotPath.c_str());
            break;
        }
    }
    if (bgTex.id) UnloadTexture(bgTex);
    if (groundTex.id) UnloadTexture(groundTex);
    atlas.unloadAll();
    CloseWindow();
    return 0;
}
