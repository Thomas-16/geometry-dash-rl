#include "viewer/world.hpp"

#include "viewer/gd_frames.hpp"
#include "viewer/levels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace gdv {

namespace {

// Classification of the object ids that appear in Stereo Madness, derived
// empirically from position patterns in the level data (see repo notes).
enum class Shape { Block, Spike, SmallSpike, GroundSpikes, Portal, Coin, Trigger, Deco, Unknown };

Shape classify(int id) {
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
        case 22: case 23: case 26: case 27: case 40: case 41: case 54: case 62:
        case 65: case 68:
            return Shape::Deco;
        default:
            return Shape::Unknown;
    }
}

// Object-local offset (GD units, y-up) -> screen point, applying flip,
// clockwise rotation, and the world y-flip.
Vector2 localToScreen(const gd::Object& o, float dx, float dy) {
    if (o.flipX) dx = -dx;
    if (o.flipY) dy = -dy;
    float t = o.rotation * DEG2RAD; // clockwise in y-up coords
    float rx = dx * std::cos(t) + dy * std::sin(t);
    float ry = -dx * std::sin(t) + dy * std::cos(t);
    return {o.x + rx, -(o.y + ry)};
}

// DrawTriangle needs counterclockwise winding in screen space.
void drawTriCCW(Vector2 a, Vector2 b, Vector2 c, Color col) {
    float cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (cross > 0) std::swap(b, c);
    DrawTriangle(a, b, c, col);
}

void drawSpikeTri(const gd::Object& o, float halfW, float baseY,
                  float apexY, Color col) {
    drawTriCCW(localToScreen(o, -halfW, baseY), localToScreen(o, halfW, baseY),
               localToScreen(o, 0, apexY), col);
}

void drawObjectShapes(const gd::Object& o) {
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
            Color col = (o.id == 13)   ? Color{240, 120, 220, 255}   // ship-side
                        : (o.id == 47) ? Color{240, 90, 90, 255}     // ball portal
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

// Draws one atlas frame at the object's transform. Frames flagged rotated are
// packed 90Â° CW in the sheet: swap source/dest dims and compensate with -90Â°.
// spriteOffset is a center-relative local offset (+Y up), so it composes with
// flip/rotation/y-negation exactly like any local offset via localToScreen.
void drawFrame(const Atlas& atlas, const AtlasFrame& f,
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
        // Physical (swapped) region; flip axes swap under the 90Â° CW packing.
        Rectangle src{f.rect.x, f.rect.y, f.rect.height, f.rect.width};
        if (o.flipY) src.width = -src.width;
        if (o.flipX) src.height = -src.height;
        DrawTexturePro(tex, src, {c.x, c.y, h, w}, {h * 0.5f, w * 0.5f}, o.rotation - 90.0f,
                       tint);
    }
}

// True if the object was fully handled by the textured path (including
// invisible objects with an empty layer list).
bool drawTexturedObject(const Atlas& atlas, const gd::Object& o) {
    const RenderSpec* spec = renderSpec(o.id);
    if (!spec) return false;
    float scale = propFloat(o, 32, 1.0f);
    Color tint = spec->tint == Tint::Black ? BLACK : WHITE;
    if (spec->tint == Tint::Glow) BeginBlendMode(BLEND_ADDITIVE);
    bool drewAny = false, missing = false;
    for (const char* name : spec->layers) {
        if (!name) continue;
        const AtlasFrame* f = atlas.find(name);
        if (!f) { missing = true; continue; }
        drawFrame(atlas, *f, o, scale, tint);
        drewAny = true;
    }
    if (spec->tint == Tint::Glow) EndBlendMode();
    bool hasLayers = spec->layers[0] != nullptr;
    return !hasLayers || (drewAny && !missing);
}

}  // namespace

float propFloat(const gd::Object& o, int key, float def) {
    auto* v = o.prop(key);
    return v ? std::strtof(v->c_str(), nullptr) : def;
}

int propInt(const gd::Object& o, int key, int def) {
    auto* v = o.prop(key);
    return v ? std::atoi(v->c_str()) : def;
}

Color ColorTrack::eval(float camX) const {
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

Color headerColor(const std::string& header, const char* key, Color fallback) {
    size_t p = header.find(std::string(key) + ",");
    if (p == std::string::npos) return fallback;
    p += std::strlen(key) + 1;
    size_t end = header.find(',', p);
    std::string s = header.substr(p, end == std::string::npos ? std::string::npos : end - p);
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

ColorTrack buildTrack(const gd::Level& level, int triggerId, Color initial) {
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

void drawBackground(const App& app, float camX) {
    Color bgCol = app.bgTrack.eval(camX);
    ClearBackground(app.useTextures ? Color{(unsigned char)(bgCol.r / 2),
                                            (unsigned char)(bgCol.g / 2),
                                            (unsigned char)(bgCol.b / 2), 255}
                                    : Color{25, 28, 48, 255});
    // Background: screen space so it fills the window at any zoom, with a
    // 0.5x parallax scroll like the game.
    if (app.useTextures && app.bgTex.id) {
        float sw = (float)viewW(app), sh = (float)viewH(app);
        float srcH = (float)app.bgTex.height;
        float srcW = sw * srcH / sh;
        float srcX = camX * 0.5f * (srcH / sh);
        DrawTexturePro(app.bgTex, {srcX, 0, srcW, srcH}, {0, 0, sw, sh}, {0, 0}, 0, bgCol);
    }
}

void drawWorld(const App& app, const Camera2D& cam, float colorX,
               const std::function<void()>& drawActors, float ceilingY) {
    Color groundCol = app.groundTrack.eval(colorX);
    BeginMode2D(cam);
    {
        // visible world x-range for grid lines
        Vector2 tl = GetScreenToWorld2D({0, 0}, cam);
        Vector2 br = GetScreenToWorld2D({(float)viewW(app), (float)viewH(app)}, cam);
        // grid (30-unit spacing), faint
        for (float gx = std::floor(tl.x / 30) * 30; gx <= br.x; gx += 30)
            DrawLineEx({gx, tl.y}, {gx, br.y}, 1.0f / cam.zoom, Color{255, 255, 255, 10});
        for (float gy = std::floor(tl.y / 30) * 30; gy <= br.y; gy += 30)
            DrawLineEx({tl.x, gy}, {br.x, gy}, 1.0f / cam.zoom, Color{255, 255, 255, 10});

        for (const gd::Object* o : app.drawOrder) {
            if (app.useTextures) {
                bool handled = drawTexturedObject(app.atlas, *o);
                if (!handled) drawObjectShapes(*o);
                else if (app.debugMarkers && !renderSpec(o->id)->layers[0])
                    drawObjectShapes(*o);  // invisible objects: show marker
            } else {
                drawObjectShapes(*o);
            }
        }

        if (app.showHitboxes) drawHitboxes(app, cam);
        if (drawActors) drawActors();

        // Ground after objects (and actors): the game's ground layer sits at
        // a higher z than the object layer (OpenGD PlayLayer addChild z=100),
        // so anything dipping below y=0 is covered by it.
        if (app.useTextures && app.groundTex.id) {
            // 512 px = 128 units at uhd scale; scrolls 1:1 with the world.
            float gh = app.groundTex.height * kUnitsPerPx;
            DrawTexturePro(app.groundTex,
                           {tl.x / kUnitsPerPx, 0, (br.x - tl.x) / kUnitsPerPx,
                            (float)app.groundTex.height},
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
        DrawLineEx({tl.x, 0}, {br.x, 0}, 2.0f / cam.zoom, app.lineColor);

        // Flying-mode ceiling: the game's inverted ground strip, underside at
        // engine ceilingY (world y = -(ceilingY - 90)).
        if (ceilingY > 0.0f) {
            float cy = -(ceilingY - 90.0f);
            if (app.useTextures && app.groundTex.id) {
                float gh = app.groundTex.height * kUnitsPerPx;
                // negative source height flips the strip vertically
                DrawTexturePro(app.groundTex,
                               {tl.x / kUnitsPerPx, (float)app.groundTex.height,
                                (br.x - tl.x) / kUnitsPerPx,
                                -(float)app.groundTex.height},
                               {tl.x, cy - gh, br.x - tl.x, gh}, {0, 0}, 0,
                               groundCol);
                if (tl.y < cy - gh)
                    DrawRectangleV({tl.x, tl.y}, {br.x - tl.x, cy - gh - tl.y},
                                   Color{(unsigned char)(groundCol.r / 2),
                                         (unsigned char)(groundCol.g / 2),
                                         (unsigned char)(groundCol.b / 2), 255});
            } else {
                DrawRectangleV({tl.x, tl.y},
                               {br.x - tl.x, std::max(0.0f, cy - tl.y)},
                               Color{40, 44, 80, 255});
            }
            DrawLineEx({tl.x, cy}, {br.x, cy}, 2.0f / cam.zoom, app.lineColor);
        }
    }
    EndMode2D();
}

void drawHitboxes(const App& app, const Camera2D& cam) {
    // Engine boxes -> world: x unchanged, world y = -(engine y - 90).
    float lw = 1.5f / cam.zoom;
    auto rect = [](const gdcore::Box& b) {
        return Rectangle{b.left, -(b.top - 90.0f), b.right - b.left, b.top - b.bottom};
    };
    for (const auto& b : app.geom.solids) {
        Rectangle r = rect(b);
        DrawRectangleRec(r, Color{80, 200, 255, 28});
        DrawRectangleLinesEx(r, lw, Color{80, 200, 255, 230});
    }
    for (const auto& b : app.geom.hazards) {
        Rectangle r = rect(b);
        DrawRectangleRec(r, Color{255, 70, 70, 40});
        DrawRectangleLinesEx(r, lw, Color{255, 70, 70, 240});
    }
    for (const auto& p : app.geom.portals) {
        Rectangle r = rect(p.box);
        Color c = p.mode == gdcore::PortalMode::Ship   ? Color{240, 120, 220, 255}
                  : p.mode == gdcore::PortalMode::Ball ? Color{240, 90, 90, 255}
                                                       : Color{110, 240, 110, 255};
        DrawRectangleLinesEx(r, lw, c);
    }
    for (const auto& sp : app.geom.specials) {
        Rectangle r = rect(sp.box);
        Color c{};
        switch (sp.kind) {
            case gdcore::SpecialKind::YellowPad: c = Color{255, 230, 60, 255}; break;
            case gdcore::SpecialKind::GravityPad: c = Color{90, 170, 255, 255}; break;
            case gdcore::SpecialKind::GravityNormal: c = Color{90, 170, 255, 255}; break;
            case gdcore::SpecialKind::GravityFlip: c = Color{255, 230, 60, 255}; break;
            case gdcore::SpecialKind::YellowRing: c = Color{255, 200, 40, 255}; break;
            case gdcore::SpecialKind::PinkPad: c = Color{255, 120, 220, 255}; break;
            case gdcore::SpecialKind::PinkRing: c = Color{255, 120, 220, 255}; break;
        }
        DrawRectangleRec(r, Color{c.r, c.g, c.b, 60});
        DrawRectangleLinesEx(r, lw, c);
    }
    // Win line: physics ends at last object + 40 (PlayMode / env_check / gdenv).
    float wx = app.maxX + 40.0f;
    DrawLineEx({wx, -400.0f}, {wx, 90.0f}, 2.0f / cam.zoom, Color{255, 230, 80, 220});
}

bool loadLevel(App& app, const std::string& path, std::string& err) {
    gd::Level lvl;
    try {
        lvl = gd::Level::loadFromFile(path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    app.level = std::move(lvl);
    app.levelPath = path;
    // id from the file stem (Resources/levels/<id>.txt); anything else = -1
    {
        std::string stem = std::filesystem::path(path).stem().string();
        bool numeric = !stem.empty();
        for (char c : stem) numeric = numeric && c >= '0' && c <= '9';
        app.levelId = numeric ? std::atoi(stem.c_str()) : -1;
    }
    app.maxX = 0.0f;
    for (const auto& o : app.level.objects) app.maxX = std::max(app.maxX, o.x);
    app.geom = gdcore::LevelGeometry::build(app.level);

    // Draw order: z-layer (key 24, else per-id default), then z-order (key 25);
    // stable sort keeps file order on ties.
    app.drawOrder.clear();
    app.drawOrder.reserve(app.level.objects.size());
    for (const auto& o : app.level.objects) app.drawOrder.push_back(&o);
    auto zLayer = [](const gd::Object& o) {
        const RenderSpec* s = renderSpec(o.id);
        return propInt(o, 24, s ? s->defaultZ : 0);
    };
    std::stable_sort(app.drawOrder.begin(), app.drawOrder.end(),
                     [&](const gd::Object* a, const gd::Object* b) {
                         int za = zLayer(*a), zb = zLayer(*b);
                         if (za != zb) return za < zb;
                         return propInt(*a, 25, 0) < propInt(*b, 25, 0);
                     });

    app.bgTrack = buildTrack(app.level, 29,
                             headerColor(app.level.header, "kS29", {40, 62, 255, 255}));
    app.groundTrack = buildTrack(app.level, 30,
                                 headerColor(app.level.header, "kS30", {0, 19, 200, 255}));
    app.lineColor = headerColor(app.level.header, "kS31", WHITE);
    return true;
}

void loadMusic(App& app) {
    if (app.musicLoaded) {
        StopMusicStream(app.music);
        UnloadMusicStream(app.music);
        app.musicLoaded = false;
        app.music = Music{};
    }
    const LevelInfo* info = levelInfo(app.levelId);
    if (!info) return;
    std::string song = std::string(info->song) + ".mp3";
    std::filesystem::path local = std::filesystem::path("assets") / song;
    std::filesystem::path p = std::filesystem::exists(local) ? local : app.resDir / song;
    if (!std::filesystem::exists(p)) {
        TraceLog(LOG_WARNING, "%s unavailable — playing silent", song.c_str());
        return;
    }
    app.music = LoadMusicStream(p.string().c_str());
    app.musicLoaded = IsMusicValid(app.music);
    if (app.musicLoaded) app.music.looping = false;
    else TraceLog(LOG_WARNING, "%s failed to load — playing silent", song.c_str());
}

void drawFrameAt(const Atlas& atlas, const AtlasFrame& f, Vector2 worldCenter,
                 float rotationDeg, float scale, Color tint) {
    // Rotate the sprite offset (y-up local units) and negate y into screen
    // space, mirroring localToScreen without flips.
    float t = rotationDeg * DEG2RAD;
    float dx = f.offset.x * kUnitsPerPx * scale;
    float dy = f.offset.y * kUnitsPerPx * scale;
    float rx = dx * std::cos(t) + dy * std::sin(t);
    float ry = -dx * std::sin(t) + dy * std::cos(t);
    Vector2 c{worldCenter.x + rx, worldCenter.y - ry};
    float w = f.rect.width * kUnitsPerPx * scale;
    float h = f.rect.height * kUnitsPerPx * scale;
    const Texture2D& tex = atlas.texture(f.sheet);
    if (!f.rotated) {
        DrawTexturePro(tex, {f.rect.x, f.rect.y, f.rect.width, f.rect.height},
                       {c.x, c.y, w, h}, {w * 0.5f, h * 0.5f}, rotationDeg, tint);
    } else {
        DrawTexturePro(tex, {f.rect.x, f.rect.y, f.rect.height, f.rect.width},
                       {c.x, c.y, h, w}, {h * 0.5f, w * 0.5f}, rotationDeg - 90.0f, tint);
    }
}

void drawPlayer(const App& app, const gdcore::PlayerState& s, float rotationDeg,
                Color primary, Color secondary, unsigned char alpha) {
    auto fade = [alpha](Color c) {
        c.a = static_cast<unsigned char>(c.a * alpha / 255);
        return c;
    };
    Color pri = fade(primary), sec = fade(secondary);
    Vector2 c{s.x, -(s.y - 90.0f)};
    // Upside-down cube: the game mirrors the icon vertically; a half-turn is
    // the same picture for the symmetric default cube (presentation only).
    if (s.upsideDown && !s.ship && !s.ball) rotationDeg += 180.0f;
    if (app.useTextures && s.ball && app.ballFrame1) {
        // Ball: the ball icon replaces the cube (PlayerObject::createRobot
        // -style swap for toggleRollMode); it rolls, so no gravity flip of
        // the picture. Secondary layer under primary like the cube.
        if (app.ballFrame2)
            drawFrameAt(app.atlas, *app.ballFrame2, c, rotationDeg,
                        app.ballScale, sec);
        drawFrameAt(app.atlas, *app.ballFrame1, c, rotationDeg,
                    app.ballScale, pri);
    } else if (app.useTextures && s.ship && app.shipFrame1) {
        // Ship: the cube rides inside at 0.55 scale, 5 units above center
        // (PlayerObject::createShip), then the ship hull draws over it. Both
        // rotate together around the player center.
        float t = rotationDeg * DEG2RAD;
        Vector2 cc{c.x + 5.0f * std::sin(t), c.y - 5.0f * std::cos(t)};
        if (app.playerFrame1) {
            if (app.playerFrame2)
                drawFrameAt(app.atlas, *app.playerFrame2, cc, rotationDeg,
                            app.iconScale * 0.55f, sec);
            drawFrameAt(app.atlas, *app.playerFrame1, cc, rotationDeg,
                        app.iconScale * 0.55f, pri);
        }
        if (app.shipFrame2)
            drawFrameAt(app.atlas, *app.shipFrame2, c, rotationDeg,
                        app.iconScale, sec);
        drawFrameAt(app.atlas, *app.shipFrame1, c, rotationDeg,
                    app.iconScale, pri);
    } else if (app.useTextures && app.playerFrame1) {
        // Default cube: secondary layer under primary.
        if (app.playerFrame2)
            drawFrameAt(app.atlas, *app.playerFrame2, c, rotationDeg,
                        app.iconScale, sec);
        drawFrameAt(app.atlas, *app.playerFrame1, c, rotationDeg,
                    app.iconScale, pri);
    } else {
        DrawRectanglePro({c.x, c.y, 30, 30}, {15, 15}, rotationDeg, pri);
        // outline via rotated corners (screen space, clockwise rotation)
        float t = rotationDeg * DEG2RAD;
        float ct = std::cos(t), st = std::sin(t);
        auto corner = [&](float dx, float dy) {
            return Vector2{c.x + dx * ct - dy * st, c.y + dx * st + dy * ct};
        };
        Vector2 p[4] = {corner(-15, -15), corner(15, -15), corner(15, 15), corner(-15, 15)};
        for (int i = 0; i < 4; ++i)
            DrawLineEx(p[i], p[(i + 1) % 4], 2.0f, fade(Color{10, 60, 30, 255}));
    }
    if (app.showHitboxes) {
        // Axis-aligned physics boxes (rotation is cosmetic): 30x30 solid,
        // 9x9 inner death box (physics.hpp half=15 / inner=4.5).
        DrawRectangleLinesEx({c.x - 15.0f, c.y - 15.0f, 30.0f, 30.0f}, 1.0f,
                             fade(Color{255, 255, 255, 230}));
        DrawRectangleLinesEx({c.x - 4.5f, c.y - 4.5f, 9.0f, 9.0f}, 1.0f,
                             fade(Color{255, 80, 80, 230}));
    }
}

}  // namespace gdv
