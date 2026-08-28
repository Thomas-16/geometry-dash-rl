// Untextured level viewer: renders a decoded level as colored shapes.
// This shape table is presentation-only — collision/hitbox data comes later
// from the binary, per gd-replica-notes.md.
//
// Usage: viewer [level.txt] [--camx N] [--zoom Z] [--shot out.png]
//   --shot renders a few frames, saves a screenshot, and exits (for
//   automated visual checks).

#include "core/level.hpp"

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

static const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

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

static void drawObject(const gd::Object& o) {
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

int main(int argc, char** argv) {
    std::string levelPath = kDefaultLevel;
    std::string shotPath;
    float camX = 0.0f, zoom = 1.5f;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc) shotPath = argv[++i];
        else if (!std::strcmp(argv[i], "--camx") && i + 1 < argc) camX = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = std::strtof(argv[++i], nullptr);
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

    const int W = 1600, H = 900;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(W, H, "gd-rl viewer — Stereo Madness");
    SetTargetFPS(60);

    Camera2D cam{};
    cam.zoom = zoom;
    cam.target = {camX, -180.0f};
    bool autoScroll = false;
    const float kCubeSpeed = 311.58f; // GD 1x speed, units/sec (approx, cosmetic)

    int frame = 0;
    while (!WindowShouldClose()) {
        cam.offset = {GetScreenWidth() * 0.25f, GetScreenHeight() * 0.6f};

        if (IsKeyPressed(KEY_SPACE)) autoScroll = !autoScroll;
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

        BeginDrawing();
        ClearBackground(Color{25, 28, 48, 255});

        BeginMode2D(cam);
        {
            // visible world x-range for grid lines
            Vector2 tl = GetScreenToWorld2D({0, 0}, cam);
            Vector2 br = GetScreenToWorld2D({(float)GetScreenWidth(), (float)GetScreenHeight()}, cam);
            // grid (30-unit spacing), faint
            for (float gx = std::floor(tl.x / 30) * 30; gx <= br.x; gx += 30)
                DrawLineEx({gx, tl.y}, {gx, br.y}, 1.0f / cam.zoom, Color{255, 255, 255, 10});
            for (float gy = std::floor(tl.y / 30) * 30; gy <= br.y; gy += 30)
                DrawLineEx({tl.x, gy}, {br.x, gy}, 1.0f / cam.zoom, Color{255, 255, 255, 10});
            // ground: floor surface is world y=0 (screen y=0), drawn solid below
            DrawRectangleV({tl.x, 0}, {br.x - tl.x, std::max(0.0f, br.y)}, Color{40, 44, 80, 255});
            DrawLineEx({tl.x, 0}, {br.x, 0}, 2.0f / cam.zoom, Color{160, 170, 255, 255});

            for (const auto& o : level.objects) drawObject(o);
        }
        EndMode2D();

        DrawText(TextFormat("objects: %d   cam.x: %.0f / %.0f   zoom: %.2f   %d fps",
                            (int)level.objects.size(), cam.target.x, maxX,
                            cam.zoom, GetFPS()),
                 12, 10, 20, RAYWHITE);
        DrawText("drag: pan   wheel: zoom   space: auto-scroll   R: reset",
                 12, 34, 16, Color{170, 175, 200, 255});
        EndDrawing();

        if (!shotPath.empty() && ++frame >= 5) {
            TakeScreenshot(shotPath.c_str());
            break;
        }
    }
    CloseWindow();
    return 0;
}
