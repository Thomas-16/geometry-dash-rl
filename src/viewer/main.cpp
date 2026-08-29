// Level viewer/player: renders a decoded level with the game's own sprites,
// loaded at runtime from the GD install (never bundled), and lets you play it
// with the bit-exact gdcore physics. Falls back to colored shapes when the
// atlases are unavailable. All render tables are presentation-only —
// collision truth is gdcore::LevelGeometry (see gd-replica-notes.md).
//
// Usage: viewer [level.txt] [--res <ResourcesDir>] [--camx N] [--zoom Z]
//               [--shot out.png] [--play]
//   --shot renders a few frames (Browse mode), saves a screenshot, and exits
//   (for automated visual checks).
// Keys: TAB / 1 / 2 = switch mode (Browse / Play)
//       T = textured/shapes toggle, D = debug markers (triggers, unknowns)
//       F11 = borderless fullscreen

#include "core/geometry.hpp"
#include "core/level.hpp"
#include "viewer/gd_frames.hpp"
#include "viewer/modes.hpp"
#include "viewer/world.hpp"

#include "raylib.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>

static const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

int main(int argc, char** argv) {
    std::string levelPath = kDefaultLevel;
    std::string shotPath, resDirOverride;
    float camX = 0.0f, zoom = 1.5f;
    int startMode = 0;  // 0 = Browse (default, keeps --shot automation), 1 = Play
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc) shotPath = argv[++i];
        else if (!std::strcmp(argv[i], "--camx") && i + 1 < argc) camX = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--res") && i + 1 < argc) resDirOverride = argv[++i];
        else if (!std::strcmp(argv[i], "--play")) startMode = 1;
        else levelPath = argv[i];
    }

    gdv::App app;
    try {
        app.level = gd::Level::loadFromFile(levelPath);
    } catch (const std::exception& e) {
        TraceLog(LOG_ERROR, "failed to load level: %s", e.what());
        return 1;
    }
    for (const auto& o : app.level.objects) app.maxX = std::max(app.maxX, o.x);
    app.geom = gdcore::LevelGeometry::build(app.level);

    // Draw order: z-layer (key 24, else per-id default), then z-order (key 25);
    // stable sort keeps file order on ties.
    app.drawOrder.reserve(app.level.objects.size());
    for (const auto& o : app.level.objects) app.drawOrder.push_back(&o);
    auto zLayer = [](const gd::Object& o) {
        const gdv::RenderSpec* s = gdv::renderSpec(o.id);
        return gdv::propInt(o, 24, s ? s->defaultZ : 0);
    };
    std::stable_sort(app.drawOrder.begin(), app.drawOrder.end(),
                     [&](const gd::Object* a, const gd::Object* b) {
                         int za = zLayer(*a), zb = zLayer(*b);
                         if (za != zb) return za < zb;
                         return gdv::propInt(*a, 25, 0) < gdv::propInt(*b, 25, 0);
                     });

    app.bgTrack = gdv::buildTrack(app.level, 29,
                                  gdv::headerColor(app.level.header, "kS29", {40, 62, 255, 255}));
    app.groundTrack = gdv::buildTrack(app.level, 30,
                                      gdv::headerColor(app.level.header, "kS30", {0, 19, 200, 255}));
    app.lineColor = gdv::headerColor(app.level.header, "kS31", WHITE);

    const int W = 1600, H = 900;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(W, H, "gd-rl viewer — Stereo Madness");
    SetTargetFPS(60);

    // Resources dir: explicit --res, else derived from the level path
    // (Resources/levels/1.txt -> Resources).
    std::filesystem::path resDir = resDirOverride.empty()
                                       ? std::filesystem::path(levelPath).parent_path().parent_path()
                                       : std::filesystem::path(resDirOverride);
    app.textured = app.atlas.loadSheet((resDir / "GJ_GameSheet-uhd.plist").string()) &&
                   app.atlas.loadSheet((resDir / "GJ_GameSheet02-uhd.plist").string());
    if (app.textured) {
        // Validate the whole id->frame table up front so mapping mistakes are
        // loud instead of silently falling back per object.
        for (int id : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                       22, 23, 27, 29, 30, 39, 40, 41, 54, 62, 65, 103, 142}) {
            const gdv::RenderSpec* s = gdv::renderSpec(id);
            if (!s) continue;
            for (const char* name : s->layers)
                if (name && !app.atlas.find(name))
                    TraceLog(LOG_WARNING, "no atlas frame %s for id %d", name, id);
        }
        // Player icon sheet (default cube). Missing sheet just means the rect
        // fallback player.
        if (app.atlas.loadSheet((resDir / "icons" / "player_01-uhd.plist").string())) {
            app.playerFrame1 = app.atlas.find("player_01_001.png");
            app.playerFrame2 = app.atlas.find("player_01_2_001.png");
            if (app.playerFrame1) {
                // Icon frames carry padding; scale so the sprite spans the
                // cube's 30 units (cosmetic — hitbox is always 30x30).
                float srcPx = std::max(app.playerFrame1->sourceSize.x,
                                       app.playerFrame1->sourceSize.y);
                if (srcPx > 0) app.iconScale = 30.0f / (srcPx * gdv::kUnitsPerPx);
            }
        } else {
            TraceLog(LOG_WARNING, "player icon sheet unavailable — rect player");
        }
    } else {
        TraceLog(LOG_WARNING, "atlases unavailable — falling back to shape rendering");
        app.atlas.unloadAll();
    }
    app.useTextures = app.textured;

    if (app.textured) {
        app.bgTex = LoadTexture((resDir / "game_bg_01_001-uhd.png").string().c_str());
        app.groundTex = LoadTexture((resDir / "groundSquare_01_001-uhd.png").string().c_str());
        if (app.bgTex.id) SetTextureWrap(app.bgTex, TEXTURE_WRAP_REPEAT);
        if (app.groundTex.id) SetTextureWrap(app.groundTex, TEXTURE_WRAP_REPEAT);
    }

    gdv::BrowseMode browse(camX, zoom);
    gdv::PlayMode play;
    gdv::Mode* modes[] = {&browse, &play};
    constexpr int kModeCount = 2;
    int cur = startMode;
    modes[cur]->enter(app);

    int frame = 0;
    while (!WindowShouldClose()) {
        int want = cur;
        if (IsKeyPressed(KEY_TAB)) want = (cur + 1) % kModeCount;
        if (IsKeyPressed(KEY_ONE)) want = 0;
        if (IsKeyPressed(KEY_TWO)) want = 1;
        if (want != cur) { cur = want; modes[cur]->enter(app); }
        if (IsKeyPressed(KEY_T) && app.textured) app.useTextures = !app.useTextures;
        if (IsKeyPressed(KEY_D)) app.debugMarkers = !app.debugMarkers;
        if (IsKeyPressed(KEY_F11)) ToggleBorderlessWindowed();

        modes[cur]->update(app);

        BeginDrawing();
        modes[cur]->draw(app);
        DrawText(TextFormat("[%s]   TAB/1/2: mode   T: textures   D: debug   F11: fullscreen",
                            modes[cur]->name()),
                 12, GetScreenHeight() - 24, 16, Color{170, 175, 200, 255});
        EndDrawing();

        if (!shotPath.empty() && ++frame >= 5) {
            TakeScreenshot(shotPath.c_str());
            break;
        }
    }
    if (app.bgTex.id) UnloadTexture(app.bgTex);
    if (app.groundTex.id) UnloadTexture(app.groundTex);
    app.atlas.unloadAll();
    CloseWindow();
    return 0;
}
