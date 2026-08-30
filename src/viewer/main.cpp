// Level viewer/player: renders a decoded level with the game's own sprites,
// loaded at runtime from the GD install (never bundled), and lets you play it
// with the bit-exact gdcore physics. Falls back to colored shapes when the
// atlases are unavailable. All render tables are presentation-only —
// collision truth is gdcore::LevelGeometry (see gd-replica-notes.md).
//
// Usage: viewer [level.txt] [--res <ResourcesDir>] [--camx N] [--zoom Z]
//               [--shot out.png] [--play] [--music-offset seconds]
//               [--watch <live.bin>] [--replay <file.gdtrace>]
//   --music-offset shifts song time vs sim time (positive = skip the song
//   ahead, like the game's own ms offset setting; compensates output latency).
//   --shot renders a few frames (Browse mode), saves a screenshot, and exits
//   (for automated visual checks).
//   --watch starts in the training view: all live PPO envs as ghost cubes
//   (trainer must run with --watch to publish runs/<name>/live.bin).
//   --replay starts in the replay view: every episode in the .gdtrace
//   re-simulated in lockstep as a ghost.
// All flags are optional conveniences: the Training and Replay views pick
// their files in-GUI (O opens a runs/ browser; Training auto-follows the
// newest live.bin), H shows every keybind, F12 takes a screenshot, and
// [ / ] nudge the music offset live.
// Keys: TAB / 1 / 2 / 3 / 4 = mode (Browse / Play / Training / Replay)
//       T = textured/shapes toggle, D = debug markers (triggers, unknowns)
//       O = open file picker (Training/Replay), H = keybind help
//       [ / ] = music offset -/+ 10 ms, F12 = screenshot
//       F11 = borderless fullscreen (ESC does not quit; close the window)

#include "core/geometry.hpp"
#include "core/level.hpp"
#include "viewer/gd_frames.hpp"
#include "viewer/modes.hpp"
#include "viewer/modes_rl.hpp"
#include "viewer/world.hpp"

#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>

static const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

int main(int argc, char** argv) {
    std::string levelPath = kDefaultLevel;
    std::string shotPath, resDirOverride;
    std::string livePath, replayPath;  // empty = pick in-GUI / auto-detect
    float camX = 0.0f, zoom = 1.5f, musicOffset = 0.0f;
    int startMode = 0;  // 0 = Browse (default, keeps --shot automation), 1 = Play
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc) shotPath = argv[++i];
        else if (!std::strcmp(argv[i], "--camx") && i + 1 < argc) camX = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--res") && i + 1 < argc) resDirOverride = argv[++i];
        else if (!std::strcmp(argv[i], "--play")) startMode = 1;
        else if (!std::strcmp(argv[i], "--watch") && i + 1 < argc) { livePath = argv[++i]; startMode = 2; }
        else if (!std::strcmp(argv[i], "--replay") && i + 1 < argc) { replayPath = argv[++i]; startMode = 3; }
        else if (!std::strcmp(argv[i], "--music-offset") && i + 1 < argc) musicOffset = std::strtof(argv[++i], nullptr);
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
    InitAudioDevice();

    // Resources dir: explicit --res, else derived from the level path
    // (Resources/levels/1.txt -> Resources).
    std::filesystem::path resDir = resDirOverride.empty()
                                       ? std::filesystem::path(levelPath).parent_path().parent_path()
                                       : std::filesystem::path(resDirOverride);
    // Smaller stream buffer: less output latency and a finer
    // GetMusicTimePlayed, both matter for the sim<->song sync.
    SetAudioStreamBufferSizeDefault(1024);
    // Local copy first (assets/ is gitignored — RobTop's content), GD install
    // as fallback.
    app.musicOffset = musicOffset;
    const char* localSong = "assets/StereoMadness.mp3";
    app.music = LoadMusicStream(std::filesystem::exists(localSong)
                                    ? localSong
                                    : (resDir / "StereoMadness.mp3").string().c_str());
    app.musicLoaded = IsMusicValid(app.music);
    if (app.musicLoaded) app.music.looping = false;
    else TraceLog(LOG_WARNING, "StereoMadness.mp3 unavailable — playing silent");

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
        // Ship icon sheet (default ship). Missing sheet just means the ship
        // sections render the cube sprite.
        if (app.atlas.loadSheet((resDir / "icons" / "ship_01-uhd.plist").string())) {
            app.shipFrame1 = app.atlas.find("ship_01_001.png");
            app.shipFrame2 = app.atlas.find("ship_01_2_001.png");
        } else {
            TraceLog(LOG_WARNING, "ship icon sheet unavailable — cube-only player");
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
    gdv::TrainingViewMode training(livePath);
    gdv::ReplayMode replay(replayPath);
    gdv::Mode* modes[] = {&browse, &play, &training, &replay};
    constexpr int kModeCount = 4;
    int cur = startMode;
    modes[cur]->enter(app);

    // ESC is the picker's cancel key, not an app-quit shortcut.
    SetExitKey(KEY_NULL);

    bool helpOpen = false;
    float offsetNote = 0.0f;  // seconds left showing the music-offset toast
    int frame = 0;
    while (!WindowShouldClose()) {
        int want = cur;
        if (IsKeyPressed(KEY_TAB)) want = (cur + 1) % kModeCount;
        if (IsKeyPressed(KEY_ONE)) want = 0;
        if (IsKeyPressed(KEY_TWO)) want = 1;
        if (IsKeyPressed(KEY_THREE)) want = 2;
        if (IsKeyPressed(KEY_FOUR)) want = 3;
        if (want != cur) { modes[cur]->exit(app); cur = want; modes[cur]->enter(app); }
        if (IsKeyPressed(KEY_T) && app.textured) app.useTextures = !app.useTextures;
        if (IsKeyPressed(KEY_D)) app.debugMarkers = !app.debugMarkers;
        if (IsKeyPressed(KEY_F11)) ToggleBorderlessWindowed();
        if (IsKeyPressed(KEY_H)) helpOpen = !helpOpen;
        if (IsKeyPressed(KEY_LEFT_BRACKET)) { app.musicOffset -= 0.010f; offsetNote = 1.5f; }
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) { app.musicOffset += 0.010f; offsetNote = 1.5f; }
        if (IsKeyPressed(KEY_F12)) {
            // Relative path: raylib's TakeScreenshot prepends the CWD itself.
            TakeScreenshot(TextFormat("shot_%lld.png", (long long)time(nullptr)));
        }

        if (app.musicLoaded) UpdateMusicStream(app.music);
        modes[cur]->update(app);

        BeginDrawing();
        modes[cur]->draw(app);
        DrawText(TextFormat("[%s]   TAB/1-4: mode   H: help   T: textures   D: debug   F11: fullscreen",
                            modes[cur]->name()),
                 12, GetScreenHeight() - 24, 16, Color{170, 175, 200, 255});
        if (offsetNote > 0.0f) {
            offsetNote -= GetFrameTime();
            const char* txt = TextFormat("music offset %+d ms",
                                         (int)std::lround(app.musicOffset * 1000.0f));
            DrawText(txt, GetScreenWidth() - MeasureText(txt, 18) - 16, 12, 18, RAYWHITE);
        }
        if (helpOpen) {
            static const char* kHelp[] = {
                "TAB / 1-4      mode: Browse / Play / Training / Replay",
                "T              textures <-> colored shapes",
                "D              debug markers (triggers, unknown ids)",
                "F11            borderless fullscreen",
                "F12            screenshot (shot_*.png in the working dir)",
                "[ / ]          music offset -/+ 10 ms",
                "H              toggle this help",
                "",
                "Browse:   drag = pan   wheel = zoom   SPACE = autoscroll   R = reset camera",
                "Play:     SPACE / UP / click = jump (hold to buffer)   R = restart",
                "Training: O = pick run to watch   F = follow / free camera",
                "Replay:   O = open trace   SPACE = pause   LEFT/RIGHT = speed",
                "          R = restart   F = follow / free camera",
                "",
                "Picker:   arrows / wheel = select   ENTER / click = open   ESC = cancel",
            };
            constexpr int kHelpLines = sizeof(kHelp) / sizeof(kHelp[0]);
            int w = 640, h = 30 + kHelpLines * 24 + 16;
            int x = (GetScreenWidth() - w) / 2, y = (GetScreenHeight() - h) / 2;
            DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), Color{0, 0, 0, 140});
            DrawRectangle(x, y, w, h, Color{24, 26, 38, 245});
            DrawRectangleLines(x, y, w, h, Color{110, 120, 170, 255});
            DrawText("Keybinds", x + 16, y + 12, 20, RAYWHITE);
            for (int i = 0; i < kHelpLines; ++i)
                DrawText(kHelp[i], x + 16, y + 42 + i * 24, 18, Color{185, 190, 210, 255});
        }
        EndDrawing();

        if (!shotPath.empty() && ++frame >= 5) {
            TakeScreenshot(shotPath.c_str());
            break;
        }
    }
    if (app.bgTex.id) UnloadTexture(app.bgTex);
    if (app.groundTex.id) UnloadTexture(app.groundTex);
    if (app.musicLoaded) UnloadMusicStream(app.music);
    CloseAudioDevice();
    app.atlas.unloadAll();
    CloseWindow();
    return 0;
}
