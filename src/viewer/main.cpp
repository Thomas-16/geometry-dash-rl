// Level viewer/player + RL control panel: renders a decoded level with the
// game's own sprites, loaded at runtime from the GD install (never bundled),
// lets you play it with the bit-exact gdcore physics, replays recorded runs,
// watches live training, and drives the whole train -> eval -> deploy loop
// from a Dear ImGui panel on the right (no second terminal needed). Falls
// back to colored shapes when the atlases are unavailable. All render tables
// are presentation-only — collision truth is gdcore::LevelGeometry.
//
// Usage: viewer [level.txt] [--res <ResourcesDir>] [--camx N] [--zoom Z]
//               [--shot out.png] [--play] [--music-offset seconds]
//               [--watch <live.bin>] [--replay <file.gdtrace|solution.json>]
//               [--panel] [--tab N]
//   --music-offset shifts song time vs sim time (positive = skip the song
//   ahead, like the game's own ms offset setting; compensates output latency).
//   --shot renders a few frames (Browse mode, panel hidden), saves a
//   screenshot, and exits (for automated visual checks); --panel keeps the
//   panel in the capture and --tab N picks its tab (0 View .. 4 Jobs).
//   --watch starts in the training view: all live PPO envs as ghost cubes
//   (trainer must run with --watch to publish runs/<name>/live.bin; the
//   panel's Train tab always passes it).
//   --replay starts in the replay view: every episode in the .gdtrace (or the
//   single solution.json run) re-simulated in lockstep as a ghost.
// Everything is reachable from the panel: mode buttons, level picker, view
// toggles, replay/training file lists, training form + live status, runs
// table (eval / deploy / check trace / resume / delete), tools and job logs.
// Hotkeys are optional shortcuts (listed in the panel's "Keyboard shortcuts").
// Paths: repo root is found from the exe location (build/viewer.exe); prefs
// persist in <repo>/viewer_prefs.ini.

#include "viewer/gui.hpp"
#include "viewer/input.hpp"
#include "viewer/jobs.hpp"
#include "viewer/levels.hpp"
#include "viewer/modes.hpp"
#include "viewer/modes_rl.hpp"
#include "viewer/paths.hpp"
#include "viewer/prefs.hpp"
#include "viewer/world.hpp"

#include "raylib.h"
#include "rlImGui.h"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>

namespace {

void loadSheets(gdv::App& app) {
    const std::filesystem::path& resDir = app.resDir;
    app.atlas.unloadAll();
    app.playerFrame1 = app.playerFrame2 = app.shipFrame1 = app.shipFrame2 = nullptr;
    app.ballFrame1 = app.ballFrame2 = nullptr;
    app.birdFrame1 = app.birdFrame2 = app.birdFrame3 = nullptr;
    app.textured = app.atlas.loadSheet((resDir / "GJ_GameSheet-uhd.plist").string()) &&
                   app.atlas.loadSheet((resDir / "GJ_GameSheet02-uhd.plist").string());
    if (app.textured) {
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
        // Ball icon sheet (default ball, Cycles). Missing sheet just means
        // the ball sections render the cube sprite.
        if (app.atlas.loadSheet((resDir / "icons" / "player_ball_01-uhd.plist").string())) {
            app.ballFrame1 = app.atlas.find("player_ball_01_001.png");
            app.ballFrame2 = app.atlas.find("player_ball_01_2_001.png");
            if (app.ballFrame1) {
                float srcPx = std::max(app.ballFrame1->sourceSize.x,
                                       app.ballFrame1->sourceSize.y);
                if (srcPx > 0) app.ballScale = 30.0f / (srcPx * gdv::kUnitsPerPx);
            }
        } else {
            TraceLog(LOG_WARNING, "ball icon sheet unavailable — cube sprite in ball sections");
        }
        // UFO icon sheet (default UFO, Theory of Everything). Missing sheet
        // just means the UFO sections render the cube sprite.
        if (app.atlas.loadSheet((resDir / "icons" / "bird_01-uhd.plist").string())) {
            app.birdFrame1 = app.atlas.find("bird_01_001.png");
            app.birdFrame2 = app.atlas.find("bird_01_2_001.png");
            app.birdFrame3 = app.atlas.find("bird_01_3_001.png");
            if (app.birdFrame1) {
                // the hull frame is ~147 px wide: span the 30-unit box
                float srcPx = std::max(app.birdFrame1->sourceSize.x,
                                       app.birdFrame1->sourceSize.y);
                if (srcPx > 0) app.birdScale = 30.0f / (srcPx * gdv::kUnitsPerPx);
            }
        } else {
            TraceLog(LOG_WARNING, "UFO icon sheet unavailable — cube sprite in UFO sections");
        }
    } else {
        TraceLog(LOG_WARNING, "atlases unavailable — falling back to shape rendering");
        app.atlas.unloadAll();
    }
    app.useTextures = app.textured;
    if (app.bgTex.id) UnloadTexture(app.bgTex);
    if (app.groundTex.id) UnloadTexture(app.groundTex);
    app.bgTex = Texture2D{};
    app.groundTex = Texture2D{};
    if (app.textured) {
        app.bgTex = LoadTexture((resDir / "game_bg_01_001-uhd.png").string().c_str());
        app.groundTex = LoadTexture((resDir / "groundSquare_01_001-uhd.png").string().c_str());
        if (app.bgTex.id) SetTextureWrap(app.bgTex, TEXTURE_WRAP_REPEAT);
        if (app.groundTex.id) SetTextureWrap(app.groundTex, TEXTURE_WRAP_REPEAT);
    }
}

void setTitle(const gdv::App& app) {
    const gdv::LevelInfo* info = gdv::levelInfo(app.levelId);
    std::string title = "gd-rl viewer — ";
    title += info ? info->name : std::filesystem::path(app.levelPath).filename().string();
    SetWindowTitle(title.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    gdv::paths::init();
    gdv::Prefs prefs;
    prefs.load(gdv::paths::repoRoot() / "viewer_prefs.ini");
    gdv::paths::setResourcesDir(prefs.get("resources_dir"));
    gdv::paths::setPythonExe(prefs.get("python_exe"));

    std::string levelPath;
    std::string shotPath, resDirOverride;
    std::string livePath, replayPath;  // empty = pick in-GUI / auto-detect
    float camX = 0.0f, zoom = 1.5f;
    float musicOffset = prefs.getFloat("music_offset", 0.0f);
    int startMode = 0;  // 0 = Browse (default, keeps --shot automation), 1 = Play
    bool keepPanel = false;  // --panel: keep the GUI panel in --shot captures
    int startTab = -1;       // --tab N: select a panel tab (View/Train/Runs/Tools/Jobs)
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc) shotPath = argv[++i];
        else if (!std::strcmp(argv[i], "--camx") && i + 1 < argc) camX = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--res") && i + 1 < argc) resDirOverride = argv[++i];
        else if (!std::strcmp(argv[i], "--play")) startMode = 1;
        else if (!std::strcmp(argv[i], "--panel")) keepPanel = true;
        else if (!std::strcmp(argv[i], "--tab") && i + 1 < argc) startTab = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--watch") && i + 1 < argc) { livePath = argv[++i]; startMode = 2; }
        else if (!std::strcmp(argv[i], "--replay") && i + 1 < argc) { replayPath = argv[++i]; startMode = 3; }
        else if (!std::strcmp(argv[i], "--music-offset") && i + 1 < argc) musicOffset = std::strtof(argv[++i], nullptr);
        else levelPath = argv[i];
    }
    if (!resDirOverride.empty()) gdv::paths::setResourcesDir(resDirOverride);
    if (levelPath.empty()) {
        int id = prefs.getInt("level_id", 1);
        if (!gdv::levelInfo(id)) id = 1;
        levelPath = gdv::paths::levelPath(id).generic_string();
    }

    gdv::App app;
    {
        std::string err;
        if (!gdv::loadLevel(app, levelPath, err)) {
            TraceLog(LOG_ERROR, "failed to load level: %s", err.c_str());
            return 1;
        }
    }
    // Resources dir: explicit --res / prefs, else derived from the level path
    // (Resources/levels/1.txt -> Resources) when that looks like a GD install.
    app.resDir = gdv::paths::resourcesDir();
    {
        std::filesystem::path derived = std::filesystem::path(levelPath).parent_path().parent_path();
        std::error_code ec;
        if (resDirOverride.empty() && std::filesystem::exists(derived / "GJ_GameSheet-uhd.plist", ec) &&
            !std::filesystem::exists(app.resDir / "GJ_GameSheet-uhd.plist", ec)) {
            app.resDir = derived;
            gdv::paths::setResourcesDir(derived);
        }
    }

    const int W = 1600, H = 900;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(W, H, "gd-rl viewer");
    SetTargetFPS(60);
    InitAudioDevice();
    setTitle(app);

    // Smaller stream buffer: less output latency and a finer
    // GetMusicTimePlayed, both matter for the sim<->song sync.
    SetAudioStreamBufferSizeDefault(1024);
    app.musicOffset = musicOffset;
    gdv::loadMusic(app);
    loadSheets(app);
    app.useTextures = app.textured && prefs.getBool("textures", true);
    app.showHitboxes = prefs.getBool("hitboxes", false);
    app.livePreview = prefs.getBool("live_preview", true);

    gdv::Jobs jobs;
    gdv::Gui gui(prefs, jobs);
    gui.init(app);
    jobs.onExit = [&gui](const gdv::JobInfo& info) { gui.jobExited(info); };
    if (!shotPath.empty() && !keepPanel) gui.setPanelVisible(false);
    if (startTab >= 0) gui.selectTab(startTab);
    // A stale viewer.exe.old from "Rebuild incl. viewer" can go now.
    {
        std::error_code ec;
        std::filesystem::remove(gdv::paths::buildDir() / "viewer.exe.old", ec);
    }

    gdv::BrowseMode browse(camX, zoom);
    gdv::PlayMode play;
    gdv::TrainingViewMode training(livePath);
    gdv::ReplayMode replay(replayPath);
    gdv::Mode* modes[] = {&browse, &play, &training, &replay};
    constexpr int kModeCount = 4;
    int cur = startMode;
    modes[cur]->enter(app);

    auto switchMode = [&](int want) {
        if (want == cur || want < 0 || want >= kModeCount) return;
        modes[cur]->exit(app);
        cur = want;
        modes[cur]->enter(app);
    };

    // ESC is ImGui's cancel key, not an app-quit shortcut.
    SetExitKey(KEY_NULL);

    int frame = 0;
    int curFps = 60;  // matches the SetTargetFPS(60) above
    while (!WindowShouldClose()) {
        gui.beginFrame(app);

        // Global hotkeys (optional shortcuts for panel buttons).
        if (gdv::keyPressed(app, KEY_TAB)) app.commands.switchMode = (cur + 1) % kModeCount;
        if (gdv::keyPressed(app, KEY_ONE)) app.commands.switchMode = 0;
        if (gdv::keyPressed(app, KEY_TWO)) app.commands.switchMode = 1;
        if (gdv::keyPressed(app, KEY_THREE)) app.commands.switchMode = 2;
        if (gdv::keyPressed(app, KEY_FOUR)) app.commands.switchMode = 3;
        if (gdv::keyPressed(app, KEY_T) && app.textured) app.useTextures = !app.useTextures;
        if (gdv::keyPressed(app, KEY_D)) app.debugMarkers = !app.debugMarkers;
        if (gdv::keyPressed(app, KEY_B)) app.showHitboxes = !app.showHitboxes;
        if (gdv::keyPressed(app, KEY_H)) gui.setPanelVisible(!gui.panelVisible());
        if (gdv::keyPressed(app, KEY_F11)) app.commands.fullscreen = true;
        if (gdv::keyPressed(app, KEY_LEFT_BRACKET)) app.musicOffset -= 0.010f;
        if (gdv::keyPressed(app, KEY_RIGHT_BRACKET)) app.musicOffset += 0.010f;
        if (gdv::keyPressed(app, KEY_F12)) app.commands.screenshot = true;

        // Panel requests.
        gdv::Commands cmd = app.commands;
        app.commands = gdv::Commands{};
        if (!cmd.loadLevel.empty()) {
            std::string err;
            modes[cur]->exit(app);
            if (gdv::loadLevel(app, cmd.loadLevel, err)) {
                gdv::loadMusic(app);
                setTitle(app);
                for (gdv::Mode* m : modes) m->onLevelChanged(app);
                gui.toast(std::string("loaded ") + std::filesystem::path(cmd.loadLevel).filename().string());
            } else {
                gui.toast("level load failed: " + err, 10);
            }
            modes[cur]->enter(app);
        }
        if (!cmd.replayTrace.empty()) { switchMode(3); replay.openTrace(cmd.replayTrace); }
        if (!cmd.replaySolution.empty()) { switchMode(3); replay.openSolution(cmd.replaySolution); }
        if (!cmd.watchLive.empty()) training.watch(cmd.watchLive);
        if (cmd.switchMode >= 0) switchMode(cmd.switchMode);
        if (cmd.fullscreen) ToggleBorderlessWindowed();
        if (cmd.screenshot) {
            // Relative path: raylib's TakeScreenshot prepends the CWD itself.
            const char* name = TextFormat("shot_%lld.png", (long long)time(nullptr));
            TakeScreenshot(name);
            gui.toast(std::string("saved ") + name);
        }

        // Frame pacing: the mode's rate (Play wants 240 for input resolution),
        // halved while a trainer/eval runs and the window is in the background
        // so the viewer stops competing with it for GPU and CPU time.
        int wantFps = modes[cur]->targetFps(app);
        bool background = jobs.anyPythonRunning() && !IsWindowFocused();
        if (background) wantFps = wantFps < 30 ? wantFps : 30;
        if (background && !app.livePreview) wantFps = 15;  // nothing animates: panel only
        if (wantFps != curFps) {
            SetTargetFPS(wantFps);
            curFps = wantFps;
        }

        if (app.musicLoaded) UpdateMusicStream(app.music);
        modes[cur]->update(app);

        BeginDrawing();
        BeginScissorMode(0, 0, app.viewW, app.viewH);
        modes[cur]->draw(app);
        EndScissorMode();
        rlImGuiBegin();
        gui.draw(app, modes, kModeCount, cur);
        rlImGuiEnd();
        EndDrawing();

        if (!shotPath.empty() && ++frame >= 5) {
            TakeScreenshot(shotPath.c_str());
            break;
        }
    }
    modes[cur]->exit(app);
    if (shotPath.empty()) gui.savePrefs(app);  // --shot is automation: never persist
    gui.shutdown();
    if (app.bgTex.id) UnloadTexture(app.bgTex);
    if (app.groundTex.id) UnloadTexture(app.groundTex);
    if (app.musicLoaded) UnloadMusicStream(app.music);
    CloseAudioDevice();
    app.atlas.unloadAll();
    CloseWindow();
    return 0;
}
