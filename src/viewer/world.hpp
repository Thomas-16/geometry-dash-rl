// Shared viewer context + world renderer, extracted from main.cpp so several
// modes (Browse, Play, Training, Replay) can draw the same level. All
// tables here are presentation-only — collision truth is gdcore::LevelGeometry.
#pragma once

#include "core/geometry.hpp"
#include "core/level.hpp"
#include "core/physics.hpp"
#include "viewer/atlas.hpp"

#include "raylib.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace gdv {

inline constexpr float kUnitsPerPx = 0.25f;   // uhd sheets: 4 px per GD unit
inline constexpr float kCubeSpeed = 311.58f;  // GD 1x speed, units/sec (approx)

// Piecewise color timeline built from the header color + 1.0 color triggers.
struct ColorTrack {
    struct Key {
        float x;
        Color target;
        float durUnits;  // trigger duration converted to world-x units
    };
    Color initial{};
    std::vector<Key> keys;

    Color eval(float camX) const;
};

// Parses "1_R_2_G_3_B_..." after "<key>," in the level header (kS29/kS30/...).
Color headerColor(const std::string& header, const char* key, Color fallback);
ColorTrack buildTrack(const gd::Level& level, int triggerId, Color initial);

float propFloat(const gd::Object& o, int key, float def);
int propInt(const gd::Object& o, int key, int def);

// Cross-tab requests from the GUI panel, consumed by main() once per frame
// before the active mode updates (the panel never reaches into mode state).
struct Commands {
    int switchMode = -1;         // 0..3, -1 = none
    std::string loadLevel;       // level file path
    std::string replayTrace;     // .gdtrace -> Replay mode
    std::string replaySolution;  // solution.json -> Replay mode
    std::string watchLive;       // live.bin -> Training mode
    bool screenshot = false;
    bool fullscreen = false;
};

// Everything the modes share: level, draw order, collision geometry, atlases,
// color tracks, and the global render toggles. Built once in main().
struct App {
    gd::Level level;
    std::vector<const gd::Object*> drawOrder;  // z-sorted
    gdcore::LevelGeometry geom;
    Atlas atlas;
    Texture2D bgTex{}, groundTex{};
    ColorTrack bgTrack, groundTrack;
    Color lineColor = WHITE;
    float maxX = 0;
    int levelId = 1;            // official level id (-1 = custom file)
    std::string levelPath;      // file the level was loaded from
    std::filesystem::path resDir;  // GD Resources dir (sheets, songs)
    bool textured = false;      // atlases loaded ok
    bool useTextures = false;   // textures <-> colored shapes
    bool debugMarkers = false;  // show triggers / invisible objects
    bool showHitboxes = false;  // gdcore collision boxes + player box + win line
    bool livePreview = true;    // Training view polls/draws live.bin; off = UI only, trainer keeps the GPU
    const AtlasFrame* playerFrame1 = nullptr;  // player_01_001.png (primary)
    const AtlasFrame* playerFrame2 = nullptr;  // player_01_2_001.png (secondary)
    const AtlasFrame* shipFrame1 = nullptr;    // ship_01_001.png (primary)
    const AtlasFrame* shipFrame2 = nullptr;    // ship_01_2_001.png (secondary)
    float iconScale = 1.0f;     // scales the icon sprite to span 30 units
    Music music{};              // <song>.mp3 stream (owned by main)
    bool musicLoaded = false;
    float musicOffset = 0.0f;   // song time = sim time + offset

    // Game viewport (window minus the GUI panel), anchored at (0,0) so mouse
    // -> world math is unchanged. Set by the GUI every frame.
    int viewW = 0, viewH = 0;
    bool guiWantMouse = false;     // panel hovered / dragging a widget
    bool guiWantKeyboard = false;  // a text field has focus
    Commands commands;
};

// Window-independent viewport size (falls back to the screen before the GUI
// has run once, e.g. --shot).
inline int viewW(const App& a) { return a.viewW > 0 ? a.viewW : GetScreenWidth(); }
inline int viewH(const App& a) { return a.viewH > 0 ? a.viewH : GetScreenHeight(); }

// Replaces the level in `app` (objects, draw order, geometry, color tracks,
// maxX, levelId/levelPath). On failure `app` is left untouched.
bool loadLevel(App& app, const std::string& path, std::string& err);
// (Re)loads the song for app.levelId: assets/<song>.mp3 first (gitignored
// local copy), then <resDir>/<song>.mp3. Silent when neither exists.
void loadMusic(App& app);

// ClearBackground + screen-space parallax background. camX drives both the
// 0.5x parallax scroll and the BG color-trigger evaluation.
void drawBackground(const App& app, float camX);

// BeginMode2D..EndMode2D: grid, objects, actors, ground, ground line.
// drawActors runs after the objects and before the ground quad — the game's
// draw order (ground layer z=100 covers anything dipping below y=0). It is
// the hook for player/ghost rendering; a training view passes a callback
// that draws N players.
// ceilingY > 0 also draws the flying-mode ceiling (an upside-down ground
// strip) with its underside at that engine y (ship band: 390).
void drawWorld(const App& app, const Camera2D& cam, float colorX,
               const std::function<void()>& drawActors = {},
               float ceilingY = 0.0f);

// Collision overlay (inside Mode2D): solids, hazards, portals and the win
// line at maxX + 40. Called by drawWorld when app.showHitboxes.
void drawHitboxes(const App& app, const Camera2D& cam);

// Draws one atlas frame at an arbitrary world point (screen coords, y-down),
// no flips. Rotation is clockwise degrees, like the object path.
void drawFrameAt(const Atlas& atlas, const AtlasFrame& f, Vector2 worldCenter,
                 float rotationDeg, float scale, Color tint);

// Cube at engine coords: world = {s.x, -(s.y - 90)} (engine y = level y + 90,
// screen y = -level y). Sprite when available and textures on, else a colored
// rect. rotationDeg is cosmetic-only — never physics. primary/secondary tint
// the two icon layers (defaults = the classic green/cyan); alpha multiplies
// both, letting training/replay views draw translucent ghosts. With
// app.showHitboxes the 30x30 solid box and 9x9 inner death box are outlined.
void drawPlayer(const App& app, const gdcore::PlayerState& s, float rotationDeg,
                Color primary = {0, 255, 0, 255},
                Color secondary = {0, 255, 255, 255},
                unsigned char alpha = 255);

}  // namespace gdv
