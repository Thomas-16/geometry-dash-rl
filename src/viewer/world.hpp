// Shared viewer context + world renderer, extracted from main.cpp so several
// modes (Browse, Play, future TrainingView) can draw the same level. All
// tables here are presentation-only — collision truth is gdcore::LevelGeometry.
#pragma once

#include "core/geometry.hpp"
#include "core/level.hpp"
#include "core/physics.hpp"
#include "viewer/atlas.hpp"

#include "raylib.h"

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
    bool textured = false;      // atlases loaded ok
    bool useTextures = false;   // T toggle
    bool debugMarkers = false;  // D toggle
    const AtlasFrame* playerFrame1 = nullptr;  // player_01_001.png (primary)
    const AtlasFrame* playerFrame2 = nullptr;  // player_01_2_001.png (secondary)
    float iconScale = 1.0f;     // scales the icon sprite to span 30 units
};

// ClearBackground + screen-space parallax background. camX drives both the
// 0.5x parallax scroll and the BG color-trigger evaluation.
void drawBackground(const App& app, float camX);

// BeginMode2D..EndMode2D: grid, objects, actors, ground, ground line.
// drawActors runs after the objects and before the ground quad — the game's
// draw order (ground layer z=100 covers anything dipping below y=0). It is
// the hook for player/ghost rendering; a training view passes a callback
// that draws N players.
void drawWorld(const App& app, const Camera2D& cam, float colorX,
               const std::function<void()>& drawActors = {});

// Draws one atlas frame at an arbitrary world point (screen coords, y-down),
// no flips. Rotation is clockwise degrees, like the object path.
void drawFrameAt(const Atlas& atlas, const AtlasFrame& f, Vector2 worldCenter,
                 float rotationDeg, float scale, Color tint);

// Cube at engine coords: world = {s.x, -(s.y - 90)} (engine y = level y + 90,
// screen y = -level y). Sprite when available and textures on, else a colored
// rect. rotationDeg is cosmetic-only — never physics.
void drawPlayer(const App& app, const gdcore::PlayerState& s, float rotationDeg);

}  // namespace gdv
