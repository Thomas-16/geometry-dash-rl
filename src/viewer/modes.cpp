#include "viewer/modes.hpp"

#include <algorithm>
#include <cmath>

namespace gdv {

// ---------------------------------------------------------------- BrowseMode

BrowseMode::BrowseMode(float camX, float zoom) {
    cam_.zoom = zoom;
    cam_.target = {camX, -180.0f};
}

void BrowseMode::enter(App&) {
    SetTargetFPS(60);
}

void BrowseMode::update(App&) {
    cam_.offset = {GetScreenWidth() * 0.25f, GetScreenHeight() * 0.6f};

    if (IsKeyPressed(KEY_SPACE)) autoScroll_ = !autoScroll_;
    if (IsKeyPressed(KEY_R)) { cam_.target = {0, -180.0f}; cam_.zoom = 1.5f; autoScroll_ = false; }
    if (autoScroll_) cam_.target.x += kCubeSpeed * GetFrameTime();
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 d = GetMouseDelta();
        cam_.target.x -= d.x / cam_.zoom;
        cam_.target.y -= d.y / cam_.zoom;
    }
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        Vector2 before = GetScreenToWorld2D(GetMousePosition(), cam_);
        cam_.zoom = std::clamp(cam_.zoom * (1.0f + 0.1f * wheel), 0.05f, 10.0f);
        Vector2 after = GetScreenToWorld2D(GetMousePosition(), cam_);
        cam_.target.x += before.x - after.x;
        cam_.target.y += before.y - after.y;
    }
}

void BrowseMode::draw(const App& app) {
    drawBackground(app, cam_.target.x);
    drawWorld(app, cam_, cam_.target.x);

    DrawText(TextFormat("objects: %d   cam.x: %.0f / %.0f   zoom: %.2f   %d fps%s",
                        (int)app.level.objects.size(), cam_.target.x, app.maxX,
                        cam_.zoom, GetFPS(), app.useTextures ? "" : "   [shapes]"),
             12, 10, 20, RAYWHITE);
    DrawText("drag: pan   wheel: zoom   space: auto-scroll   R: reset",
             12, 34, 16, Color{170, 175, 200, 255});
}

// ------------------------------------------------------------------ PlayMode

void PlayMode::resetAttempt() {
    // Full default-construct: also clears the landing-snap memory, so every
    // attempt is bit-identical to a fresh run. Never partially reset.
    player_ = gdcore::PlayerState{};
    accum_ = 0.0;
    tick_ = 0;
    cubeRot_ = 0.0f;
    deathTimer_ = 0.0f;
    camY_ = 0.0f;
}

void PlayMode::enter(App&) {
    // ~1 physics tick per frame, so hold/release edges quantize to 1/240s
    // instead of 1/60s. Purely an input-resolution improvement.
    SetTargetFPS(240);
    attempt_ = 1;
    won_ = false;
    resetAttempt();
}

void PlayMode::update(App& app) {
    if (won_) {
        if (IsKeyPressed(KEY_R)) { attempt_ = 1; won_ = false; resetAttempt(); }
        return;
    }
    if (player_.dead) {
        deathTimer_ -= GetFrameTime();
        if (deathTimer_ <= 0.0f) { ++attempt_; resetAttempt(); }
        return;
    }
    if (IsKeyPressed(KEY_R)) { attempt_ = 1; resetAttempt(); return; }

    // Input is polled per render frame, so at F fps every 240/F physics ticks
    // in this frame see the same hold state — press/release edges quantize to
    // frame boundaries. The IsKeyPressed latch guarantees a tap shorter than
    // one frame still jumps. Physics stays exact for the per-tick input
    // sequence actually fed; only human input resolution is frame-limited.
    bool hold = IsKeyDown(KEY_SPACE) || IsKeyDown(KEY_UP) ||
                IsMouseButtonDown(MOUSE_BUTTON_LEFT) ||
                IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_UP) ||
                IsMouseButtonPressed(MOUSE_BUTTON_LEFT);

    constexpr double kTick = 1.0 / 240.0;
    accum_ += GetFrameTime();
    // Cap so heavy frame drops slow the sim instead of spiraling.
    if (accum_ > 0.25) accum_ = 0.25;
    while (accum_ >= kTick && !player_.dead && !won_) {
        gdcore::step(player_, cfg_, app.geom, hold);
        accum_ -= kTick;
        ++tick_;
        if (player_.x >= app.maxX + 300.0f) won_ = true;
    }
    if (player_.dead) deathTimer_ = 0.9f;

    // Cosmetic spin: ~half turn per full hop (airtime ≈ 0.43s at these
    // constants). Snaps upright on the ground. Never feeds back into physics.
    if (!player_.onGround && !player_.dead) {
        cubeRot_ += 420.0f * GetFrameTime();
    } else {
        cubeRot_ = std::round(cubeRot_ / 90.0f) * 90.0f;
    }
}

void PlayMode::draw(const App& app) {
    // GD's viewport (OpenGD PlayLayer): 320 design units tall, player 40% of
    // the screen width from the left (cam.x = player.x - winSize.width/2.5).
    // Vertical: cube-mode band follow — camera moves down when the player is
    // less than 120 units above the camera bottom, up when less than 90 units
    // below the camera top, and never sinks below the ground (camY >= 0).
    // camY_ is the engine y of the screen bottom; ground top (engine 90) sits
    // 90 units above it at rest.
    if (player_.y - camY_ < 120.0f) camY_ = player_.y - 120.0f;
    else if (player_.y - camY_ > 320.0f - 90.0f) camY_ = player_.y - (320.0f - 90.0f);
    if (camY_ < 0.0f) camY_ = 0.0f;
    cam_.zoom = GetScreenHeight() / 320.0f;
    cam_.offset = {GetScreenWidth() / 2.5f, (float)GetScreenHeight()};
    // world y = -(engine y - 90); target sits at the screen bottom (offset.y=H)
    cam_.target = {player_.x, 90.0f - camY_};

    drawBackground(app, player_.x);
    drawWorld(app, cam_, player_.x,
              [&] { drawPlayer(app, player_, cubeRot_); });

    float progress = app.maxX > 0 ? std::min(1.0f, player_.x / app.maxX) : 0.0f;
    DrawText(TextFormat("Attempt %d   x: %.0f / %.0f  (%d%%)   %d fps%s",
                        attempt_, player_.x, app.maxX, (int)(progress * 100),
                        GetFPS(), app.useTextures ? "" : "   [shapes]"),
             12, 10, 20, RAYWHITE);
    DrawText("space/up/click: jump (hold to keep hopping)   R: restart",
             12, 34, 16, Color{170, 175, 200, 255});

    int sw = GetScreenWidth(), sh = GetScreenHeight();
    if (player_.dead) {
        DrawRectangle(0, 0, sw, sh, Fade(RED, 0.35f * (deathTimer_ / 0.9f)));
        const char* msg = TextFormat("Attempt %d", attempt_ + 1);
        int tw = MeasureText(msg, 48);
        DrawText(msg, (sw - tw) / 2, sh / 2 - 24, 48, RAYWHITE);
    }
    if (won_) {
        const char* msg = "LEVEL COMPLETE";
        int tw = MeasureText(msg, 56);
        DrawText(msg, (sw - tw) / 2, sh / 2 - 48, 56, RAYWHITE);
        const char* sub = TextFormat("attempt %d — R: play again", attempt_);
        int tw2 = MeasureText(sub, 24);
        DrawText(sub, (sw - tw2) / 2, sh / 2 + 16, 24, Color{170, 175, 200, 255});
    }
}

}  // namespace gdv
