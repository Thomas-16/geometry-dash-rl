#include "viewer/modes.hpp"

#include "viewer/input.hpp"
#include "viewer/paths.hpp"
#include "viewer/runs.hpp"

#include "imgui.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <ctime>

namespace gdv {

// ---------------------------------------------------------------- BrowseMode

BrowseMode::BrowseMode(float camX, float zoom) {
    cam_.zoom = zoom;
    cam_.target = {camX, -180.0f};
}

void BrowseMode::enter(App&) {
}

void BrowseMode::update(App& app) {
    cam_.offset = {viewW(app) * 0.25f, viewH(app) * 0.6f};

    if (keyPressed(app, KEY_SPACE)) autoScroll_ = !autoScroll_;
    if (keyPressed(app, KEY_R)) { cam_.target = {0, -180.0f}; cam_.zoom = 1.5f; autoScroll_ = false; }
    if (autoScroll_) cam_.target.x += kCubeSpeed * GetFrameTime();
    if (mouseDown(app, MOUSE_BUTTON_LEFT)) {
        Vector2 d = GetMouseDelta();
        cam_.target.x -= d.x / cam_.zoom;
        cam_.target.y -= d.y / cam_.zoom;
    }
    float wheel = mouseWheel(app);
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

void BrowseMode::drawPanel(App& app) {
    ImGui::Text("cam.x %.0f / %.0f   zoom %.2f", cam_.target.x, app.maxX, cam_.zoom);
    if (ImGui::Button("Reset camera")) { cam_.target = {0, -180.0f}; cam_.zoom = 1.5f; autoScroll_ = false; }
    ImGui::SameLine();
    ImGui::Checkbox("Auto-scroll", &autoScroll_);
    ImGui::SetNextItemWidth(120);
    ImGui::InputFloat("##gotox", &gotoX_, 0, 0, "%.0f");
    ImGui::SameLine();
    if (ImGui::Button("Go to x")) cam_.target.x = gotoX_;
    ImGui::SameLine();
    if (ImGui::Button("End")) cam_.target.x = app.maxX;
    ImGui::TextDisabled("drag = pan, wheel = zoom");
}

// ------------------------------------------------------------------ PlayMode

void PlayMode::resetAttempt(App& app) {
    // Full default-construct: also clears the landing-snap memory, so every
    // attempt is bit-identical to a fresh run. Never partially reset.
    player_ = gdcore::PlayerState{};
    accum_ = 0.0;
    tick_ = 0;
    cubeRot_ = 0.0f;
    shipRot_ = 0.0f;
    deathTimer_ = 0.0f;
    camY_ = 0.0f;
    holds_.clear();
    // Every attempt starts at x=0 (no checkpoints), so the song restarts —
    // same as the real game. update() starts playback at the right position.
    if (app.musicLoaded) StopMusicStream(app.music);
}

void PlayMode::enter(App& app) {
    attempt_ = 1;
    won_ = false;
    paused_ = false;
    status_.clear();
    resetAttempt(app);
}

void PlayMode::exit(App& app) {
    if (app.musicLoaded) StopMusicStream(app.music);
}

void PlayMode::onLevelChanged(App& app) {
    attempt_ = 1;
    won_ = false;
    paused_ = false;
    status_.clear();
    resetAttempt(app);
}

void PlayMode::setPaused(App& app, bool paused) {
    if (paused == paused_) return;
    paused_ = paused;
    if (paused && app.musicLoaded && IsMusicStreamPlaying(app.music)) PauseMusicStream(app.music);
    if (!paused && app.musicLoaded) ResumeMusicStream(app.music);
}

void PlayMode::stepTick(App& app, bool hold) {
    gdcore::step(player_, cfg_, app.geom, hold);
    holds_.push_back(hold ? 1 : 0);
    ++tick_;
    if (player_.ship) {
        // Cosmetic tilt toward the velocity direction, like
        // updateShipRotation: slerp 0.15/tick toward atan2(dy, dx).
        float dy = static_cast<float>(player_.yvel) * cfg_.yFactor;
        float target = std::atan2(-dy, cfg_.dx) * RAD2DEG;
        shipRot_ += (target - shipRot_) * 0.15f;
    } else {
        shipRot_ = 0.0f;
    }
    // Real completion trigger: the session6 full-run trace shows normal
    // physics ending at last-object + ~40 units (x froze at +40.49; the
    // end-of-level auto-walk takes over from there).
    if (player_.x >= app.maxX + 40.0f) won_ = true;
    // tick_ is the authoritative clock; the accumulator clamp can let sim
    // time fall behind wall time, so resnap the song once a second when it
    // drifts audibly.
    if (app.musicLoaded && tick_ % 240 == 0 && IsMusicStreamPlaying(app.music)) {
        float target = (float)(tick_ / 240.0) + app.musicOffset;
        if (target >= 0.0f && std::fabs(GetMusicTimePlayed(app.music) - target) > 0.1f)
            SeekMusicStream(app.music, target);
    }
}

void PlayMode::update(App& app) {
    if (won_) {
        if (keyPressed(app, KEY_R)) { attempt_ = 1; won_ = false; resetAttempt(app); }
        return;
    }
    if (player_.dead) {
        deathTimer_ -= GetFrameTime();
        if (deathTimer_ <= 0.0f) { ++attempt_; resetAttempt(app); }
        return;
    }
    if (keyPressed(app, KEY_R)) { attempt_ = 1; resetAttempt(app); return; }
    if (keyPressed(app, KEY_P)) setPaused(app, !paused_);

    // Input is polled per render frame, so at F fps every 240/F physics ticks
    // in this frame see the same hold state — press/release edges quantize to
    // frame boundaries. The IsKeyPressed latch guarantees a tap shorter than
    // one frame still jumps. Physics stays exact for the per-tick input
    // sequence actually fed; only human input resolution is frame-limited.
    bool hold = keyDown(app, KEY_SPACE) || keyDown(app, KEY_UP) ||
                mouseDown(app, MOUSE_BUTTON_LEFT) ||
                keyPressed(app, KEY_SPACE) || keyPressed(app, KEY_UP) ||
                mousePressed(app, MOUSE_BUTTON_LEFT);

    if (paused_) {
        if (stepOnce_) {
            stepOnce_ = false;
            stepTick(app, hold);
        }
        accum_ = 0.0;
    } else {
        constexpr double kTick = 1.0 / 240.0;
        accum_ += GetFrameTime();
        // Cap so heavy frame drops slow the sim instead of spiraling.
        if (accum_ > 0.25) accum_ = 0.25;
        while (accum_ >= kTick && !player_.dead && !won_) {
            stepTick(app, hold);
            accum_ -= kTick;
        }
    }
    if (player_.dead) {
        deathTimer_ = 0.9f;
        if (app.musicLoaded) StopMusicStream(app.music);
    } else if (!paused_ && app.musicLoaded && !IsMusicStreamPlaying(app.music)) {
        // Attempt (re)start, or delayed start under a negative --music-offset.
        float target = (float)(tick_ / 240.0) + app.musicOffset;
        if (target >= 0.0f && target < GetMusicTimeLength(app.music)) {
            PlayMusicStream(app.music);
            SeekMusicStream(app.music, target);
        }
    }

    // Cosmetic spin: ~half turn per full hop (airtime ≈ 0.43s at these
    // constants). Snaps upright on the ground. Never feeds back into physics.
    if (!player_.onGround && !player_.dead && !paused_) {
        cubeRot_ += 420.0f * GetFrameTime();
    } else if (player_.onGround) {
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
    if (player_.ship) {
        // Ship band: floor 90 / ceiling 390 both on screen; ease the camera
        // toward the band centered vertically (80 = 90 - (320-300)/2).
        camY_ += (80.0f - camY_) * std::min(1.0f, 6.0f * GetFrameTime());
    } else {
        if (player_.y - camY_ < 120.0f) camY_ = player_.y - 120.0f;
        else if (player_.y - camY_ > 320.0f - 90.0f) camY_ = player_.y - (320.0f - 90.0f);
        if (camY_ < 0.0f) camY_ = 0.0f;
    }
    int sw = viewW(app), sh = viewH(app);
    cam_.zoom = sh / 320.0f;
    cam_.offset = {sw / 2.5f, (float)sh};
    // world y = -(engine y - 90); target sits at the screen bottom (offset.y=H)
    cam_.target = {player_.x, 90.0f - camY_};

    drawBackground(app, player_.x);
    drawWorld(app, cam_, player_.x,
              [&] { drawPlayer(app, player_, player_.ship ? shipRot_ : cubeRot_); },
              player_.ship ? 390.0f : 0.0f);

    float progress = app.maxX > 0 ? std::min(1.0f, player_.x / app.maxX) : 0.0f;
    DrawText(TextFormat("Attempt %d   x: %.0f / %.0f  (%d%%)   tick %lld   %d fps%s%s",
                        attempt_, player_.x, app.maxX, (int)(progress * 100), tick_,
                        GetFPS(), app.useTextures ? "" : "   [shapes]",
                        paused_ ? "   [PAUSED]" : ""),
             12, 10, 20, RAYWHITE);
    DrawText("space/up/click: jump (hold to keep hopping)   R: restart   P: pause",
             12, 34, 16, Color{170, 175, 200, 255});

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

void PlayMode::saveSolution(App& app) {
    Solution s;
    s.levelId = app.levelId;
    s.endX = app.maxX + 40.0f;
    s.nTicks = (int)holds_.size();
    s.deterministic = false;
    s.holds = holds_;
    char stamp[32];
    std::time_t t = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", std::localtime(&t));
    std::filesystem::path dir =
        paths::runsDir() / ("human_" + std::to_string(app.levelId) + "_" + stamp);
    std::string err;
    if (writeSolution(dir / "solution.json", s, err))
        status_ = "saved " + (dir / "solution.json").generic_string();
    else
        status_ = "save failed: " + err;
}

void PlayMode::drawPanel(App& app) {
    float progress = app.maxX > 0 ? std::min(1.0f, player_.x / app.maxX) : 0.0f;
    ImGui::Text("attempt %d   x %.1f (%.1f%%)   tick %lld", attempt_, player_.x,
                progress * 100.0f, tick_);
    ImGui::Text("y %.2f  yvel %.3f  %s%s%s", player_.y, player_.yvel,
                player_.ship ? "ship" : (player_.upsideDown ? "cube (flipped)" : "cube"),
                player_.onGround ? "  ground" : "",
                player_.dead ? "  DEAD" : "");
    if (ImGui::Button("Restart")) { attempt_ = 1; won_ = false; setPaused(app, false); resetAttempt(app); }
    ImGui::SameLine();
    if (ImGui::Button(paused_ ? "Resume" : "Pause")) setPaused(app, !paused_);
    ImGui::SameLine();
    ImGui::BeginDisabled(!paused_ || player_.dead || won_);
    if (ImGui::Button("Step 1 tick")) stepOnce_ = true;
    ImGui::EndDisabled();
    if (paused_) ImGui::TextDisabled("paused: hold jump while stepping to feed a press");
    ImGui::TextDisabled("holds recorded: %d ticks", (int)holds_.size());
    if (won_) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.6f, 1.0f), "LEVEL COMPLETE in %d ticks", (int)holds_.size());
        if (ImGui::Button("Save as solution.json (runs/human_*)")) saveSolution(app);
        ImGui::TextDisabled("deployable like any run: Runs tab -> Deploy");
    }
    if (!status_.empty()) ImGui::TextWrapped("%s", status_.c_str());
    ImGui::TextDisabled("space / up / click = jump   R = restart   P = pause");
}

}  // namespace gdv
