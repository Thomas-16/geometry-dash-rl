// Viewer modes: Browse (free camera / level inspection) and Play (human plays
// the cube with the bit-exact gdcore physics). RL modes live in modes_rl.hpp.
// Every mode draws its own controls into the GUI panel's View tab via
// drawPanel() (ImGui, immediate mode), so the panel never touches mode state.
#pragma once

#include "viewer/world.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gdv {

class Mode {
public:
    virtual ~Mode() = default;
    virtual const char* name() const = 0;
    virtual void enter(App&) {}         // called on switch-in
    virtual void exit(App&) {}          // called on switch-away
    virtual void update(App&) = 0;      // input + sim, once per render frame
    virtual void draw(const App&) = 0;  // inside Begin/EndDrawing
    virtual void drawPanel(App&) {}     // ImGui widgets, inside the panel's View tab
    virtual void onLevelChanged(App&) {}  // app.level/geom were replaced
    // Desired render rate; main.cpp drives SetTargetFPS from this (and
    // halves it while a trainer runs in the background).
    virtual int targetFps(const App&) const { return 60; }
};

class BrowseMode : public Mode {
public:
    BrowseMode(float camX, float zoom);
    const char* name() const override { return "Browse"; }
    void enter(App&) override;
    void update(App&) override;
    void draw(const App&) override;
    void drawPanel(App&) override;

private:
    Camera2D cam_{};
    bool autoScroll_ = false;
    float gotoX_ = 0.0f;
};

class PlayMode : public Mode {
public:
    const char* name() const override { return "Play"; }
    // ~1 physics tick per frame, so hold/release edges quantize to 1/240s
    // instead of 1/60s. Purely an input-resolution improvement.
    int targetFps(const App&) const override { return 240; }
    void enter(App&) override;
    void exit(App&) override;
    void update(App&) override;
    void draw(const App&) override;
    void drawPanel(App&) override;
    void onLevelChanged(App&) override;

private:
    void resetAttempt(App&);
    void stepTick(App& app, bool hold);  // one 1/240 s physics tick
    void setPaused(App& app, bool paused);
    void saveSolution(App& app);

    gdcore::PhysicsConfig cfg_ = gdcore::stereoMadnessCube();
    gdcore::PlayerState player_{};
    Camera2D cam_{};
    double accum_ = 0.0;   // fixed-timestep accumulator (240 TPS)
    long long tick_ = 0;
    int attempt_ = 1;
    float deathTimer_ = 0.0f;  // >0: death pause counting down (wall time)
    float camY_ = 0.0f;        // engine y of the screen bottom (band follow)
    bool won_ = false;
    bool paused_ = false;
    bool stepOnce_ = false;
    float cubeRot_ = 0.0f;     // cosmetic-only spin, never physics
    float shipRot_ = 0.0f;     // cosmetic ship tilt (updateShipRotation-style)
    std::vector<uint8_t> holds_;  // per-tick input of the current attempt
    std::string status_;          // last save result shown in the panel
};

}  // namespace gdv
