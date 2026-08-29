// Viewer modes: Browse (free camera / level inspection) and Play (human plays
// the cube with the bit-exact gdcore physics). A future TrainingView mode is
// one more Mode subclass feeding many PlayerStates into drawWorld's actor hook.
#pragma once

#include "viewer/world.hpp"

namespace gdv {

class Mode {
public:
    virtual ~Mode() = default;
    virtual const char* name() const = 0;
    virtual void enter(App&) {}         // called on switch-in
    virtual void exit(App&) {}          // called on switch-away
    virtual void update(App&) = 0;      // input + sim, once per render frame
    virtual void draw(const App&) = 0;  // inside Begin/EndDrawing
};

class BrowseMode : public Mode {
public:
    BrowseMode(float camX, float zoom);
    const char* name() const override { return "Browse"; }
    void enter(App&) override;
    void update(App&) override;
    void draw(const App&) override;

private:
    Camera2D cam_{};
    bool autoScroll_ = false;
};

class PlayMode : public Mode {
public:
    const char* name() const override { return "Play"; }
    void enter(App&) override;
    void exit(App&) override;
    void update(App&) override;
    void draw(const App&) override;

private:
    void resetAttempt(App&);

    gdcore::PhysicsConfig cfg_ = gdcore::stereoMadnessCube();
    gdcore::PlayerState player_{};
    Camera2D cam_{};
    double accum_ = 0.0;   // fixed-timestep accumulator (240 TPS)
    long long tick_ = 0;
    int attempt_ = 1;
    float deathTimer_ = 0.0f;  // >0: death pause counting down (wall time)
    float camY_ = 0.0f;        // engine y of the screen bottom (band follow)
    bool won_ = false;
    float cubeRot_ = 0.0f;     // cosmetic-only spin, never physics
};

}  // namespace gdv
