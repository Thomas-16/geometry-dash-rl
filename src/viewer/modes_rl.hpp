// RL visualization modes: ReplayMode re-simulates .gdtrace episodes (per-tick
// hold bits; exact by determinism) as tinted ghosts, TrainingViewMode renders
// a live snapshot of every training env from the trainer's live.bin.
#pragma once

#include "viewer/modes.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gdv {

class ReplayMode : public Mode {
public:
    explicit ReplayMode(std::string tracePath) : path_(std::move(tracePath)) {}
    const char* name() const override { return "Replay"; }
    void enter(App&) override;
    void update(App&) override;
    void draw(const App&) override;

private:
    struct Ghost {
        std::vector<uint8_t> holds;  // one per physics tick
        float finalX = 0.0f;
        bool won = false;
        gdcore::PlayerState s;
        int tick = 0;
        int deathAge = -1;  // ticks since death; -1 = alive
        Color tint{};
    };
    void restart();

    std::string path_;
    std::string error_;
    gdcore::PhysicsConfig cfg_ = gdcore::stereoMadnessCube();
    std::vector<Ghost> ghosts_;
    double accum_ = 0.0;
    long long tick_ = 0;
    int speedIdx_ = 1;  // index into kSpeeds
    bool paused_ = false;
    bool follow_ = true;  // F: follow leader vs free drag/zoom camera
    float camY_ = 0.0f;
    Camera2D cam_{};
};

class TrainingViewMode : public Mode {
public:
    explicit TrainingViewMode(std::string livePath)
        : path_(std::move(livePath)) {}
    const char* name() const override { return "Training"; }
    void enter(App&) override;
    void update(App&) override;
    void draw(const App&) override;

private:
    struct Snapshot {
        uint64_t globalTicks = 0;
        float bestX = 0.0f;
        float sps = 0.0f;
        uint32_t episodes = 0;
        // per env: x, y, yvel + ship/dead flags
        std::vector<float> xyv;      // n*3
        std::vector<uint8_t> ship, dead;
        bool valid = false;
    };
    bool poll();

    std::string path_;
    Snapshot snap_;
    bool follow_ = true;  // F: follow leader vs free drag/zoom camera
    float camY_ = 0.0f;
    Camera2D cam_{};
};

}  // namespace gdv
