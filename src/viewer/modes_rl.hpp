// RL visualization modes: ReplayMode re-simulates recorded hold sequences
// (.gdtrace episodes or a solution.json) as tinted ghosts — exact by
// determinism — and TrainingViewMode renders a live snapshot of every
// training env from the trainer's live.bin. Files are chosen from the GUI
// panel (drawPanel) or pushed in via App::commands.
#pragma once

#include "viewer/modes.hpp"
#include "viewer/runs.hpp"

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
    void drawPanel(App&) override;
    void onLevelChanged(App&) override;

    void openTrace(const std::string& path);     // .gdtrace
    void openSolution(const std::string& path);  // solution.json

private:
    struct Ghost {
        std::vector<uint8_t> holds;  // one per physics tick
        float finalX = 0.0f;
        bool won = false;
        gdcore::PlayerState s;
        int tick = 0;
        int deathAge = -1;  // ticks since death; -1 = alive
        bool hidden = false;
        Color tint{};
        std::vector<Vector2> presses;  // world positions of hold rising edges
    };
    void restart();
    void stepOne(const App& app);         // all ghosts, one tick
    void stepTo(const App& app, int t);   // deterministic fast-forward (restarts if t < tick_)
    void rescan();

    std::string path_;
    std::string error_;
    gdcore::PhysicsConfig cfg_ = gdcore::stereoMadnessCube();
    std::vector<Ghost> ghosts_;
    int maxTicks_ = 0;
    double accum_ = 0.0;
    long long tick_ = 0;
    int speedIdx_ = 1;  // index into kSpeeds
    bool paused_ = false;
    bool follow_ = true;  // follow leader vs free drag/zoom camera
    bool markers_ = true;
    float camY_ = 0.0f;
    Camera2D cam_{};
    // panel state
    std::vector<FileEntry> traces_, solutions_;
    double lastScan_ = -1.0;
    int selTrace_ = -1, selSolution_ = -1;
    int scrub_ = 0;
};

class TrainingViewMode : public Mode {
public:
    explicit TrainingViewMode(std::string livePath)
        : path_(std::move(livePath)) {}
    const char* name() const override { return "Training"; }
    void enter(App&) override;
    void update(App&) override;
    void draw(const App&) override;
    void drawPanel(App&) override;
    int targetFps(const App& app) const override { return app.livePreview ? 60 : 30; }

    void watch(const std::string& livePath);

private:
    struct Snapshot {
        uint64_t globalTicks = 0;
        float bestX = 0.0f;
        float sps = 0.0f;       // sim ticks/s of the last completed update
        float samplesS = 0.0f;  // decisions/s of the last completed update
        uint32_t episodes = 0;
        uint32_t seq = 0;
        // per env: x, y, yvel + ship/dead flags
        std::vector<float> xyv;      // n*3
        std::vector<uint8_t> ship, dead;
        bool valid = false;
    };
    bool poll();
    void rescan();

    std::string path_;
    Snapshot snap_;
    std::vector<unsigned char> rec_;        // payload staging, reused per frame
    std::vector<Color> tint_, tintDark_;    // ghost palette, built once per n
    bool follow_ = true;
    float camY_ = 0.0f;
    Camera2D cam_{};
    double lastSeqChange_ = 0.0;
    std::vector<FileEntry> lives_;
    double lastScan_ = -1.0;
};

}  // namespace gdv
