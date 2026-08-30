#include "viewer/modes_rl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace gdv {

namespace {

// Same GD-exact follow as PlayMode::draw, parameterized on the leader.
void followCamera(Camera2D& cam, float& camY, float x, float y, bool ship,
                  float dt) {
    if (ship) {
        camY += (80.0f - camY) * std::min(1.0f, 6.0f * dt);
    } else {
        if (y - camY < 120.0f) camY = y - 120.0f;
        else if (y - camY > 320.0f - 90.0f) camY = y - (320.0f - 90.0f);
        if (camY < 0.0f) camY = 0.0f;
    }
    cam.zoom = GetScreenHeight() / 320.0f;
    cam.offset = {GetScreenWidth() / 2.5f, (float)GetScreenHeight()};
    cam.target = {x, 90.0f - camY};
}

// BrowseMode-style manual camera: drag pans, wheel zooms around the cursor.
void freeCamUpdate(Camera2D& cam) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 d = GetMouseDelta();
        cam.target.x -= d.x / cam.zoom;
        cam.target.y -= d.y / cam.zoom;
    }
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        Vector2 before = GetScreenToWorld2D(GetMousePosition(), cam);
        cam.zoom = std::clamp(cam.zoom * (1.0f + 0.1f * wheel), 0.05f, 10.0f);
        Vector2 after = GetScreenToWorld2D(GetMousePosition(), cam);
        cam.target.x += before.x - after.x;
        cam.target.y += before.y - after.y;
    }
}

Color ghostTint(int i) {
    // Golden-angle hue walk: adjacent indices get well-separated colors.
    return ColorFromHSV(std::fmod(i * 137.508f, 360.0f), 0.7f, 1.0f);
}

uint32_t rdU32(const unsigned char* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
float rdF32(const unsigned char* p) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
}

} // namespace

// ------------------------------------------------------------------ ReplayMode

void ReplayMode::restart() {
    for (auto& g : ghosts_) {
        g.s = gdcore::PlayerState{};
        g.tick = 0;
        g.deathAge = -1;
    }
    accum_ = 0.0;
    tick_ = 0;
}

void ReplayMode::enter(App&) {
    SetTargetFPS(60);
    if (!ghosts_.empty()) return;  // already loaded; keep replay position
    error_.clear();
    // .gdtrace: see python/gdrl/traces.py for the format definition.
    FILE* f = std::fopen(path_.c_str(), "rb");
    if (!f) {
        error_ = "cannot open " + path_;
        return;
    }
    unsigned char head[16];
    if (std::fread(head, 1, 16, f) != 16 || rdU32(head) != 0x52544447u ||
        rdU32(head + 4) != 1 || rdU32(head + 8) != 240) {
        error_ = "bad .gdtrace header in " + path_;
        std::fclose(f);
        return;
    }
    uint32_t n = rdU32(head + 12);
    for (uint32_t i = 0; i < n; ++i) {
        unsigned char eh[12];
        if (std::fread(eh, 1, 12, f) != 12) break;
        Ghost g;
        uint32_t nTicks = rdU32(eh);
        g.finalX = rdF32(eh + 4);
        g.won = eh[8] != 0;
        std::vector<unsigned char> bits((nTicks + 7) / 8);
        if (std::fread(bits.data(), 1, bits.size(), f) != bits.size()) break;
        g.holds.resize(nTicks);
        for (uint32_t t = 0; t < nTicks; ++t)
            g.holds[t] = (bits[t / 8] >> (t % 8)) & 1;  // LSB-first
        g.tint = ghostTint((int)i);
        ghosts_.push_back(std::move(g));
    }
    std::fclose(f);
    if (ghosts_.empty() && error_.empty()) error_ = "no episodes in " + path_;
    restart();
}

void ReplayMode::update(App& app) {
    static constexpr float kSpeeds[] = {0.25f, 1.0f, 2.0f, 4.0f, 8.0f, 32.0f};
    constexpr int kNumSpeeds = 6;
    if (IsKeyPressed(KEY_SPACE)) paused_ = !paused_;
    if (IsKeyPressed(KEY_R)) restart();
    if (IsKeyPressed(KEY_F)) follow_ = !follow_;
    if (!follow_) freeCamUpdate(cam_);
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_UP))
        speedIdx_ = std::min(speedIdx_ + 1, kNumSpeeds - 1);
    if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_DOWN))
        speedIdx_ = std::max(speedIdx_ - 1, 0);
    if (paused_ || ghosts_.empty()) return;

    accum_ += GetFrameTime() * kSpeeds[speedIdx_] * 240.0;
    // Cap the per-frame work so 32x on long traces stays responsive.
    int ticks = std::min((int)accum_, 240 * 32);
    accum_ -= ticks;
    for (int t = 0; t < ticks; ++t) {
        ++tick_;
        for (auto& g : ghosts_) {
            if (g.deathAge >= 0) {
                ++g.deathAge;
                continue;
            }
            if (g.tick >= (int)g.holds.size()) continue;  // finished: freeze
            gdcore::step(g.s, cfg_, app.geom, g.holds[g.tick] != 0);
            ++g.tick;
            if (g.s.dead) g.deathAge = 0;
        }
    }
}

void ReplayMode::draw(const App& app) {
    // Leader: furthest ghost not dead (finished ones stay eligible).
    const Ghost* lead = nullptr;
    int alive = 0, wonCount = 0, finished = 0;
    for (const auto& g : ghosts_) {
        if (g.deathAge < 0) {
            ++alive;
            if (!lead || g.s.x > lead->s.x) lead = &g;
        }
        if (g.won) ++wonCount;
        if (g.deathAge < 0 && g.tick >= (int)g.holds.size()) ++finished;
    }
    if (!lead && !ghosts_.empty()) lead = &ghosts_[0];
    float lx = lead ? lead->s.x : 0.0f;
    if (follow_)
        followCamera(cam_, camY_, lx, lead ? lead->s.y : 105.0f,
                     lead && lead->s.ship, GetFrameTime());
    float colorX = follow_ ? lx : cam_.target.x;

    drawBackground(app, colorX);
    drawWorld(app, cam_, colorX, [&] {
        for (const auto& g : ghosts_) {
            if (g.deathAge >= 0) {
                if (g.deathAge > 90) continue;  // fade-out done
                unsigned char a =
                    (unsigned char)(150 * (1.0f - g.deathAge / 90.0f));
                drawPlayer(app, g.s, 0.0f, Color{255, 60, 60, 255},
                           Color{160, 20, 20, 255}, a);
            } else {
                drawPlayer(app, g.s, 0.0f, g.tint,
                           ColorBrightness(g.tint, -0.35f), 150);
            }
        }
    }, lead && lead->s.ship ? 390.0f : 0.0f);

    if (!error_.empty()) {
        DrawText(error_.c_str(), 12, 10, 20, RED);
        return;
    }
    static constexpr float kSpeeds[] = {0.25f, 1.0f, 2.0f, 4.0f, 8.0f, 32.0f};
    float pct = app.maxX > 0 ? 100.0f * lx / app.maxX : 0.0f;
    DrawText(TextFormat("Replay  %zu episodes   alive %d   finished %d   "
                        "wins %d   lead x %.0f (%.0f%%)   tick %lld   %gx%s",
                        ghosts_.size(), alive, finished, wonCount, lx, pct,
                        tick_, kSpeeds[speedIdx_], paused_ ? "  [paused]" : ""),
             12, 10, 20, RAYWHITE);
    DrawText(TextFormat("space: pause   left/right: speed   R: restart   "
                        "F: %s camera (drag/wheel when free)",
                        follow_ ? "follow" : "FREE"),
             12, 34, 16, Color{170, 175, 200, 255});
}

// ------------------------------------------------------------ TrainingViewMode

void TrainingViewMode::enter(App&) { SetTargetFPS(60); }

bool TrainingViewMode::poll() {
    // live.bin layout: see python/gdrl/live.py. Seqlock: odd seq = write in
    // progress; keep the previous snapshot and retry next frame.
    FILE* f = std::fopen(path_.c_str(), "rb");
    if (!f) return false;
    unsigned char head[40];
    if (std::fread(head, 1, 40, f) != 40 || rdU32(head) != 0x564C4447u ||
        rdU32(head + 4) != 1) {
        std::fclose(f);
        return false;
    }
    uint32_t seq = rdU32(head + 8);
    uint32_t n = rdU32(head + 12);
    if (seq % 2 == 1 || n == 0 || n > 65536) {
        std::fclose(f);
        return false;
    }
    std::vector<unsigned char> rec(n * 20u);
    size_t got = std::fread(rec.data(), 1, rec.size(), f);
    std::fclose(f);
    if (got != rec.size()) return false;

    std::memcpy(&snap_.globalTicks, head + 16, 8);
    snap_.bestX = rdF32(head + 24);
    snap_.sps = rdF32(head + 28);
    snap_.episodes = rdU32(head + 32);
    snap_.xyv.resize(n * 3);
    snap_.ship.resize(n);
    snap_.dead.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char* p = rec.data() + i * 20;
        snap_.xyv[i * 3 + 0] = rdF32(p);
        snap_.xyv[i * 3 + 1] = rdF32(p + 4);
        snap_.xyv[i * 3 + 2] = rdF32(p + 8);
        uint32_t flags = rdU32(p + 12);
        snap_.ship[i] = flags & 1;
        snap_.dead[i] = (flags >> 8) & 1;
    }
    snap_.valid = true;
    return true;
}

void TrainingViewMode::update(App&) {
    if (IsKeyPressed(KEY_F)) follow_ = !follow_;
    if (!follow_) freeCamUpdate(cam_);
    poll();
}

void TrainingViewMode::draw(const App& app) {
    if (!snap_.valid) {
        drawBackground(app, 0.0f);
        DrawText(TextFormat("Training view: waiting for %s", path_.c_str()),
                 12, 10, 20, RAYWHITE);
        DrawText("start the trainer with --watch to publish live states",
                 12, 34, 16, Color{170, 175, 200, 255});
        return;
    }
    size_t n = snap_.ship.size();
    int alive = 0;
    size_t leadIdx = 0;
    float leadX = -1.0f;
    for (size_t i = 0; i < n; ++i) {
        if (snap_.dead[i]) continue;
        ++alive;
        if (snap_.xyv[i * 3] > leadX) {
            leadX = snap_.xyv[i * 3];
            leadIdx = i;
        }
    }
    if (leadX < 0.0f) leadX = snap_.xyv.empty() ? 0.0f : snap_.xyv[leadIdx * 3];
    bool leadShip = snap_.ship[leadIdx] != 0;
    if (follow_)
        followCamera(cam_, camY_, leadX, snap_.xyv[leadIdx * 3 + 1], leadShip,
                     GetFrameTime());
    float colorX = follow_ ? leadX : cam_.target.x;

    drawBackground(app, colorX);
    drawWorld(app, cam_, colorX, [&] {
        gdcore::PlayerState s;
        for (size_t i = 0; i < n; ++i) {
            s.x = snap_.xyv[i * 3];
            s.y = snap_.xyv[i * 3 + 1];
            s.ship = snap_.ship[i] != 0;
            // The trainer auto-resets on death, so dead=1 snapshots are rare
            // single frames; drawing them red still reads as a death flash
            // when the write rate is high.
            if (snap_.dead[i]) {
                drawPlayer(app, s, 0.0f, Color{255, 60, 60, 255},
                           Color{160, 20, 20, 255}, 120);
            } else {
                drawPlayer(app, s, 0.0f, ghostTint((int)i),
                           ColorBrightness(ghostTint((int)i), -0.35f), 140);
            }
        }
    }, leadShip ? 390.0f : 0.0f);

    float pct = app.maxX > 0 ? 100.0f * snap_.bestX / app.maxX : 0.0f;
    double gameHours =
        (double)snap_.globalTicks / 240.0 / 3600.0;
    DrawText(TextFormat("Training  %zu envs   alive %d   lead x %.0f   "
                        "best %.0f (%.1f%%)   ep %u   %.2fM ticks/s   "
                        "%.1fh game time",
                        n, alive, leadX, snap_.bestX, pct, snap_.episodes,
                        snap_.sps / 1e6, gameHours),
             12, 10, 20, RAYWHITE);
    DrawText(TextFormat("live view of the PPO trainer (read-only)   "
                        "F: %s camera (drag/wheel when free)",
                        follow_ ? "follow" : "FREE"),
             12, 34, 16, Color{170, 175, 200, 255});
}

}  // namespace gdv
