#include "viewer/modes_rl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

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

// "3m ago" / "2h ago" from a file's last write time.
std::string ageLabel(const std::filesystem::path& p) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(p, ec);
    if (ec) return "";
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                    std::filesystem::file_time_type::clock::now() - t)
                    .count();
    if (secs < 0) secs = 0;
    char buf[32];
    if (secs < 90) std::snprintf(buf, sizeof buf, "%llds ago", (long long)secs);
    else if (secs < 5400)
        std::snprintf(buf, sizeof buf, "%lldm ago", (long long)(secs / 60));
    else if (secs < 60 * 60 * 48)
        std::snprintf(buf, sizeof buf, "%lldh ago", (long long)(secs / 3600));
    else
        std::snprintf(buf, sizeof buf, "%lldd ago", (long long)(secs / 86400));
    return buf;
}

// Episode count from a .gdtrace header (16 bytes), -1 on failure.
int traceEpisodeCount(const std::filesystem::path& p) {
    FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return -1;
    unsigned char head[16];
    bool ok = std::fread(head, 1, 16, f) == 16 && rdU32(head) == 0x52544447u;
    std::fclose(f);
    return ok ? (int)rdU32(head + 12) : -1;
}

std::string runsRelLabel(const std::filesystem::path& p) {
    // "runs/opt1/wins.gdtrace" -> "opt1/wins.gdtrace"
    auto rel = std::filesystem::relative(p, "runs");
    return rel.empty() ? p.string() : rel.generic_string();
}

} // namespace

std::vector<PickEntry> scanReplayFiles() {
    namespace fs = std::filesystem;
    std::vector<fs::path> found;
    std::error_code ec;
    for (auto& run : fs::directory_iterator("runs", ec)) {
        if (!run.is_directory()) continue;
        for (const char* name : {"wins.gdtrace", "eval.gdtrace"}) {
            fs::path p = run.path() / name;
            if (fs::exists(p, ec)) found.push_back(p);
        }
        std::error_code ec2;
        for (auto& t : fs::directory_iterator(run.path() / "traces", ec2))
            if (t.path().extension() == ".gdtrace") found.push_back(t.path());
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
        std::error_code e;
        return fs::last_write_time(a, e) > fs::last_write_time(b, e);
    });
    std::vector<PickEntry> out;
    for (const auto& p : found) {
        int eps = traceEpisodeCount(p);
        std::string label = runsRelLabel(p);
        if (eps >= 0) label += TextFormat("   (%d ep)", eps);
        std::string age = ageLabel(p);
        if (!age.empty()) label += "   " + age;
        out.push_back({std::move(label), p.string()});
    }
    return out;
}

std::vector<PickEntry> scanLiveFiles() {
    namespace fs = std::filesystem;
    std::vector<fs::path> found;
    std::error_code ec;
    for (auto& run : fs::directory_iterator("runs", ec)) {
        fs::path p = run.path() / "live.bin";
        if (run.is_directory() && fs::exists(p, ec)) found.push_back(p);
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
        std::error_code e;
        return fs::last_write_time(a, e) > fs::last_write_time(b, e);
    });
    std::vector<PickEntry> out;
    for (const auto& p : found) {
        std::string label = runsRelLabel(p);
        std::string age = ageLabel(p);
        if (!age.empty()) label += "   updated " + age;
        out.push_back({std::move(label), p.string()});
    }
    return out;
}

std::string newestLiveFile() {
    auto files = scanLiveFiles();
    return files.empty() ? "" : files.front().path;
}

// --------------------------------------------------------------------- Picker

void Picker::open(std::string title, std::vector<PickEntry> entries) {
    title_ = std::move(title);
    entries_ = std::move(entries);
    sel_ = 0;
    scroll_ = 0;
    open_ = true;
}

Rectangle Picker::panel() const {
    float w = std::min(780.0f, GetScreenWidth() - 80.0f);
    float h = std::min(540.0f, GetScreenHeight() - 120.0f);
    return {(GetScreenWidth() - w) / 2.0f, (GetScreenHeight() - h) / 2.0f, w, h};
}

int Picker::visibleRows() const {
    return std::max(1, (int)((panel().height - 84.0f) / 24.0f));
}

std::string Picker::update() {
    if (!open_) return "";
    if (IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_O)) {
        open_ = false;
        return "";
    }
    int n = (int)entries_.size();
    if (n > 0) {
        if (IsKeyPressed(KEY_DOWN)) sel_ = (sel_ + 1) % n;
        if (IsKeyPressed(KEY_UP)) sel_ = (sel_ + n - 1) % n;
        sel_ -= (int)GetMouseWheelMove();
        sel_ = std::clamp(sel_, 0, n - 1);
        int rows = visibleRows();
        if (sel_ < scroll_) scroll_ = sel_;
        if (sel_ >= scroll_ + rows) scroll_ = sel_ - rows + 1;

        Rectangle p = panel();
        Vector2 m = GetMousePosition();
        int hover = -1;
        if (m.x >= p.x + 8 && m.x <= p.x + p.width - 8) {
            int row = (int)((m.y - (p.y + 48.0f)) / 24.0f);
            if (row >= 0 && row < rows && scroll_ + row < n)
                hover = scroll_ + row;
        }
        if (hover >= 0) sel_ = hover;
        if (IsKeyPressed(KEY_ENTER) ||
            (hover >= 0 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))) {
            open_ = false;
            return entries_[sel_].path;
        }
    } else if (IsKeyPressed(KEY_ENTER)) {
        open_ = false;
    }
    return "";
}

void Picker::draw() const {
    if (!open_) return;
    DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(),
                  Color{0, 0, 0, 140});
    Rectangle p = panel();
    DrawRectangleRec(p, Color{24, 26, 38, 245});
    DrawRectangleLinesEx(p, 2.0f, Color{110, 120, 170, 255});
    DrawText(title_.c_str(), (int)p.x + 16, (int)p.y + 14, 20, RAYWHITE);
    int rows = visibleRows();
    int n = (int)entries_.size();
    if (n == 0) {
        DrawText("nothing found under runs/", (int)p.x + 16, (int)p.y + 56, 18,
                 Color{220, 160, 160, 255});
    }
    for (int r = 0; r < rows && scroll_ + r < n; ++r) {
        int i = scroll_ + r;
        float y = p.y + 48.0f + r * 24.0f;
        if (i == sel_)
            DrawRectangle((int)p.x + 8, (int)y - 2, (int)p.width - 16, 22,
                          Color{60, 70, 120, 255});
        DrawText(entries_[i].label.c_str(), (int)p.x + 16, (int)y, 18,
                 i == sel_ ? RAYWHITE : Color{185, 190, 210, 255});
    }
    if (n > rows)
        DrawText(TextFormat("%d-%d of %d", scroll_ + 1,
                            std::min(scroll_ + rows, n), n),
                 (int)p.x + 16, (int)(p.y + p.height - 26), 16,
                 Color{140, 145, 170, 255});
    DrawText("up/down or wheel: select   Enter/click: open   Esc: cancel",
             (int)(p.x + p.width - 470), (int)(p.y + p.height - 26), 16,
             Color{140, 145, 170, 255});
}

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
    if (path_.empty()) {
        // Reached without --replay: go straight to the file picker.
        if (!picker_.isOpen())
            picker_.open("Open replay trace", scanReplayFiles());
        return;
    }
    loadFile();
}

void ReplayMode::loadFile() {
    ghosts_.clear();
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
    if (picker_.isOpen()) {
        std::string sel = picker_.update();
        if (!sel.empty()) {
            path_ = sel;
            loadFile();
            restart();
        }
        return;  // modal: swallow the mode's own keys
    }
    if (IsKeyPressed(KEY_O))
        picker_.open("Open replay trace", scanReplayFiles());
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
        DrawText("O: open another trace", 12, 34, 16,
                 Color{170, 175, 200, 255});
        picker_.draw();
        return;
    }
    if (ghosts_.empty()) {
        DrawText("Replay — no trace loaded", 12, 10, 20, RAYWHITE);
        DrawText("O: open a trace from runs/", 12, 34, 16,
                 Color{170, 175, 200, 255});
        picker_.draw();
        return;
    }
    static constexpr float kSpeeds[] = {0.25f, 1.0f, 2.0f, 4.0f, 8.0f, 32.0f};
    float pct = app.maxX > 0 ? 100.0f * lx / app.maxX : 0.0f;
    DrawText(TextFormat("Replay [%s]  %zu episodes   alive %d   finished %d   "
                        "wins %d   lead x %.0f (%.0f%%)   tick %lld   %gx%s",
                        path_.c_str(), ghosts_.size(), alive, finished,
                        wonCount, lx, pct, tick_, kSpeeds[speedIdx_],
                        paused_ ? "  [paused]" : ""),
             12, 10, 20, RAYWHITE);
    DrawText(TextFormat("space: pause   left/right: speed   R: restart   "
                        "O: open trace   F: %s camera (drag/wheel when free)",
                        follow_ ? "follow" : "FREE"),
             12, 34, 16, Color{170, 175, 200, 255});
    picker_.draw();
}

// ------------------------------------------------------------ TrainingViewMode

void TrainingViewMode::enter(App&) {
    SetTargetFPS(60);
    // No --watch flag: follow the most recently updated run automatically.
    if (path_.empty()) path_ = newestLiveFile();
}

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
    if (picker_.isOpen()) {
        std::string sel = picker_.update();
        if (!sel.empty()) {
            path_ = sel;
            snap_ = Snapshot{};
        }
        return;
    }
    if (IsKeyPressed(KEY_O))
        picker_.open("Watch training run", scanLiveFiles());
    if (IsKeyPressed(KEY_F)) follow_ = !follow_;
    if (!follow_) freeCamUpdate(cam_);
    if (!path_.empty()) poll();
}

void TrainingViewMode::draw(const App& app) {
    if (!snap_.valid) {
        drawBackground(app, 0.0f);
        DrawText(path_.empty()
                     ? "Training view: no live.bin found under runs/"
                     : TextFormat("Training view: waiting for %s",
                                  path_.c_str()),
                 12, 10, 20, RAYWHITE);
        DrawText("start the trainer with --watch to publish live states   "
                 "O: pick a run",
                 12, 34, 16, Color{170, 175, 200, 255});
        picker_.draw();
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
    DrawText(TextFormat("Training [%s]  %zu envs   alive %d   lead x %.0f   "
                        "best %.0f (%.1f%%)   ep %u   %.2fM ticks/s   "
                        "%.1fh game time",
                        path_.c_str(), n, alive, leadX, snap_.bestX, pct,
                        snap_.episodes, snap_.sps / 1e6, gameHours),
             12, 10, 20, RAYWHITE);
    DrawText(TextFormat("live view of the PPO trainer (read-only)   O: pick "
                        "run   F: %s camera (drag/wheel when free)",
                        follow_ ? "follow" : "FREE"),
             12, 34, 16, Color{170, 175, 200, 255});
    picker_.draw();
}

}  // namespace gdv
