#include "viewer/modes_rl.hpp"

#include "viewer/input.hpp"

#include "imgui.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace gdv {

namespace {

constexpr float kSpeeds[] = {0.25f, 1.0f, 2.0f, 4.0f, 8.0f, 32.0f};
constexpr int kNumSpeeds = 6;
const char* const kSpeedLabels[] = {"0.25x", "1x", "2x", "4x", "8x", "32x"};

// Same GD-exact follow as PlayMode::draw, parameterized on the leader.
void followCamera(const App& app, Camera2D& cam, float& camY, float x, float y,
                  bool ship, float dt) {
    if (ship) {
        camY += (80.0f - camY) * std::min(1.0f, 6.0f * dt);
    } else {
        if (y - camY < 120.0f) camY = y - 120.0f;
        else if (y - camY > 320.0f - 90.0f) camY = y - (320.0f - 90.0f);
        if (camY < 0.0f) camY = 0.0f;
    }
    cam.zoom = viewH(app) / 320.0f;
    cam.offset = {viewW(app) / 2.5f, (float)viewH(app)};
    cam.target = {x, 90.0f - camY};
}

// BrowseMode-style manual camera: drag pans, wheel zooms around the cursor.
void freeCamUpdate(const App& app, Camera2D& cam) {
    if (mouseDown(app, MOUSE_BUTTON_LEFT)) {
        Vector2 d = GetMouseDelta();
        cam.target.x -= d.x / cam.zoom;
        cam.target.y -= d.y / cam.zoom;
    }
    float wheel = mouseWheel(app);
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

// Simple selectable list for FileEntry vectors; returns the clicked index or -1.
int fileList(const char* id, const std::vector<FileEntry>& entries, int& sel, float height) {
    int clicked = -1;
    if (ImGui::BeginListBox(id, ImVec2(-FLT_MIN, height))) {
        for (int i = 0; i < (int)entries.size(); ++i) {
            bool isSel = sel == i;
            if (ImGui::Selectable(entries[i].label.c_str(), isSel,
                                  ImGuiSelectableFlags_AllowDoubleClick)) {
                sel = i;
                if (ImGui::IsMouseDoubleClicked(0)) clicked = i;
            }
        }
        if (entries.empty()) ImGui::TextDisabled("(nothing found)");
        ImGui::EndListBox();
    }
    return clicked;
}

}  // namespace

// ------------------------------------------------------------------ ReplayMode

void ReplayMode::restart() {
    for (auto& g : ghosts_) {
        g.s = gdcore::PlayerState{};
        g.tick = 0;
        g.deathAge = -1;
        g.presses.clear();
    }
    accum_ = 0.0;
    tick_ = 0;
    scrub_ = 0;
}

void ReplayMode::enter(App&) {
    SetTargetFPS(60);
    if (!ghosts_.empty()) return;  // already loaded; keep replay position
    if (!path_.empty()) {
        if (path_.size() > 5 && path_.compare(path_.size() - 5, 5, ".json") == 0)
            openSolution(path_);
        else
            openTrace(path_);
    }
}

void ReplayMode::onLevelChanged(App&) {
    restart();  // geometry changed: positions are only valid from tick 0
}

void ReplayMode::openTrace(const std::string& path) {
    path_ = path;
    ghosts_.clear();
    error_.clear();
    maxTicks_ = 0;
    std::vector<TraceEpisode> eps;
    if (!readGdTrace(path, eps, error_)) return;
    for (size_t i = 0; i < eps.size(); ++i) {
        Ghost g;
        g.holds = std::move(eps[i].holds);
        g.finalX = eps[i].finalX;
        g.won = eps[i].won;
        g.tint = ghostTint((int)i);
        maxTicks_ = std::max(maxTicks_, (int)g.holds.size());
        ghosts_.push_back(std::move(g));
    }
    restart();
}

void ReplayMode::openSolution(const std::string& path) {
    path_ = path;
    ghosts_.clear();
    error_.clear();
    maxTicks_ = 0;
    Solution s;
    if (!readSolution(path, s, error_)) return;
    Ghost g;
    g.holds = std::move(s.holds);
    g.finalX = s.endX;
    g.won = true;
    g.tint = Color{255, 220, 60, 255};
    maxTicks_ = (int)g.holds.size();
    ghosts_.push_back(std::move(g));
    restart();
}

void ReplayMode::stepOne(const App& app) {
    ++tick_;
    for (auto& g : ghosts_) {
        if (g.deathAge >= 0) {
            ++g.deathAge;
            continue;
        }
        if (g.tick >= (int)g.holds.size()) continue;  // finished: freeze
        bool hold = g.holds[g.tick] != 0;
        if (hold && (g.tick == 0 || !g.holds[g.tick - 1]))
            g.presses.push_back({g.s.x, -(g.s.y - 90.0f)});
        gdcore::step(g.s, cfg_, app.geom, hold);
        ++g.tick;
        if (g.s.dead) g.deathAge = 0;
    }
}

void ReplayMode::stepTo(const App& app, int t) {
    if (t < tick_) restart();
    while (tick_ < t) stepOne(app);
    scrub_ = (int)tick_;
}

void ReplayMode::update(App& app) {
    if (keyPressed(app, KEY_SPACE)) paused_ = !paused_;
    if (keyPressed(app, KEY_R)) restart();
    if (keyPressed(app, KEY_F)) follow_ = !follow_;
    if (!follow_) freeCamUpdate(app, cam_);
    if (keyPressed(app, KEY_RIGHT) || keyPressed(app, KEY_UP))
        speedIdx_ = std::min(speedIdx_ + 1, kNumSpeeds - 1);
    if (keyPressed(app, KEY_LEFT) || keyPressed(app, KEY_DOWN))
        speedIdx_ = std::max(speedIdx_ - 1, 0);
    if (paused_ || ghosts_.empty()) return;

    accum_ += GetFrameTime() * kSpeeds[speedIdx_] * 240.0;
    // Cap the per-frame work so 32x on long traces stays responsive.
    int ticks = std::min((int)accum_, 240 * 32);
    accum_ -= ticks;
    for (int t = 0; t < ticks; ++t) stepOne(app);
    scrub_ = (int)tick_;
}

void ReplayMode::draw(const App& app) {
    // Leader: furthest visible ghost not dead (finished ones stay eligible).
    const Ghost* lead = nullptr;
    int alive = 0, wonCount = 0, finished = 0;
    for (const auto& g : ghosts_) {
        if (g.deathAge < 0) {
            ++alive;
            if (!g.hidden && (!lead || g.s.x > lead->s.x)) lead = &g;
        }
        if (g.won) ++wonCount;
        if (g.deathAge < 0 && g.tick >= (int)g.holds.size()) ++finished;
    }
    if (!lead && !ghosts_.empty()) lead = &ghosts_[0];
    float lx = lead ? lead->s.x : 0.0f;
    if (follow_)
        followCamera(app, cam_, camY_, lx, lead ? lead->s.y : 105.0f,
                     lead && lead->s.ship, GetFrameTime());
    float colorX = follow_ ? lx : cam_.target.x;

    drawBackground(app, colorX);
    drawWorld(app, cam_, colorX, [&] {
        for (const auto& g : ghosts_) {
            if (g.hidden) continue;
            if (markers_) {
                float s = 5.0f;
                for (const auto& p : g.presses)
                    DrawTriangle({p.x, p.y + 16.0f + s}, {p.x + s, p.y + 16.0f + 2 * s},
                                 {p.x - s, p.y + 16.0f + 2 * s}, Fade(g.tint, 0.8f));
            }
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
        DrawText("pick another file in the panel", 12, 34, 16, Color{170, 175, 200, 255});
        return;
    }
    if (ghosts_.empty()) {
        DrawText("Replay — no trace loaded", 12, 10, 20, RAYWHITE);
        DrawText("open a .gdtrace or solution.json from the panel", 12, 34, 16,
                 Color{170, 175, 200, 255});
        return;
    }
    float pct = app.maxX > 0 ? 100.0f * lx / app.maxX : 0.0f;
    DrawText(TextFormat("Replay [%s]  %zu episodes   alive %d   finished %d   "
                        "wins %d   lead x %.0f (%.0f%%)   tick %lld   %gx%s",
                        std::filesystem::path(path_).filename().string().c_str(),
                        ghosts_.size(), alive, finished, wonCount, lx, pct, tick_,
                        kSpeeds[speedIdx_], paused_ ? "  [paused]" : ""),
             12, 10, 20, RAYWHITE);
    DrawText(TextFormat("space: pause   left/right: speed   R: restart   "
                        "F: %s camera (drag/wheel when free)",
                        follow_ ? "follow" : "FREE"),
             12, 34, 16, Color{170, 175, 200, 255});
}

void ReplayMode::rescan() {
    traces_ = scanReplayFiles();
    solutions_ = scanSolutionFiles();
    lastScan_ = GetTime();
    selTrace_ = std::min(selTrace_, (int)traces_.size() - 1);
    selSolution_ = std::min(selSolution_, (int)solutions_.size() - 1);
}

void ReplayMode::drawPanel(App& app) {
    if (lastScan_ < 0 || GetTime() - lastScan_ > 10.0) rescan();

    if (ImGui::CollapsingHeader("Traces (.gdtrace)", ImGuiTreeNodeFlags_DefaultOpen)) {
        int dbl = fileList("##traces", traces_, selTrace_, 120);
        if (dbl >= 0) openTrace(traces_[dbl].path);
        ImGui::BeginDisabled(selTrace_ < 0);
        if (ImGui::Button("Open trace")) openTrace(traces_[selTrace_].path);
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Rescan##t")) rescan();
    }
    if (ImGui::CollapsingHeader("Solutions (solution.json)", ImGuiTreeNodeFlags_DefaultOpen)) {
        int dbl = fileList("##solutions", solutions_, selSolution_, 90);
        if (dbl >= 0) openSolution(solutions_[dbl].path);
        ImGui::BeginDisabled(selSolution_ < 0);
        if (ImGui::Button("Open solution")) openSolution(solutions_[selSolution_].path);
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Rescan##s")) rescan();
    }
    ImGui::Separator();
    if (!error_.empty()) ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%s", error_.c_str());
    if (ghosts_.empty()) {
        ImGui::TextDisabled("no replay loaded");
        return;
    }
    ImGui::TextWrapped("%s", path_.c_str());
    if (ImGui::Button("Restart")) restart();
    ImGui::SameLine();
    if (ImGui::Button(paused_ ? "Play" : "Pause")) paused_ = !paused_;
    ImGui::SameLine();
    if (ImGui::Button("Step")) { paused_ = true; stepOne(app); scrub_ = (int)tick_; }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    ImGui::Combo("##speed", &speedIdx_, kSpeedLabels, kNumSpeeds);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::SliderInt("##scrub", &scrub_, 0, maxTicks_, "tick %d")) {
        paused_ = true;
        stepTo(app, scrub_);
    }
    ImGui::Checkbox("Follow camera", &follow_);
    ImGui::SameLine();
    ImGui::Checkbox("Jump markers", &markers_);
    if (ImGui::BeginTable("ghosts", 6, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY |
                          ImGuiTableFlags_RowBg, ImVec2(0, 140))) {
        ImGui::TableSetupColumn("#");
        ImGui::TableSetupColumn("show");
        ImGui::TableSetupColumn("x");
        ImGui::TableSetupColumn("final");
        ImGui::TableSetupColumn("tick");
        ImGui::TableSetupColumn("state");
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < ghosts_.size(); ++i) {
            Ghost& g = ghosts_[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ImVec4(g.tint.r / 255.0f, g.tint.g / 255.0f, g.tint.b / 255.0f, 1), "%d", (int)i);
            ImGui::TableNextColumn();
            bool show = !g.hidden;
            ImGui::PushID((int)i);
            if (ImGui::Checkbox("##show", &show)) g.hidden = !show;
            ImGui::PopID();
            ImGui::TableNextColumn();
            ImGui::Text("%.0f", g.s.x);
            ImGui::TableNextColumn();
            ImGui::Text("%.0f%s", g.finalX, g.won ? " W" : "");
            ImGui::TableNextColumn();
            ImGui::Text("%d/%d", g.tick, (int)g.holds.size());
            ImGui::TableNextColumn();
            ImGui::Text("%s", g.deathAge >= 0 ? "dead" : g.tick >= (int)g.holds.size() ? "done" : "alive");
        }
        ImGui::EndTable();
    }
}

// ------------------------------------------------------------ TrainingViewMode

void TrainingViewMode::enter(App&) {
    SetTargetFPS(60);
    // No --watch flag: follow the most recently updated run automatically.
    if (path_.empty()) path_ = newestLiveFile();
}

void TrainingViewMode::watch(const std::string& livePath) {
    path_ = livePath;
    snap_ = Snapshot{};
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

    if (seq != snap_.seq) lastSeqChange_ = GetTime();
    snap_.seq = seq;
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

void TrainingViewMode::update(App& app) {
    if (keyPressed(app, KEY_F)) follow_ = !follow_;
    if (!follow_) freeCamUpdate(app, cam_);
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
        DrawText("start a training from the Train tab (the viewer passes --watch), "
                 "or pick a run in the panel",
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
        followCamera(app, cam_, camY_, leadX, snap_.xyv[leadIdx * 3 + 1], leadShip,
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
    double gameHours = (double)snap_.globalTicks / 240.0 / 3600.0;
    bool stale = GetTime() - lastSeqChange_ > 5.0;
    DrawText(TextFormat("Training [%s]  %zu envs   alive %d   lead x %.0f   "
                        "best %.0f (%.1f%%)   ep %u   %.2fM ticks/s   "
                        "%.1fh game time%s",
                        std::filesystem::path(path_).parent_path().filename().string().c_str(),
                        n, alive, leadX, snap_.bestX, pct, snap_.episodes,
                        snap_.sps / 1e6, gameHours, stale ? "   [STALE]" : ""),
             12, 10, 20, RAYWHITE);
    DrawText(TextFormat("live view of the PPO trainer (read-only)   F: %s camera "
                        "(drag/wheel when free)",
                        follow_ ? "follow" : "FREE"),
             12, 34, 16, Color{170, 175, 200, 255});
}

void TrainingViewMode::rescan() {
    lives_ = scanLiveFiles();
    lastScan_ = GetTime();
}

void TrainingViewMode::drawPanel(App&) {
    if (lastScan_ < 0 || GetTime() - lastScan_ > 10.0) rescan();
    int sel = -1;
    for (int i = 0; i < (int)lives_.size(); ++i)
        if (lives_[i].path == path_) sel = i;
    int before = sel;
    int dbl = fileList("##lives", lives_, sel, 110);
    if (dbl >= 0 || (sel != before && sel >= 0)) watch(lives_[sel].path);
    if (ImGui::Button("Rescan")) rescan();
    ImGui::SameLine();
    if (ImGui::Button("Newest")) watch(newestLiveFile());
    ImGui::SameLine();
    ImGui::Checkbox("Follow camera", &follow_);
    if (!snap_.valid) {
        ImGui::TextDisabled("waiting for live.bin data");
        return;
    }
    size_t n = snap_.ship.size();
    int alive = 0;
    for (size_t i = 0; i < n; ++i) alive += !snap_.dead[i];
    bool stale = GetTime() - lastSeqChange_ > 5.0;
    ImGui::Text("%zu envs, %d alive%s", n, alive, stale ? "   [STALE]" : "   [LIVE]");
    ImGui::Text("best x %.0f   episodes %u", snap_.bestX, snap_.episodes);
    ImGui::Text("%.2fM ticks/s   %.1f h game time", snap_.sps / 1e6,
                (double)snap_.globalTicks / 240.0 / 3600.0);
}

}  // namespace gdv
