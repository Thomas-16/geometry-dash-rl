#include "viewer/gui.hpp"

#include "viewer/levels.hpp"
#include "viewer/modes.hpp"
#include "viewer/paths.hpp"

#include "imgui.h"
#include "rlImGui.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <regex>

namespace gdv {

namespace fs = std::filesystem;

namespace {

const char* const kArchNames[] = {"cnn", "mlp"};
const char* const kAmpNames[] = {"bf16", "none"};

std::string fmtSecs(double s) {
    int t = (int)s;
    char buf[32];
    if (t >= 3600) std::snprintf(buf, sizeof buf, "%dh%02dm", t / 3600, (t / 60) % 60);
    else if (t >= 60) std::snprintf(buf, sizeof buf, "%dm%02ds", t / 60, t % 60);
    else std::snprintf(buf, sizeof buf, "%ds", t);
    return buf;
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

void helpMarker(const char* desc) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

std::string fmtG(float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

}  // namespace

Gui::Gui(Prefs& prefs, Jobs& jobs) : prefs_(prefs), jobs_(jobs) {}

void Gui::init(App& app) {
    rlImGuiSetup(true);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // fixed layout; nothing worth persisting
    io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::GetStyle().WindowRounding = 0.0f;

    panelWidth_ = std::clamp(prefs_.getInt("panel_width", 440), 320, 900);
    panelVisible_ = prefs_.getBool("panel_visible", true);
    stopTimeout_ = prefs_.getInt("stop_timeout", 120);
    std::snprintf(resDirBuf_, sizeof resDirBuf_, "%s", paths::resourcesDir().generic_string().c_str());
    std::snprintf(pythonBuf_, sizeof pythonBuf_, "%s", paths::pythonExe().generic_string().c_str());
    std::snprintf(form_.runName, sizeof form_.runName, "%s",
                  uniqueRunName(prefs_.get("train_run_name", "run")).c_str());
    form_.nEnvs = prefs_.getInt("train_n_envs", form_.nEnvs);
    form_.updates = prefs_.getInt("train_updates", form_.updates);
    form_.endX = prefs_.getFloat("train_end_x", form_.endX);
    (void)app;
    rescanLevels();
}

void Gui::shutdown() {
    rlImGuiShutdown();
}

void Gui::savePrefs(const App& app) {
    prefs_.setInt("panel_width", panelWidth_);
    prefs_.setBool("panel_visible", panelVisible_);
    prefs_.setInt("stop_timeout", stopTimeout_);
    prefs_.setInt("level_id", app.levelId);
    prefs_.setFloat("music_offset", app.musicOffset);
    prefs_.set("resources_dir", paths::resourcesDir().generic_string());
    prefs_.set("python_exe", paths::pythonExe().generic_string());
    prefs_.set("train_run_name", form_.runName);
    prefs_.setInt("train_n_envs", form_.nEnvs);
    prefs_.setInt("train_updates", form_.updates);
    prefs_.setFloat("train_end_x", form_.endX);
    prefs_.setBool("textures", app.useTextures);
    prefs_.setBool("hitboxes", app.showHitboxes);
    prefs_.setBool("live_preview", app.livePreview);
    prefs_.save();
}

void Gui::rescanLevels() {
    levelIds_.clear();
    std::error_code ec;
    for (const auto& l : kLevels)
        if (fs::exists(paths::levelPath(l.id), ec)) levelIds_.push_back(l.id);
}

void Gui::toast(const std::string& msg, float secs) {
    toasts_.push_back({msg, GetTime() + secs});
    if (toasts_.size() > 6) toasts_.erase(toasts_.begin());
}

void Gui::beginFrame(App& app) {
    int W = GetScreenWidth(), H = GetScreenHeight();
    app.viewW = panelVisible_ ? std::max(200, W - panelWidth_) : W;
    app.viewH = H;
    ImGuiIO& io = ImGui::GetIO();
    Vector2 m = GetMousePosition();
    app.guiWantMouse = io.WantCaptureMouse || (panelVisible_ && m.x >= app.viewW);
    app.guiWantKeyboard = io.WantCaptureKeyboard || io.WantTextInput;
    jobs_.poll();
    pollPipelines();
}

// ------------------------------------------------------------------- helpers

JobId Gui::launch(JobSpec spec) {
    std::string err;
    JobId id = jobs_.start(spec, err);
    if (!id) {
        toast("launch failed: " + err, 10);
        return 0;
    }
    if (!selectedJob_ || !jobs_.running(selectedJob_)) selectedJob_ = id;
    return id;
}

JobSpec Gui::pythonSpec(const std::string& name, JobKind kind, const std::string& module,
                        std::vector<std::string> args, const std::string& runName) {
    JobSpec s;
    s.name = name;
    s.kind = kind;
    s.exe = paths::pythonExe();
    s.args = {"-u", "-m", module};
    s.args.insert(s.args.end(), args.begin(), args.end());
    s.cwd = paths::pythonCwd();
    s.python = true;
    s.runName = runName;
    if (!runName.empty()) s.runDir = paths::runsDir() / runName;
    return s;
}

std::string Gui::levelArgFor(const App& app) const {
    std::error_code ec;
    return fs::absolute(app.levelPath, ec).generic_string();
}

std::string Gui::levelArgFor(const RunInfo& r, const App& app) const {
    if (!r.levelPath.empty()) return r.levelPath;
    if (r.levelId == 1 || r.levelId <= 0) return "";  // trainer default = Stereo Madness
    (void)app;
    return paths::levelPath(r.levelId).generic_string();
}

std::vector<std::string> Gui::buildTrainArgs(const App& app) const {
    const TrainForm d;  // defaults
    const TrainForm& f = form_;
    std::vector<std::string> a;
    a.push_back("--run-name");
    a.push_back(f.runName);
    a.push_back("--watch");
    a.push_back("--level");
    a.push_back(levelArgFor(app));
    auto addI = [&](const char* flag, int v, int def) {
        if (v != def) { a.push_back(flag); a.push_back(std::to_string(v)); }
    };
    auto addF = [&](const char* flag, float v, float def) {
        if (v != def) { a.push_back(flag); a.push_back(fmtG(v)); }
    };
    addI("--n-envs", f.nEnvs, d.nEnvs);
    addI("--updates", f.updates, d.updates);
    addF("--end-x", f.endX, d.endX);
    addF("--respawn", f.respawn, d.respawn);
    addF("--lr", f.lr, d.lr);
    addF("--ent-coef", f.entCoef, d.entCoef);
    if (f.arch != d.arch && f.resume[0] == 0) { a.push_back("--arch"); a.push_back(kArchNames[f.arch]); }
    addI("--det-every", f.detEvery, d.detEvery);
    addF("--sil-coef", f.silCoef, d.silCoef);
    if (f.amp != d.amp) { a.push_back("--amp"); a.push_back(kAmpNames[f.amp]); }
    if (f.compile) a.push_back("--compile");
    if (f.resume[0]) { a.push_back("--resume"); a.push_back(f.resume); }
    addI("--ticks-per-action", f.ticksPerAction, d.ticksPerAction);
    addI("--rollout", f.rollout, d.rollout);
    addF("--gamma", f.gamma, d.gamma);
    addF("--reward-scale", f.rewardScale, d.rewardScale);
    addI("--trace-every", f.traceEvery, d.traceEvery);
    addI("--ckpt-every", f.ckptEvery, d.ckptEvery);
    addI("--polish-wins", f.polishWins, d.polishWins);
    addI("--polish-at", f.polishAt, d.polishAt);
    if (f.noPolish) a.push_back("--no-polish");
    addI("--sil-episodes", f.silEpisodes, d.silEpisodes);
    if (f.noGraph) a.push_back("--no-graph");
    if (f.device[0]) { a.push_back("--device"); a.push_back(f.device); }
    if (f.extra[0]) {
        std::string cur;
        for (char c : std::string(f.extra) + " ") {
            if (c == ' ') { if (!cur.empty()) a.push_back(cur); cur.clear(); }
            else cur += c;
        }
    }
    return a;
}

void Gui::startTraining(App& app, bool overwrite) {
    std::string name = trim(form_.runName);
    if (name.empty()) { toast("run name is empty"); return; }
    std::error_code ec;
    if (!overwrite && fs::exists(paths::runsDir() / name, ec)) {
        modalOverwrite_ = true;
        return;
    }
    if (jobs_.runInUse(name)) { toast("a job is already running on " + name); return; }
    JobSpec s = pythonSpec("train " + name, JobKind::Train, "gdrl.train", buildTrainArgs(app), name);
    JobId id = launch(s);
    if (!id) return;
    trainJob_ = id;
    Pipeline p;
    p.runName = name;
    p.runDir = paths::runsDir() / name;
    p.levelArg = levelArgFor(app);
    pipelines_[id] = p;
    trainLog_ = LogCache{};
    toast("training started: " + name);
    // The Training view follows this run's live.bin once it appears.
    app.commands.watchLive = (p.runDir / "live.bin").generic_string();
    lastRunsScan_ = -1.0;
}

void Gui::startEval(const RunInfo& r, const App& app) {
    std::error_code ec;
    fs::path ckpt = r.hasBestDet ? r.dir / "ckpt_best_det.pt"
                  : r.hasBest ? r.dir / "ckpt_best.pt" : r.dir / "ckpt_latest.pt";
    if (!fs::exists(ckpt, ec)) { toast("no checkpoint in " + r.name); return; }
    std::vector<std::string> args = {fs::absolute(ckpt, ec).generic_string()};
    std::string lvl = levelArgFor(r, app);
    if (!lvl.empty()) { args.push_back("--level"); args.push_back(lvl); }
    launch(pythonSpec("eval " + r.name, JobKind::Eval, "gdrl.eval", args, r.name));
    toast("eval started on " + ckpt.filename().string() + " (" + r.name + ")");
}

void Gui::startDeploy(const RunInfo& r, const App& app, bool noVerify) {
    std::error_code ec;
    std::vector<std::string> args = {fs::absolute(r.dir, ec).generic_string()};
    std::string lvl = levelArgFor(r, app);
    if (!lvl.empty()) { args.push_back("--level"); args.push_back(lvl); }
    if (noVerify) args.push_back("--no-verify");
    launch(pythonSpec("deploy " + r.name, JobKind::Deploy, "gdrl.deploy", args, r.name));
    toast("deploy started: " + r.name);
}

void Gui::startCheckTrace(const RunInfo& r, const App& app) {
    modTraces_ = scanModTraces();
    lastModScan_ = GetTime();
    if (modTraces_.empty()) { toast("no attempt_NNNN.csv under " + paths::modTracesDir().generic_string()); return; }
    const ModTrace& t = modTraces_.front();
    std::error_code ec;
    std::string lvl = levelArgFor(r, app);
    if (lvl.empty()) lvl = paths::levelPath(1).generic_string();
    JobSpec tc;
    tc.name = "trace_check " + t.csv.filename().string();
    tc.kind = JobKind::Tool;
    tc.exe = paths::traceCheckExe();
    tc.args = {t.csv.generic_string()};
    tc.cwd = paths::repoRoot();
    tc.env = {{"GD_LEVEL_PATH", lvl}};
    launch(tc);
    std::vector<std::string> args = {"--check-trace", t.csv.generic_string(),
                                     fs::absolute(r.dir, ec).generic_string()};
    launch(pythonSpec("check-trace " + r.name, JobKind::Tool, "gdrl.deploy", args, r.name));
    if (t.hasResult)
        toast(t.csv.filename().string() + ": " + t.result.outcome + " at tick " +
              std::to_string(t.result.tick) + " (" + fmtG(t.result.pct) + "%)", 10);
}

void Gui::prefillResume(const RunInfo& r) {
    std::error_code ec;
    fs::path ckpt = r.hasLatest ? r.dir / "ckpt_latest.pt" : r.dir / "ckpt_best.pt";
    std::snprintf(form_.runName, sizeof form_.runName, "%s", r.name.c_str());
    std::snprintf(form_.resume, sizeof form_.resume, "%s", fs::absolute(ckpt, ec).generic_string().c_str());
    if (r.nEnvs > 0) form_.nEnvs = r.nEnvs;
    if (r.updates > 0) form_.updates = r.updates;
    form_.endX = r.endX > 0 && r.endX < 1e8f ? r.endX : -1.0f;
    form_.arch = r.arch == "mlp" ? 1 : 0;
    form_.amp = r.amp == "none" ? 1 : 0;
    tab_ = 1;
    wantTab_ = true;
    toast("Train tab prefilled to resume " + r.name + " (same run dir: log.csv is appended)");
}

// ------------------------------------------------------------------ pipeline

void Gui::pollPipelines() {
    static const std::regex kUpd(
        R"(^upd\s+(\d+) \| x\s+([-\d.]+) avg\s+([-\d.]+) max \(best\s+([-\d.]+),\s+([-\d.]+)%\) \| win ([\d.]+)% \| ([\d.]+)M ticks/s \| ([\d.]+)k dec/s \| ent ([\d.]+))");
    static const std::regex kDet(R"(\| det\s+([-\d.]+)( WIN)?)");
    std::vector<std::string> fresh;
    for (auto& [id, p] : pipelines_) {
        size_t n = jobs_.lineCount(id);
        if (n <= p.cursor) continue;
        fresh.clear();
        jobs_.lines(id, p.cursor, fresh);
        p.cursor = n;
        for (const auto& line : fresh) {
            std::smatch m;
            if (std::regex_search(line, m, kUpd)) {
                p.status.update = std::atoi(m[1].str().c_str());
                p.status.meanX = std::strtof(m[2].str().c_str(), nullptr);
                p.status.maxX = std::strtof(m[3].str().c_str(), nullptr);
                p.status.bestX = std::strtof(m[4].str().c_str(), nullptr);
                p.status.bestPct = std::strtof(m[5].str().c_str(), nullptr);
                p.status.winRate = std::strtof(m[6].str().c_str(), nullptr);
                p.status.sps = std::strtof(m[7].str().c_str(), nullptr);
                p.status.decS = std::strtof(m[8].str().c_str(), nullptr);
                p.status.entropy = std::strtof(m[9].str().c_str(), nullptr);
                std::smatch d;
                if (std::regex_search(line, d, kDet)) {
                    p.status.detX = std::strtof(d[1].str().c_str(), nullptr);
                    p.status.detWin = d[2].matched;
                    if (p.status.detWin && !p.sawWin) {
                        p.sawWin = true;
                        toast(p.runName + ": deterministic WIN at update " +
                              std::to_string(p.status.update), 10);
                    }
                    if (p.status.detWin && !p.evalLaunched) {
                        // ckpt_best_det.pt is written before this line is
                        // printed (train.py), so it's safe to evaluate now.
                        p.evalLaunched = true;
                        std::vector<std::string> args = {
                            (p.runDir / "ckpt_best_det.pt").generic_string()};
                        if (!p.levelArg.empty()) { args.push_back("--level"); args.push_back(p.levelArg); }
                        launch(pythonSpec("eval " + p.runName + " (auto)", JobKind::Eval,
                                          "gdrl.eval", args, p.runName));
                        toast("auto-eval launched for " + p.runName, 8);
                    }
                }
            } else if (line.rfind("explore:", 0) == 0) {
                p.status.phase = line;
            } else if (line.find("POLISH") != std::string::npos ||
                       line.find("EXPLORE:") != std::string::npos ||
                       line.find("COLLAPSE") != std::string::npos ||
                       line.rfind("stopped:", 0) == 0 || line.rfind("done:", 0) == 0 ||
                       line.rfind("resumed from", 0) == 0 || line.rfind("WARNING", 0) == 0 ||
                       line.rfind("Traceback", 0) == 0) {
                p.status.phase = line;
            }
        }
    }
}

void Gui::jobExited(const JobInfo& info) {
    std::string status = info.status == JobStatus::Killed ? "killed"
                         : info.exitCode == 0 ? "finished" : "FAILED (exit " + std::to_string(info.exitCode) + ")";
    lastRunsScan_ = -1.0;  // rescan runs on the next frame
    if (info.spec.kind == JobKind::Train) {
        auto it = pipelines_.find(info.id);
        if (it != pipelines_.end()) {
            Pipeline& p = it->second;
            std::error_code ec;
            bool wins = p.sawWin || fs::exists(p.runDir / "wins.gdtrace", ec);
            fs::path sol = p.runDir / "solution.json", det = p.runDir / "ckpt_best_det.pt";
            bool solStale = !fs::exists(sol, ec) ||
                            (fs::exists(det, ec) && fs::last_write_time(det, ec) > fs::last_write_time(sol, ec));
            if (info.status != JobStatus::Killed && wins && fs::exists(det, ec) && solStale) {
                std::vector<std::string> args = {det.generic_string()};
                if (!p.levelArg.empty()) { args.push_back("--level"); args.push_back(p.levelArg); }
                launch(pythonSpec("eval " + p.runName + " (final)", JobKind::Eval, "gdrl.eval", args, p.runName));
                toast("training " + status + " with wins — final eval launched", 8);
            } else {
                toast("training " + p.runName + " " + status + (wins ? "" : " (no det win)"), 8);
            }
        }
        if (trainJob_ == info.id) trainJob_ = 0;
        return;
    }
    if (info.spec.kind == JobKind::Eval) {
        std::string tail = jobs_.joined(info.id);
        if (tail.find("WROTE SOLUTION:") != std::string::npos)
            toast(info.spec.runName + ": solution.json ready — Deploy from the Runs tab", 12);
        else
            toast(info.spec.runName + ": eval " + status + ", no winning episode", 10);
        return;
    }
    if (info.spec.kind == JobKind::Deploy) {
        std::string tail = jobs_.joined(info.id);
        if (tail.find("installed ->") != std::string::npos)
            toast(info.spec.runName + ": installed into the mod's solutions/ — play the level in GD", 12);
        else
            toast(info.spec.runName + ": deploy " + status + " (see Jobs log)", 10);
        return;
    }
    toast(info.spec.name + " " + status, 6);
}

// ---------------------------------------------------------------------- draw

void Gui::draw(App& app, Mode* const* modes, int modeCount, int cur) {
    if (!panelVisible_) {
        ImGui::SetNextWindowPos(ImVec2((float)GetScreenWidth() - 110, 8));
        ImGui::Begin("##showpanel", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground);
        if (ImGui::Button("Show panel")) panelVisible_ = true;
        ImGui::End();
        drawToasts(app);
        return;
    }
    int W = GetScreenWidth(), H = GetScreenHeight();
    ImGui::SetNextWindowPos(ImVec2((float)app.viewW, 0));
    ImGui::SetNextWindowSize(ImVec2((float)(W - app.viewW), (float)H));
    ImGui::Begin("##panel", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Splitter on the panel's left edge.
    ImGui::SetCursorPos(ImVec2(0, 0));
    ImGui::InvisibleButton("##splitter", ImVec2(6, (float)H));
    if (ImGui::IsItemActive()) panelWidth_ = std::clamp(panelWidth_ - (int)ImGui::GetIO().MouseDelta.x, 320, 900);
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::SetCursorPos(ImVec2(10, 8));
    ImGui::BeginGroup();

    drawHeader(app, modes, modeCount, cur);

    if (ImGui::BeginTabBar("tabs")) {
        const char* names[] = {"View", "Train", "Runs", "Tools", "Jobs"};
        for (int i = 0; i < 5; ++i) {
            ImGuiTabItemFlags fl = (wantTab_ && tab_ == i) ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem(names[i], nullptr, fl)) {
                if (!wantTab_) tab_ = i;
                ImGui::BeginChild("##tabbody", ImVec2(0, 0), ImGuiChildFlags_None,
                                  ImGuiWindowFlags_NoSavedSettings);
                switch (i) {
                    case 0: drawViewTab(app, modes, cur); break;
                    case 1: drawTrainTab(app); break;
                    case 2: drawRunsTab(app); break;
                    case 3: drawToolsTab(app); break;
                    case 4: drawJobsTab(app); break;
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
        }
        wantTab_ = false;
        ImGui::EndTabBar();
    }
    ImGui::EndGroup();
    drawModals(app);
    ImGui::End();
    drawToasts(app);
}

void Gui::drawHeader(App& app, Mode* const* modes, int modeCount, int cur) {
    // Mode buttons
    for (int i = 0; i < modeCount; ++i) {
        if (i) ImGui::SameLine();
        bool active = i == cur;
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::Button(modes[i]->name(), ImVec2(0, 0)) && !active) app.commands.switchMode = i;
        if (active) ImGui::PopStyleColor();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("hide")) panelVisible_ = false;

    // Level combo
    {
        const LevelInfo* curInfo = levelInfo(app.levelId);
        std::string curLabel = curInfo ? std::to_string(curInfo->id) + "  " + curInfo->name
                                       : fs::path(app.levelPath).filename().string();
        ImGui::SetNextItemWidth(-90);
        if (ImGui::BeginCombo("Level", curLabel.c_str())) {
            for (int id : levelIds_) {
                const LevelInfo* l = levelInfo(id);
                std::string label = std::to_string(id) + "  " + l->name +
                                    (l->verified ? "  [verified]" : "  [sim unverified]");
                if (ImGui::Selectable(label.c_str(), id == app.levelId) && id != app.levelId)
                    app.commands.loadLevel = paths::levelPath(id).generic_string();
            }
            ImGui::EndCombo();
        }
        if (curInfo && !curInfo->verified)
            ImGui::TextColored(ImVec4(1, 0.75f, 0.3f, 1),
                               "sim unverified for this level (no real-game traces checked yet)");
    }

    // View toggles
    ImGui::BeginDisabled(!app.textured);
    ImGui::Checkbox("Textures", &app.useTextures);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Checkbox("Debug markers", &app.debugMarkers);
    ImGui::SameLine();
    ImGui::Checkbox("Hitboxes", &app.showHitboxes);

    ImGui::Text("Music offset %+d ms", (int)std::lround(app.musicOffset * 1000.0f));
    ImGui::SameLine();
    if (ImGui::SmallButton("-10")) app.musicOffset -= 0.010f;
    ImGui::SameLine();
    if (ImGui::SmallButton("+10")) app.musicOffset += 0.010f;
    ImGui::SameLine();
    if (ImGui::SmallButton("0")) app.musicOffset = 0.0f;
    ImGui::SameLine();
    if (ImGui::SmallButton("Fullscreen")) app.commands.fullscreen = true;
    ImGui::SameLine();
    if (ImGui::SmallButton("Screenshot")) app.commands.screenshot = true;
    ImGui::Separator();
}

void Gui::drawKeys() {
    if (ImGui::CollapsingHeader("Keyboard shortcuts (optional)")) {
        ImGui::TextDisabled("TAB / 1-4   mode      T textures   D debug   B hitboxes");
        ImGui::TextDisabled("H panel     F11 fullscreen   F12 screenshot   [ ] music offset");
        ImGui::TextDisabled("Browse: drag/wheel, SPACE autoscroll, R reset");
        ImGui::TextDisabled("Play:   SPACE/UP/click jump, R restart, P pause");
        ImGui::TextDisabled("Replay: SPACE pause, LEFT/RIGHT speed, R restart, F camera");
        ImGui::TextDisabled("Training: F camera");
    }
}

void Gui::drawViewTab(App& app, Mode* const* modes, int cur) {
    ImGui::Text("%s", modes[cur]->name());
    ImGui::Separator();
    modes[cur]->drawPanel(app);
    ImGui::Separator();
    drawKeys();
}

// ----------------------------------------------------------------- Train tab

void Gui::refreshLog(LogCache& c, const fs::path& p) {
    double t = GetTime();
    if (c.path == p && c.lastCheck >= 0 && t - c.lastCheck < 1.0) return;
    c.lastCheck = t;
    std::error_code ec;
    auto mt = fs::last_write_time(p, ec);
    uintmax_t sz = ec ? 0 : fs::file_size(p, ec);
    if (c.path == p && mt == c.mtime && sz == c.size) return;
    c.path = p;
    c.mtime = mt;
    c.size = sz;
    c.rows.clear();
    c.mean.clear(); c.best.clear(); c.det.clear(); c.win.clear();
    if (ec || !readLogCsv(p, c.rows)) return;
    float lastDet = 0.0f;
    for (const auto& r : c.rows) {
        c.mean.push_back(r.meanFinalX);
        c.best.push_back(r.bestXEver);
        if (r.detX >= 0) lastDet = r.detX;
        c.det.push_back(lastDet);
        c.win.push_back(r.winRate * 100.0f);
    }
}

static void plotRows(const char* label, const std::vector<float>& v, const char* fmt) {
    if (v.empty()) return;
    float mx = *std::max_element(v.begin(), v.end());
    char overlay[64];
    std::snprintf(overlay, sizeof overlay, fmt, v.back());
    ImGui::PlotLines(label, v.data(), (int)v.size(), 0, overlay, 0.0f, mx <= 0 ? 1.0f : mx * 1.05f,
                     ImVec2(-FLT_MIN, 60));
}

void Gui::drawTrainTab(App& app) {
    const LevelInfo* li = levelInfo(app.levelId);
    if (li && !li->verified)
        ImGui::TextColored(ImVec4(1, 0.75f, 0.3f, 1), "training on an unverified level");
    bool running = trainJob_ && jobs_.running(trainJob_);

    ImGui::BeginDisabled(running);
    ImGui::SetNextItemWidth(-110);
    ImGui::InputText("run name", form_.runName, sizeof form_.runName);
    {
        std::error_code ec;
        std::string name = trim(form_.runName);
        if (!name.empty() && fs::exists(paths::runsDir() / name, ec)) {
            ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "exists: starting overwrites it (log.csv is truncated)");
            ImGui::SameLine();
            if (ImGui::SmallButton("uniquify")) {
                std::string u = uniqueRunName(name);
                std::snprintf(form_.runName, sizeof form_.runName, "%s", u.c_str());
            }
        }
    }
    ImGui::SetNextItemWidth(120); ImGui::InputInt("n-envs", &form_.nEnvs, 64, 256);
    ImGui::SameLine(); ImGui::SetNextItemWidth(110); ImGui::InputInt("updates", &form_.updates, 10, 100);
    helpMarker("Don't shrink --updates for a regression check: the lr/entropy anneal spans the whole run.");
    ImGui::SetNextItemWidth(120); ImGui::InputFloat("end-x", &form_.endX, 0, 0, "%g");
    helpMarker("-1 = full level. A curriculum end-x makes solutions that deploy.py cannot verify.");
    ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::InputFloat("respawn", &form_.respawn, 0, 0, "%g");
    ImGui::SetNextItemWidth(120); ImGui::InputFloat("lr", &form_.lr, 0, 0, "%g");
    ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::InputFloat("ent-coef", &form_.entCoef, 0, 0, "%g");
    ImGui::SetNextItemWidth(120); ImGui::Combo("arch", &form_.arch, kArchNames, 2);
    ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::Combo("amp", &form_.amp, kAmpNames, 2);
    ImGui::SetNextItemWidth(120); ImGui::InputInt("det-every", &form_.detEvery, 1, 10);
    ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::InputFloat("sil-coef", &form_.silCoef, 0, 0, "%g");
    ImGui::Checkbox("--compile (needs triton-windows)", &form_.compile);
    ImGui::SetNextItemWidth(-110);
    ImGui::InputText("resume from", form_.resume, sizeof form_.resume);
    helpMarker("checkpoint .pt path; arch comes from the checkpoint. Use 'Resume' in the Runs tab to prefill.");
    if (ImGui::TreeNode("Advanced")) {
        ImGui::SetNextItemWidth(100); ImGui::InputInt("ticks-per-action", &form_.ticksPerAction);
        ImGui::SetNextItemWidth(100); ImGui::InputInt("rollout", &form_.rollout);
        ImGui::SetNextItemWidth(100); ImGui::InputFloat("gamma", &form_.gamma, 0, 0, "%g");
        ImGui::SetNextItemWidth(100); ImGui::InputFloat("reward-scale", &form_.rewardScale, 0, 0, "%g");
        ImGui::SetNextItemWidth(100); ImGui::InputInt("trace-every", &form_.traceEvery);
        ImGui::SetNextItemWidth(100); ImGui::InputInt("ckpt-every", &form_.ckptEvery);
        ImGui::SetNextItemWidth(100); ImGui::InputInt("polish-wins", &form_.polishWins);
        ImGui::SetNextItemWidth(100); ImGui::InputInt("polish-at", &form_.polishAt);
        ImGui::Checkbox("--no-polish", &form_.noPolish);
        ImGui::SameLine();
        ImGui::Checkbox("--no-graph", &form_.noGraph);
        ImGui::SetNextItemWidth(100); ImGui::InputInt("sil-episodes", &form_.silEpisodes);
        ImGui::SetNextItemWidth(100); ImGui::InputText("device", form_.device, sizeof form_.device);
        ImGui::SetNextItemWidth(-80); ImGui::InputText("extra flags", form_.extra, sizeof form_.extra);
        if (ImGui::Button("Reset form to defaults")) {
            TrainForm d;
            std::string keep = form_.runName;
            form_ = d;
            std::snprintf(form_.runName, sizeof form_.runName, "%s", keep.c_str());
        }
        ImGui::TreePop();
    }
    ImGui::EndDisabled();

    {
        std::string cmd = "python -u -m gdrl.train";
        for (const auto& a : buildTrainArgs(app)) cmd += " " + paths::quoteArg(a);
        ImGui::InputText("##cmd", cmd.data(), cmd.size() + 1, ImGuiInputTextFlags_ReadOnly);
        ImGui::SameLine();
        if (ImGui::SmallButton("Copy")) ImGui::SetClipboardText(cmd.c_str());
    }

    if (!running) {
        if (ImGui::Button("Start training", ImVec2(150, 0))) startTraining(app, false);
        ImGui::SameLine();
        ImGui::TextDisabled("--watch is added: the Training view follows it");
    } else {
        const JobInfo* ji = jobs_.info(trainJob_);
        if (ImGui::Button(ji && ji->stopRequested ? "Kill now" : "Stop (graceful)", ImVec2(150, 0)))
            jobs_.stop(trainJob_, (double)stopTimeout_);
        ImGui::SameLine();
        if (ImGui::Button("Watch live")) {
            app.commands.watchLive = (paths::runsDir() / trim(form_.runName) / "live.bin").generic_string();
            app.commands.switchMode = 2;
        }
        ImGui::SameLine();
        if (ImGui::Button("Log")) { selectedJob_ = trainJob_; tab_ = 4; wantTab_ = true; }
        ImGui::SameLine();
        ImGui::Checkbox("Live preview", &app.livePreview);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Off: the Training view stops reading live.bin and drawing the ghosts\n"
                              "and the viewer idles at 30 FPS (15 when unfocused), so the trainer\n"
                              "keeps the whole GPU. The stats below and the plots keep updating.");
        if (ji) ImGui::Text("running %s", fmtSecs(Jobs::now() - ji->startTime).c_str());
    }

    // Status of the active (or last) training pipeline.
    JobId showId = trainJob_;
    if (!showId && !pipelines_.empty()) showId = pipelines_.rbegin()->first;
    auto it = pipelines_.find(showId);
    if (it != pipelines_.end()) {
        const Pipeline& p = it->second;
        const TrainStatus& s = p.status;
        ImGui::Separator();
        ImGui::Text("%s", p.runName.c_str());
        if (s.update >= 0) {
            ImGui::Text("upd %d   mean x %.0f   max %.0f   best %.0f (%.1f%%)", s.update, s.meanX, s.maxX,
                        s.bestX, s.bestPct);
            ImGui::Text("win %.2f%%   %.2fM ticks/s   %.1fk dec/s   ent %.3f", s.winRate, s.sps, s.decS,
                        s.entropy);
            if (s.detX >= 0)
                ImGui::TextColored(s.detWin ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 1, 1, 1),
                                   "det x %.0f%s", s.detX, s.detWin ? "  WIN" : "");
        } else {
            ImGui::TextDisabled("starting up (DLL load, env creation, explore scout)...");
        }
        if (!s.phase.empty()) ImGui::TextWrapped("%s", s.phase.c_str());
        if (p.sawWin) ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "deterministic win seen%s",
                                         p.evalLaunched ? " — eval launched" : "");
        refreshLog(trainLog_, p.runDir / "log.csv");
        if (!trainLog_.rows.empty()) {
            plotRows("mean x", trainLog_.mean, "%.0f");
            plotRows("best x", trainLog_.best, "%.0f");
            plotRows("det x", trainLog_.det, "%.0f");
            plotRows("win %", trainLog_.win, "%.2f%%");
        }
        if (ImGui::SmallButton("Open run folder")) paths::openInExplorer(p.runDir);
    }
}

// ------------------------------------------------------------------ Runs tab

void Gui::rescanRuns(bool force) {
    double t = GetTime();
    if (!force && lastRunsScan_ >= 0 && t - lastRunsScan_ < 5.0) return;
    lastRunsScan_ = t;
    runs_ = scanRuns();
    // live.bin freshness: the seq counter advancing between scans means a
    // trainer (ours or an external one) is publishing right now.
    for (const auto& r : runs_) {
        if (!r.hasLive) continue;
        auto it = liveSeen_.find(r.name);
        if (it == liveSeen_.end()) liveSeen_[r.name] = {r.liveSeq, -1e9};  // first sight: not live yet
        else if (it->second.first != r.liveSeq) it->second = {r.liveSeq, t};
    }
}

bool Gui::isLive(const std::string& run) const {
    auto it = liveSeen_.find(run);
    return it != liveSeen_.end() && GetTime() - it->second.second < 15.0;
}

const RunInfo* Gui::selectedRun() const {
    for (const auto& r : runs_)
        if (r.name == selectedRun_) return &r;
    return nullptr;
}

void Gui::drawRunsTab(App& app) {
    rescanRuns(false);
    if (ImGui::SmallButton("Rescan")) rescanRuns(true);
    ImGui::SameLine();
    if (ImGui::SmallButton("Open runs/")) paths::openInExplorer(paths::runsDir());
    ImGui::SameLine();
    ImGui::TextDisabled("%d runs", (int)runs_.size());

    if (ImGui::BeginTable("runs", 6,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerH |
                              ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0, 200))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("run", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("lvl");
        ImGui::TableSetupColumn("upd");
        ImGui::TableSetupColumn("best%");
        ImGui::TableSetupColumn("det");
        ImGui::TableSetupColumn("has");
        ImGui::TableHeadersRow();
        for (const auto& r : runs_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool sel = r.name == selectedRun_;
            std::string label = r.name;
            if (jobs_.runInUse(r.name)) label += " [RUNNING]";
            else if (isLive(r.name)) label += " [LIVE]";
            if (ImGui::Selectable(label.c_str(), sel, ImGuiSelectableFlags_SpanAllColumns)) selectedRun_ = r.name;
            ImGui::TableNextColumn();
            ImGui::Text("%d", r.levelId);
            ImGui::TableNextColumn();
            if (r.hasLog) ImGui::Text("%d/%d", r.last.update + 1, r.updates);
            else ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            if (r.hasLog && r.endX > 0) ImGui::Text("%.1f", 100.0f * r.last.bestXEver / r.endX);
            else ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            if (r.detWins) ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "WIN x%d", r.detWins);
            else if (r.bestDetX >= 0 && r.endX > 0) ImGui::Text("%.0f%%", 100.0f * r.bestDetX / r.endX);
            else ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            ImGui::Text("%s%s%s%s", r.hasBestDet ? "D" : r.hasBest ? "b" : "-", r.hasLatest ? "L" : "-",
                        r.hasSolution ? "S" : "-", r.hasWins ? "W" : "-");
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("has: D=ckpt_best_det (b=best only) L=latest S=solution W=wins");

    const RunInfo* r = selectedRun();
    if (!r) {
        ImGui::TextDisabled("select a run");
        return;
    }
    ImGui::Separator();
    ImGui::Text("%s", r->name.c_str());
    if (r->hasConfig)
        ImGui::TextDisabled("level %d  n_envs %d  updates %d  arch %s  amp %s  end_x %.0f", r->levelId, r->nEnvs,
                            r->updates, r->arch.c_str(), r->amp.c_str(), r->endX);
    if (r->hasLog)
        ImGui::TextDisabled("last: upd %d  mean %.0f  best %.0f  win %.2f%%  det %.0f%s", r->last.update,
                            r->last.meanFinalX, r->last.bestXEver, r->last.winRate * 100.0f, r->last.detX,
                            r->last.detWin ? " WIN" : "");
    bool busy = jobs_.runInUse(r->name);
    if (busy) ImGui::TextColored(ImVec4(1, 0.8f, 0.4f, 1), "a job is running on this run");

    // playback
    if (r->hasLive && ImGui::Button("Watch live")) {
        app.commands.watchLive = (r->dir / "live.bin").generic_string();
        app.commands.switchMode = 2;
    }
    if (r->hasLive) ImGui::SameLine();
    ImGui::BeginDisabled(!r->hasWins);
    if (ImGui::Button("Replay wins")) { app.commands.replayTrace = (r->dir / "wins.gdtrace").generic_string(); app.commands.switchMode = 3; }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!r->hasEval);
    if (ImGui::Button("Replay eval")) { app.commands.replayTrace = (r->dir / "eval.gdtrace").generic_string(); app.commands.switchMode = 3; }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!r->hasSolution);
    if (ImGui::Button("Replay solution")) { app.commands.replaySolution = (r->dir / "solution.json").generic_string(); app.commands.switchMode = 3; }
    ImGui::EndDisabled();

    // pipeline
    ImGui::BeginDisabled(busy || !(r->hasBestDet || r->hasBest || r->hasLatest));
    if (ImGui::Button("Eval -> solution.json")) startEval(*r, app);
    ImGui::EndDisabled();
    helpMarker("python -m gdrl.eval on ckpt_best_det.pt (else ckpt_best.pt): 64 stochastic + 1 deterministic "
               "episodes; writes solution.json if any wins, plus eval.gdtrace.");
    ImGui::SameLine();
    ImGui::BeginDisabled(busy || !r->hasSolution);
    if (ImGui::Button("Deploy to GD...")) { modalDeployRun_ = r->name; modalDeployNoVerify_ = false; }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(busy || !r->hasSolution);
    if (ImGui::Button("Check newest trace")) startCheckTrace(*r, app);
    ImGui::EndDisabled();
    helpMarker("Runs trace_check.exe and deploy --check-trace on the newest attempt_NNNN.csv the replay mod "
               "recorded (mod save dir/traces).");

    ImGui::BeginDisabled(busy || !(r->hasLatest || r->hasBest));
    if (ImGui::Button("Resume training...")) prefillResume(*r);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Open folder")) paths::openInExplorer(r->dir);
    ImGui::SameLine();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Delete...")) modalDeleteRun_ = r->name;
    ImGui::EndDisabled();

    // log plots
    if (r->hasLog) {
        refreshLog(runLog_, r->dir / "log.csv");
        if (!runLog_.rows.empty() && ImGui::TreeNode("log.csv plots")) {
            plotRows("mean x", runLog_.mean, "%.0f");
            plotRows("best x", runLog_.best, "%.0f");
            plotRows("det x", runLog_.det, "%.0f");
            plotRows("win %", runLog_.win, "%.2f%%");
            ImGui::TreePop();
        }
    }

    // newest mod results
    if (ImGui::TreeNode("Real-game attempts (replay mod)")) {
        if (lastModScan_ < 0 || GetTime() - lastModScan_ > 5.0) {
            modTraces_ = scanModTraces();
            lastModScan_ = GetTime();
        }
        if (modTraces_.empty()) ImGui::TextDisabled("none under %s", paths::modTracesDir().generic_string().c_str());
        int shown = 0;
        for (const auto& t : modTraces_) {
            if (shown++ >= 8) break;
            if (t.hasResult)
                ImGui::Text("%s: %s at tick %lld, x %.0f (%.1f%%)", t.csv.filename().string().c_str(),
                            t.result.outcome.c_str(), t.result.tick, t.result.x, t.result.pct);
            else
                ImGui::Text("%s: (no result json)", t.csv.filename().string().c_str());
        }
        if (ImGui::SmallButton("Open traces folder")) paths::openInExplorer(paths::modTracesDir());
        ImGui::TreePop();
    }
}

// ----------------------------------------------------------------- Tools tab

void Gui::drawToolsTab(App& app) {
    bool py = jobs_.anyPythonRunning();
    std::string lvl = levelArgFor(app);

    ImGui::Text("Checks");
    if (ImGui::Button("env_check det")) {
        JobSpec s;
        s.name = "env_check det";
        s.exe = paths::envCheckExe();
        s.args = {"det"};
        s.cwd = paths::repoRoot();
        s.env = {{"GD_LEVEL_PATH", lvl}};
        launch(s);
    }
    ImGui::SameLine();
    if (ImGui::Button("selftest")) launch(pythonSpec("selftest", JobKind::Tool, "gdrl.selftest", {}));
    ImGui::SameLine();
    if (ImGui::Button("dump level")) {
        JobSpec s;
        s.name = "dump";
        s.exe = paths::dumpExe();
        s.args = {lvl};
        s.cwd = paths::repoRoot();
        launch(s);
    }
    helpMarker("env_check det + selftest are the env-layer regression checks (CLAUDE.md). dump must report "
               "2291 objects / 31 ids for Stereo Madness.");

    ImGui::Separator();
    ImGui::Text("Bench");
    ImGui::SetNextItemWidth(-90);
    ImGui::InputText("compare", benchCompare_, sizeof benchCompare_);
    ImGui::SetNextItemWidth(-90);
    ImGui::InputText("save", benchSave_, sizeof benchSave_);
    if (jobs_.anyRunning(JobKind::Train))
        ImGui::TextColored(ImVec4(1, 0.8f, 0.4f, 1), "a trainer is running: bench numbers will be skewed");
    if (ImGui::Button("Run bench")) {
        std::vector<std::string> a;
        if (benchCompare_[0]) { a.push_back("--compare"); a.push_back(benchCompare_); }
        if (benchSave_[0]) { a.push_back("--save"); a.push_back(benchSave_); }
        launch(pythonSpec("bench", JobKind::Tool, "gdrl.bench", a));
    }

    ImGui::Separator();
    ImGui::Text("Build");
    ImGui::BeginDisabled(py);
    if (ImGui::Button("Rebuild (all but viewer)")) {
        JobSpec s;
        s.name = "cmake --build (no viewer)";
        s.exe = "cmake";
        s.args = {"--build", paths::buildDir().generic_string(), "--target", "gdcore", "gdenv", "env_check",
                  "dump", "trace_check"};
        s.cwd = paths::repoRoot();
        launch(s);
    }
    ImGui::SameLine();
    if (ImGui::Button("Rebuild incl. viewer...")) modalRebuildViewer_ = true;
    ImGui::EndDisabled();
    if (py) ImGui::TextDisabled("disabled while Python jobs hold gdenv.dll");

    ImGui::Separator();
    ImGui::Text("Folders");
    if (ImGui::Button("runs/")) paths::openInExplorer(paths::runsDir());
    ImGui::SameLine();
    if (ImGui::Button("mod save dir")) paths::openInExplorer(paths::modDir());
    ImGui::SameLine();
    if (ImGui::Button("mod solutions/")) paths::openInExplorer(paths::modSolutionsDir());
    ImGui::SameLine();
    if (ImGui::Button("repo")) paths::openInExplorer(paths::repoRoot());

    ImGui::Separator();
    ImGui::Text("Settings");
    ImGui::SetNextItemWidth(-100);
    if (ImGui::InputText("python exe", pythonBuf_, sizeof pythonBuf_, ImGuiInputTextFlags_EnterReturnsTrue))
        paths::setPythonExe(pythonBuf_);
    ImGui::SetNextItemWidth(-100);
    if (ImGui::InputText("resources", resDirBuf_, sizeof resDirBuf_, ImGuiInputTextFlags_EnterReturnsTrue)) {
        paths::setResourcesDir(resDirBuf_);
        rescanLevels();
    }
    ImGui::TextDisabled("(Enter to apply)");
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("stop timeout (s)", &stopTimeout_, 10, 60);
    helpMarker("Stop writes runs/<name>/STOP; the trainer saves ckpt_latest and exits at the end of the "
               "current update. After this many seconds it is killed instead.");
    ImGui::TextDisabled("repo: %s", paths::repoRoot().generic_string().c_str());
    ImGui::TextDisabled("prefs saved on exit to viewer_prefs.ini");
}

// ------------------------------------------------------------------ Jobs tab

void Gui::drawJobsTab(App&) {
    auto ids = jobs_.ids();
    if (ImGui::SmallButton("Clear finished")) {
        jobs_.clearFinished();
        ids = jobs_.ids();
        if (!jobs_.info(selectedJob_)) selectedJob_ = 0;
    }
    if (ImGui::BeginTable("jobs", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0, 130))) {
        ImGui::TableSetupColumn("job", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("status");
        ImGui::TableSetupColumn("time");
        ImGui::TableSetupColumn("");
        ImGui::TableHeadersRow();
        for (JobId id : ids) {
            const JobInfo* j = jobs_.info(id);
            if (!j) continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(id);
            if (ImGui::Selectable(j->spec.name.c_str(), selectedJob_ == id, ImGuiSelectableFlags_SpanAllColumns))
                selectedJob_ = id;
            ImGui::TableNextColumn();
            if (j->status == JobStatus::Running)
                ImGui::TextColored(ImVec4(0.6f, 0.9f, 1, 1), j->stopRequested ? "stopping" : "running");
            else if (j->status == JobStatus::Killed)
                ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "killed");
            else if (j->exitCode == 0)
                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "ok");
            else
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "exit %d", j->exitCode);
            ImGui::TableNextColumn();
            double end = j->status == JobStatus::Running ? Jobs::now() : j->endTime;
            ImGui::Text("%s", fmtSecs(end - j->startTime).c_str());
            ImGui::TableNextColumn();
            if (j->status == JobStatus::Running) {
                if (ImGui::SmallButton(j->spec.kind == JobKind::Train && !j->stopRequested ? "stop" : "kill"))
                    jobs_.stop(id, (double)stopTimeout_);
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    const JobInfo* j = jobs_.info(selectedJob_);
    if (!j) {
        ImGui::TextDisabled("select a job to see its output");
        return;
    }
    ImGui::Checkbox("Auto-scroll", &autoScroll_);
    ImGui::SameLine();
    if (ImGui::SmallButton("Copy log")) ImGui::SetClipboardText(jobs_.joined(selectedJob_).c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Copy command")) ImGui::SetClipboardText(j->commandLine.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("%zu lines", jobs_.lineCount(selectedJob_));

    // Incremental fetch into the local view buffer.
    if (logViewJob_ != selectedJob_) {
        logViewJob_ = selectedJob_;
        logView_.clear();
        logViewFrom_ = jobs_.firstLine(selectedJob_);
    }
    size_t total = jobs_.lineCount(selectedJob_);
    size_t first = jobs_.firstLine(selectedJob_);
    if (first > logViewFrom_ + logView_.size()) {  // ring buffer wrapped past us
        logView_.clear();
        logViewFrom_ = first;
    }
    size_t have = logViewFrom_ + logView_.size();
    if (total > have) jobs_.lines(selectedJob_, have, logView_);

    ImGui::BeginChild("##log", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
    ImGuiListClipper clipper;
    clipper.Begin((int)logView_.size());
    while (clipper.Step())
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const std::string& l = logView_[i];
            bool bad = l.find("Traceback") != std::string::npos || l.find("Error") != std::string::npos ||
                       l.find("FAIL") != std::string::npos || l.find("DIVERGED") != std::string::npos;
            bool good = l.find(" WIN") != std::string::npos || l.find("WROTE SOLUTION") != std::string::npos ||
                        l.find("installed ->") != std::string::npos || l.find("MATCHED") != std::string::npos ||
                        l.find("OK") != std::string::npos;
            if (bad) ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%s", l.c_str());
            else if (good) ImGui::TextColored(ImVec4(0.6f, 1, 0.6f, 1), "%s", l.c_str());
            else ImGui::TextUnformatted(l.c_str());
        }
    if (ImGui::GetIO().MouseWheel > 0 && ImGui::IsWindowHovered()) autoScroll_ = false;
    if (autoScroll_) ImGui::SetScrollY(ImGui::GetScrollMaxY());
    ImGui::EndChild();
}

// -------------------------------------------------------------------- modals

void Gui::drawModals(App& app) {
    if (modalOverwrite_) { ImGui::OpenPopup("Overwrite run?"); modalOverwrite_ = false; }
    if (ImGui::BeginPopupModal("Overwrite run?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("runs/%s already exists.", form_.runName);
        ImGui::Text("Starting overwrites its checkpoints and truncates log.csv.");
        if (ImGui::Button("Overwrite")) { ImGui::CloseCurrentPopup(); startTraining(app, true); }
        ImGui::SameLine();
        if (ImGui::Button("Use a new name")) {
            std::string u = uniqueRunName(trim(form_.runName));
            std::snprintf(form_.runName, sizeof form_.runName, "%s", u.c_str());
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    if (!modalDeployRun_.empty() && !ImGui::IsPopupOpen("Deploy to Geometry Dash")) ImGui::OpenPopup("Deploy to Geometry Dash");
    if (ImGui::BeginPopupModal("Deploy to Geometry Dash", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        const RunInfo* r = nullptr;
        for (const auto& x : runs_) if (x.name == modalDeployRun_) r = &x;
        if (!r) { modalDeployRun_.clear(); ImGui::CloseCurrentPopup(); }
        else {
            Solution s;
            std::string err;
            bool ok = readSolution(r->dir / "solution.json", s, err);
            ImGui::Text("run: %s", r->name.c_str());
            if (ok)
                ImGui::Text("solution: level %d, %d ticks, %d hold ranges, end_x %.0f, det=%s", s.levelId, s.nTicks,
                            s.rangeCount, s.endX, s.deterministic ? "yes" : "no");
            else
                ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "cannot read solution.json: %s", err.c_str());
            ImGui::Text("installs -> %s", (paths::modSolutionsDir() / (std::to_string(ok ? s.levelId : 1) + ".json")).generic_string().c_str());
            ImGui::TextDisabled("deploy.py first re-verifies the raw holds win via env_check run");
            ImGui::Checkbox("--no-verify (curriculum / partial solutions)", &modalDeployNoVerify_);
            ImGui::Separator();
            ImGui::TextWrapped("Then in GD: normal mode, no start pos, click-between-frames off. F8 toggles the "
                               "replayer. Results land in the mod's traces/.");
            ImGui::BeginDisabled(!ok);
            if (ImGui::Button("Deploy")) { startDeploy(*r, app, modalDeployNoVerify_); modalDeployRun_.clear(); ImGui::CloseCurrentPopup(); }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) { modalDeployRun_.clear(); ImGui::CloseCurrentPopup(); }
        }
        ImGui::EndPopup();
    }

    if (!modalDeleteRun_.empty() && !ImGui::IsPopupOpen("Delete run?")) ImGui::OpenPopup("Delete run?");
    if (ImGui::BeginPopupModal("Delete run?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Permanently delete runs/%s (checkpoints, traces, solution)?", modalDeleteRun_.c_str());
        if (ImGui::Button("Delete")) {
            std::error_code ec;
            fs::remove_all(paths::runsDir() / modalDeleteRun_, ec);
            toast(ec ? "delete failed: " + ec.message() : "deleted " + modalDeleteRun_);
            if (selectedRun_ == modalDeleteRun_) selectedRun_.clear();
            modalDeleteRun_.clear();
            lastRunsScan_ = -1.0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) { modalDeleteRun_.clear(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }

    if (modalRebuildViewer_) { ImGui::OpenPopup("Rebuild viewer?"); modalRebuildViewer_ = false; }
    if (ImGui::BeginPopupModal("Rebuild viewer?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("The running viewer.exe is renamed to viewer.exe.old so the linker can write a new "
                           "one; the new build is used on the next launch.");
        if (ImGui::Button("Rebuild")) {
            std::error_code ec;
            fs::path exe = paths::viewerExe(), old = paths::buildDir() / "viewer.exe.old";
            fs::remove(old, ec);
            fs::rename(exe, old, ec);
            if (ec) toast("rename failed: " + ec.message());
            JobSpec s;
            s.name = "cmake --build (all)";
            s.exe = "cmake";
            s.args = {"--build", paths::buildDir().generic_string()};
            s.cwd = paths::repoRoot();
            launch(s);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void Gui::drawToasts(App& app) {
    double t = GetTime();
    toasts_.erase(std::remove_if(toasts_.begin(), toasts_.end(), [t](const Toast& x) { return x.expiry < t; }),
                  toasts_.end());
    if (toasts_.empty()) return;
    ImGui::SetNextWindowPos(ImVec2((float)app.viewW - 12, 12), ImGuiCond_Always, ImVec2(1, 0));
    ImGui::SetNextWindowBgAlpha(0.85f);
    ImGui::Begin("##toasts", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
    for (const auto& x : toasts_) {
        ImGui::PushTextWrapPos(420);
        ImGui::TextUnformatted(x.text.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::End();
}

}  // namespace gdv
