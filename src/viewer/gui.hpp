// The right-side control panel (Dear ImGui via rlImGui). Owns the GUI-only
// state: tabs (View / Train / Runs / Tools / Jobs), the training form, the
// runs table cache, modals, toasts, and the train->eval pipeline. Talks to
// the modes only through Mode::drawPanel and App::commands.
#pragma once

#include "viewer/jobs.hpp"
#include "viewer/prefs.hpp"
#include "viewer/runs.hpp"
#include "viewer/world.hpp"

#include <map>
#include <string>
#include <vector>

namespace gdv {

class Mode;

class Gui {
public:
    Gui(Prefs& prefs, Jobs& jobs);

    void init(App& app);       // after InitWindow: rlImGui setup, prefs -> app
    void shutdown();
    // Top of frame: viewport size + input-capture flags into app.
    void beginFrame(App& app);
    // Inside rlImGuiBegin/End: the whole panel + overlays.
    void draw(App& app, Mode* const* modes, int modeCount, int cur);
    void toast(const std::string& msg, float secs = 6.0f);
    bool panelVisible() const { return panelVisible_; }
    void setPanelVisible(bool v) { panelVisible_ = v; }
    void selectTab(int i) { tab_ = i; wantTab_ = true; }
    void savePrefs(const App& app);
    // Jobs::onExit hook (pipeline + toasts).
    void jobExited(const JobInfo& info);
    // Level list (ids present under <res>/levels), refreshed by rescanLevels().
    void rescanLevels();

private:
    // Trainer CLI mirror. Defaults MUST match python/gdrl/ppo.py PPOConfig and
    // train.py parse_args (the viewer only mirrors them; only non-default
    // flags are emitted so the command line stays readable).
    struct TrainForm {
        char runName[64] = "run";
        int nEnvs = 1024;
        int updates = 1000;
        float endX = -1.0f;
        float respawn = 0.5f;
        float lr = 1e-3f;
        float entCoef = 0.005f;
        int arch = 0;         // 0 cnn, 1 mlp
        int detEvery = 10;
        float silCoef = 1.0f;
        int amp = 0;          // 0 bf16, 1 none
        bool compile = false;
        char resume[512] = "";
        // contracts (train.py --obs-version / --action-mode / --teacher-coef)
        int obsVersion = 1;   // 0 -> 5 (legacy), 1 -> 6 (default)
        int actionMode = 1;   // 0 macro, 1 precision (default)
        float teacherCoef = 0.25f;
        // advanced
        int ticksPerAction = 4;
        int rollout = 64;
        float gamma = 0.9995f;
        float rewardScale = 0.1f;
        int traceEvery = 20;
        int ckptEvery = 50;
        int polishWins = 25;
        int polishAt = -1;
        bool noPolish = false;
        int silEpisodes = 8;
        bool noGraph = false;
        char device[32] = "";
        char extra[256] = "";  // free-form extra flags
    };
    struct TrainStatus {
        int update = -1;
        float meanX = 0, maxX = 0, bestX = 0, bestPct = 0, winRate = 0, sps = 0, decS = 0,
              entropy = 0;
        float detX = -1;
        bool detWin = false;
        // teacher-loss diagnostics (precision / planner demos): mean
        // probability of the demo action, disagreement rate; -1 = none
        float teacherP = -1, teacherDis = -1;
        std::string phase;  // explore / polish / collapse / stopped / done
    };
    struct Pipeline {
        std::string runName;
        std::filesystem::path runDir;
        std::string levelArg;
        size_t cursor = 0;
        bool sawWin = false;
        bool evalLaunched = false;
        TrainStatus status;
    };
    struct LogCache {
        std::filesystem::path path;
        std::filesystem::file_time_type mtime{};
        uintmax_t size = 0;
        double lastCheck = -1.0;
        std::vector<LogRow> rows;
        std::vector<float> mean, best, det, win;
    };
    struct Toast {
        std::string text;
        double expiry;
    };

    // tabs
    void drawHeader(App& app, Mode* const* modes, int modeCount, int cur);
    void drawViewTab(App& app, Mode* const* modes, int cur);
    void drawTrainTab(App& app);
    void drawRunsTab(App& app);
    void drawToolsTab(App& app);
    void drawJobsTab(App& app);
    void drawModals(App& app);
    void drawToasts(App& app);
    void drawKeys();

    // actions
    JobId launch(JobSpec spec);
    JobSpec pythonSpec(const std::string& name, JobKind kind, const std::string& module,
                       std::vector<std::string> args, const std::string& runName = "");
    std::string levelArgFor(const App& app) const;         // current level file
    std::string levelArgFor(const RunInfo& r, const App& app) const;
    void startTraining(App& app, bool overwrite);
    void startEval(const RunInfo& r, const App& app);
    void startDeploy(const RunInfo& r, const App& app, bool noVerify);
    void startCheckTrace(const RunInfo& r, const App& app);
    void prefillResume(const RunInfo& r);
    std::vector<std::string> buildTrainArgs(const App& app) const;
    void pollPipelines();
    void refreshLog(LogCache& c, const std::filesystem::path& p);
    void rescanRuns(bool force);
    const RunInfo* selectedRun() const;
    bool isLive(const std::string& run) const;

    Prefs& prefs_;
    Jobs& jobs_;
    bool panelVisible_ = true;
    int panelWidth_ = 440;
    int tab_ = 0;
    bool wantTab_ = false;
    std::vector<int> levelIds_;  // present in <res>/levels
    char resDirBuf_[512] = "";
    char pythonBuf_[512] = "";
    int stopTimeout_ = 120;

    // train
    TrainForm form_;
    JobId trainJob_ = 0;
    std::map<JobId, Pipeline> pipelines_;
    LogCache trainLog_;

    // runs
    std::vector<RunInfo> runs_;
    double lastRunsScan_ = -1.0;
    std::string selectedRun_;
    LogCache runLog_;
    std::map<std::string, std::pair<uint32_t, double>> liveSeen_;  // run -> (seq, when it changed)
    std::vector<ModTrace> modTraces_;
    double lastModScan_ = -1.0;

    // tools
    char benchCompare_[512] = "";
    char benchSave_[512] = "";

    // jobs
    JobId selectedJob_ = 0;
    bool autoScroll_ = true;
    std::vector<std::string> logView_;
    size_t logViewFrom_ = 0;
    JobId logViewJob_ = 0;

    // modals
    std::string modalDeleteRun_;
    std::string modalDeployRun_;
    bool modalDeployNoVerify_ = false;
    bool modalOverwrite_ = false;
    bool modalRebuildViewer_ = false;

    std::vector<Toast> toasts_;
};

}  // namespace gdv
