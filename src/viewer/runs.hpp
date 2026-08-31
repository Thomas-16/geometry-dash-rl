// runs/ discovery and the on-disk formats the GUI reads/writes:
//   .gdtrace      (python/gdrl/traces.py)  per-episode bitpacked holds
//   solution.json (python/gdrl/eval.py)    [start,end) hold ranges, the mod's input
//   log.csv / config.json (python/gdrl/train.py)
//   result_NNNN.json (mod_replay)          real-game attempt outcome
// No raylib here.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gdv {

struct TraceEpisode {
    std::vector<uint8_t> holds;  // one per physics tick
    float finalX = 0.0f;
    bool won = false;
};
bool readGdTrace(const std::filesystem::path& p, std::vector<TraceEpisode>& out, std::string& err);
int traceEpisodeCount(const std::filesystem::path& p);  // header only; -1 on failure

struct Solution {
    int tps = 240;
    int levelId = 1;         // legacy files omit it -> 1 (deploy.py does the same)
    float endX = -1.0f;
    int nTicks = 0;
    bool deterministic = false;
    std::vector<uint8_t> holds;  // expanded per tick
    int rangeCount = 0;
};
bool readSolution(const std::filesystem::path& p, Solution& out, std::string& err);
bool writeSolution(const std::filesystem::path& p, const Solution& s, std::string& err);

struct FileEntry {
    std::string label;
    std::string path;
};
std::vector<FileEntry> scanReplayFiles();    // runs/*/{wins,eval}.gdtrace, runs/*/traces/*.gdtrace
std::vector<FileEntry> scanLiveFiles();      // runs/*/live.bin
std::vector<FileEntry> scanSolutionFiles();  // runs/*/solution.json + mod solutions/*.json
std::string newestLiveFile();
std::string ageLabel(const std::filesystem::path& p);

// One log.csv row (columns named in train.py LOG_COLUMNS; unknown -> 0).
struct LogRow {
    int update = 0;
    double globalTicks = 0;
    float meanFinalX = 0, bestFinalX = 0, bestXEver = 0, winRate = 0;
    int episodes = 0;
    float sps = 0;
    float entropy = 0;
    float detX = -1;
    int detWin = 0;
};
bool readLogCsv(const std::filesystem::path& p, std::vector<LogRow>& out);

struct RunInfo {
    std::string name;
    std::filesystem::path dir;
    std::filesystem::file_time_type mtime{};
    // config.json
    bool hasConfig = false;
    int levelId = 1;
    std::string levelPath;  // "" = default (Stereo Madness)
    int nEnvs = 0, updates = 0;
    std::string arch, amp;
    float endX = -1.0f;
    // files
    bool hasBestDet = false, hasBest = false, hasLatest = false;
    bool hasSolution = false, hasWins = false, hasEval = false, hasLive = false;
    // log.csv
    bool hasLog = false;
    LogRow last;
    float bestDetX = -1.0f;
    int detWins = 0;
    // live.bin seq (0 = none)
    uint32_t liveSeq = 0;
};
std::vector<RunInfo> scanRuns();
uint32_t liveSeq(const std::filesystem::path& liveBin);

struct ModResult {
    int levelId = -1;
    std::string outcome;  // "win" / "death"
    long long tick = 0;
    float x = 0, pct = 0;
    std::string solutionPath;
    int nTicks = 0;
};
struct ModTrace {
    int number = 0;
    std::filesystem::path csv;
    std::filesystem::path resultJson;
    bool hasResult = false;
    ModResult result;
};
std::vector<ModTrace> scanModTraces();  // newest first

// "run", "run-2", "run-3", ... that doesn't exist under runs/.
std::string uniqueRunName(const std::string& base);

}  // namespace gdv
