#include "viewer/runs.hpp"

#include "viewer/paths.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace gdv {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

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

bool readText(const fs::path& p, std::string& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

std::string runsRelLabel(const fs::path& p) {
    std::error_code ec;
    auto rel = fs::relative(p, paths::runsDir(), ec);
    return (ec || rel.empty()) ? p.filename().string() : rel.generic_string();
}

template <class F>
void sortNewest(std::vector<fs::path>& v, F&&) {
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        std::error_code e;
        return fs::last_write_time(a, e) > fs::last_write_time(b, e);
    });
}

}  // namespace

// ------------------------------------------------------------------ .gdtrace

bool readGdTrace(const fs::path& p, std::vector<TraceEpisode>& out, std::string& err) {
    out.clear();
    FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) {
        err = "cannot open " + p.string();
        return false;
    }
    unsigned char head[16];
    if (std::fread(head, 1, 16, f) != 16 || rdU32(head) != 0x52544447u ||
        rdU32(head + 4) != 1 || rdU32(head + 8) != 240) {
        err = "bad .gdtrace header in " + p.string();
        std::fclose(f);
        return false;
    }
    uint32_t n = rdU32(head + 12);
    for (uint32_t i = 0; i < n; ++i) {
        unsigned char eh[12];
        if (std::fread(eh, 1, 12, f) != 12) break;
        TraceEpisode e;
        uint32_t nTicks = rdU32(eh);
        e.finalX = rdF32(eh + 4);
        e.won = eh[8] != 0;
        std::vector<unsigned char> bits((nTicks + 7) / 8);
        if (std::fread(bits.data(), 1, bits.size(), f) != bits.size()) break;
        e.holds.resize(nTicks);
        for (uint32_t t = 0; t < nTicks; ++t) e.holds[t] = (bits[t / 8] >> (t % 8)) & 1;
        out.push_back(std::move(e));
    }
    std::fclose(f);
    if (out.empty()) {
        err = "no episodes in " + p.string();
        return false;
    }
    return true;
}

int traceEpisodeCount(const fs::path& p) {
    FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return -1;
    unsigned char head[16];
    bool ok = std::fread(head, 1, 16, f) == 16 && rdU32(head) == 0x52544447u;
    std::fclose(f);
    return ok ? (int)rdU32(head + 12) : -1;
}

// ------------------------------------------------------------- solution.json

bool readSolution(const fs::path& p, Solution& s, std::string& err) {
    std::string text;
    if (!readText(p, text)) {
        err = "cannot open " + p.string();
        return false;
    }
    json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        err = "json parse error in " + p.string();
        return false;
    }
    s = Solution{};
    s.tps = j.value("tps", 240);
    if (s.tps != 240) {
        err = "tps " + std::to_string(s.tps) + " != 240";
        return false;
    }
    if (j.contains("level_id") && j["level_id"].is_number()) s.levelId = j["level_id"].get<int>();
    if (j.contains("level_end_x") && j["level_end_x"].is_number())
        s.endX = j["level_end_x"].get<float>();
    s.nTicks = j.value("n_ticks", 0);
    if (s.nTicks <= 0 || s.nTicks > (1 << 20)) {
        err = "bad n_ticks";
        return false;
    }
    s.deterministic = j.value("deterministic_policy", false);
    s.holds.assign((size_t)s.nTicks, 0);
    if (!j.contains("holds") || !j["holds"].is_array()) {
        err = "missing holds";
        return false;
    }
    int prevEnd = 0;
    for (const auto& r : j["holds"]) {
        if (!r.is_array() || r.size() != 2) {
            err = "hold range is not a pair";
            return false;
        }
        int a = r[0].get<int>(), b = r[1].get<int>();
        if (a < prevEnd || b <= a || b > s.nTicks) {
            err = "bad hold range [" + std::to_string(a) + "," + std::to_string(b) + ")";
            return false;
        }
        for (int t = a; t < b; ++t) s.holds[(size_t)t] = 1;
        prevEnd = b;
        ++s.rangeCount;
    }
    return true;
}

bool writeSolution(const fs::path& p, const Solution& s, std::string& err) {
    json ranges = json::array();
    int n = (int)s.holds.size();
    for (int t = 0; t < n;) {
        if (!s.holds[t]) { ++t; continue; }
        int a = t;
        while (t < n && s.holds[t]) ++t;
        ranges.push_back({a, t});
    }
    json j;
    j["tps"] = 240;
    j["level_id"] = s.levelId;
    j["level_end_x"] = s.endX;
    j["n_ticks"] = n;
    j["deterministic_policy"] = s.deterministic;
    j["holds"] = ranges;
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream out(p, std::ios::trunc);
    if (!out) {
        err = "cannot write " + p.string();
        return false;
    }
    out << j.dump() << '\n';
    return true;
}

// ------------------------------------------------------------------ scanning

std::string ageLabel(const fs::path& p) {
    std::error_code ec;
    auto t = fs::last_write_time(p, ec);
    if (ec) return "";
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                    fs::file_time_type::clock::now() - t).count();
    if (secs < 0) secs = 0;
    char buf[32];
    if (secs < 90) std::snprintf(buf, sizeof buf, "%llds ago", (long long)secs);
    else if (secs < 5400) std::snprintf(buf, sizeof buf, "%lldm ago", (long long)(secs / 60));
    else if (secs < 60 * 60 * 48) std::snprintf(buf, sizeof buf, "%lldh ago", (long long)(secs / 3600));
    else std::snprintf(buf, sizeof buf, "%lldd ago", (long long)(secs / 86400));
    return buf;
}

std::vector<FileEntry> scanReplayFiles() {
    std::vector<fs::path> found;
    std::error_code ec;
    for (auto& run : fs::directory_iterator(paths::runsDir(), ec)) {
        if (!run.is_directory()) continue;
        for (const char* name : {"wins.gdtrace", "eval.gdtrace"}) {
            fs::path p = run.path() / name;
            if (fs::exists(p, ec)) found.push_back(p);
        }
        std::error_code ec2;
        for (auto& t : fs::directory_iterator(run.path() / "traces", ec2))
            if (t.path().extension() == ".gdtrace") found.push_back(t.path());
    }
    sortNewest(found, 0);
    std::vector<FileEntry> out;
    for (const auto& p : found) {
        int eps = traceEpisodeCount(p);
        std::string label = runsRelLabel(p);
        if (eps >= 0) label += "  (" + std::to_string(eps) + " ep)";
        std::string age = ageLabel(p);
        if (!age.empty()) label += "  " + age;
        out.push_back({std::move(label), p.string()});
    }
    return out;
}

std::vector<FileEntry> scanLiveFiles() {
    std::vector<fs::path> found;
    std::error_code ec;
    for (auto& run : fs::directory_iterator(paths::runsDir(), ec)) {
        fs::path p = run.path() / "live.bin";
        if (run.is_directory() && fs::exists(p, ec)) found.push_back(p);
    }
    sortNewest(found, 0);
    std::vector<FileEntry> out;
    for (const auto& p : found) {
        std::string label = runsRelLabel(p);
        std::string age = ageLabel(p);
        if (!age.empty()) label += "  updated " + age;
        out.push_back({std::move(label), p.string()});
    }
    return out;
}

std::vector<FileEntry> scanSolutionFiles() {
    std::vector<fs::path> found;
    std::error_code ec;
    for (auto& run : fs::directory_iterator(paths::runsDir(), ec)) {
        fs::path p = run.path() / "solution.json";
        if (run.is_directory() && fs::exists(p, ec)) found.push_back(p);
    }
    sortNewest(found, 0);
    std::vector<FileEntry> out;
    for (const auto& p : found) {
        std::string label = runsRelLabel(p);
        std::string age = ageLabel(p);
        if (!age.empty()) label += "  " + age;
        out.push_back({std::move(label), p.string()});
    }
    std::vector<fs::path> mod;
    std::error_code ec2;
    for (auto& e : fs::directory_iterator(paths::modSolutionsDir(), ec2))
        if (e.path().extension() == ".json") mod.push_back(e.path());
    sortNewest(mod, 0);
    for (const auto& p : mod) {
        std::string label = "[mod] " + p.filename().string();
        std::string age = ageLabel(p);
        if (!age.empty()) label += "  " + age;
        out.push_back({std::move(label), p.string()});
    }
    return out;
}

std::string newestLiveFile() {
    auto files = scanLiveFiles();
    return files.empty() ? "" : files.front().path;
}

// ------------------------------------------------------------------- log.csv

bool readLogCsv(const fs::path& p, std::vector<LogRow>& out) {
    out.clear();
    std::ifstream in(p);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> cols;
    {
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) cols.push_back(c);
    }
    auto idx = [&](const char* name) {
        for (size_t i = 0; i < cols.size(); ++i)
            if (cols[i] == name) return (int)i;
        return -1;
    };
    int iUpd = idx("update"), iTicks = idx("global_ticks"), iMean = idx("mean_final_x"),
        iBest = idx("best_final_x"), iBestEver = idx("best_x_ever"), iWin = idx("win_rate"),
        iEps = idx("episodes"), iSps = idx("sps"), iEnt = idx("entropy"), iDetX = idx("det_x"),
        iDetWin = idx("det_win");
    std::vector<std::string> f;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        f.clear();
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) f.push_back(c);
        auto num = [&](int i) { return (i >= 0 && i < (int)f.size()) ? std::atof(f[i].c_str()) : 0.0; };
        LogRow r;
        r.update = (int)num(iUpd);
        r.globalTicks = num(iTicks);
        r.meanFinalX = (float)num(iMean);
        r.bestFinalX = (float)num(iBest);
        r.bestXEver = (float)num(iBestEver);
        r.winRate = (float)num(iWin);
        r.episodes = (int)num(iEps);
        r.sps = (float)num(iSps);
        r.entropy = (float)num(iEnt);
        r.detX = iDetX >= 0 ? (float)num(iDetX) : -1.0f;
        r.detWin = (int)num(iDetWin);
        out.push_back(r);
    }
    return true;
}

// ---------------------------------------------------------------------- runs

uint32_t liveSeq(const fs::path& liveBin) {
    FILE* f = std::fopen(liveBin.string().c_str(), "rb");
    if (!f) return 0;
    unsigned char head[16];
    size_t got = std::fread(head, 1, 16, f);
    std::fclose(f);
    if (got != 16 || rdU32(head) != 0x564C4447u) return 0;
    return rdU32(head + 8);
}

std::vector<RunInfo> scanRuns() {
    std::vector<RunInfo> out;
    std::error_code ec;
    for (auto& run : fs::directory_iterator(paths::runsDir(), ec)) {
        if (!run.is_directory()) continue;
        RunInfo r;
        r.dir = run.path();
        r.name = run.path().filename().string();
        r.mtime = fs::last_write_time(run.path(), ec);
        auto has = [&](const char* n) { return fs::exists(run.path() / n, ec); };
        r.hasBestDet = has("ckpt_best_det.pt");
        r.hasBest = has("ckpt_best.pt");
        r.hasLatest = has("ckpt_latest.pt");
        r.hasSolution = has("solution.json");
        r.hasWins = has("wins.gdtrace");
        r.hasEval = has("eval.gdtrace");
        r.hasLive = has("live.bin");
        if (r.hasLive) {
            r.liveSeq = liveSeq(run.path() / "live.bin");
            auto lt = fs::last_write_time(run.path() / "live.bin", ec);
            if (!ec && lt > r.mtime) r.mtime = lt;
        }
        if (has("log.csv")) {
            auto lt = fs::last_write_time(run.path() / "log.csv", ec);
            if (!ec && lt > r.mtime) r.mtime = lt;
        }
        std::string text;
        if (readText(run.path() / "config.json", text)) {
            json j = json::parse(text, nullptr, false);
            if (j.is_object()) {
                r.hasConfig = true;
                if (j.contains("level") && j["level"].is_string()) {
                    r.levelPath = j["level"].get<std::string>();
                    int id = paths::levelIdOf(r.levelPath);
                    r.levelId = id > 0 ? id : -1;
                }
                r.nEnvs = j.value("n_envs", 0);
                r.updates = j.value("updates", 0);
                r.arch = j.value("arch", std::string("mlp"));
                r.amp = j.value("amp", std::string(""));
                if (j.contains("end_x") && j["end_x"].is_number()) r.endX = j["end_x"].get<float>();
            }
        }
        std::vector<LogRow> rows;
        if (readLogCsv(run.path() / "log.csv", rows) && !rows.empty()) {
            r.hasLog = true;
            r.last = rows.back();
            for (const auto& row : rows) {
                if (row.detX > r.bestDetX) r.bestDetX = row.detX;
                if (row.detWin) ++r.detWins;
            }
        }
        out.push_back(std::move(r));
    }
    std::sort(out.begin(), out.end(), [](const RunInfo& a, const RunInfo& b) { return a.mtime > b.mtime; });
    return out;
}

// ---------------------------------------------------------------- mod traces

std::vector<ModTrace> scanModTraces() {
    std::vector<ModTrace> out;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(paths::modTracesDir(), ec)) {
        std::string name = e.path().filename().string();
        if (name.rfind("attempt_", 0) != 0 || e.path().extension() != ".csv") continue;
        ModTrace t;
        t.number = std::atoi(name.c_str() + 8);
        t.csv = e.path();
        char buf[64];
        std::snprintf(buf, sizeof buf, "result_%04d.json", t.number);
        t.resultJson = e.path().parent_path() / buf;
        std::string text;
        if (readText(t.resultJson, text)) {
            json j = json::parse(text, nullptr, false);
            if (j.is_object()) {
                t.hasResult = true;
                t.result.levelId = j.value("level_id", -1);
                t.result.outcome = j.value("outcome", std::string(""));
                t.result.tick = j.value("tick", 0LL);
                t.result.x = j.value("x", 0.0f);
                t.result.pct = j.value("pct", 0.0f);
                t.result.solutionPath = j.value("solution_path", std::string(""));
                t.result.nTicks = j.value("n_ticks", 0);
            }
        }
        out.push_back(std::move(t));
    }
    std::sort(out.begin(), out.end(), [](const ModTrace& a, const ModTrace& b) { return a.number > b.number; });
    return out;
}

std::string uniqueRunName(const std::string& base) {
    std::error_code ec;
    std::string b = base.empty() ? "run" : base;
    if (!fs::exists(paths::runsDir() / b, ec)) return b;
    for (int i = 2; i < 10000; ++i) {
        std::string n = b + "-" + std::to_string(i);
        if (!fs::exists(paths::runsDir() / n, ec)) return n;
    }
    return b;
}

}  // namespace gdv
