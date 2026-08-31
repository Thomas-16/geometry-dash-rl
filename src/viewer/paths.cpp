#include "viewer/paths.hpp"

#include "viewer/levels.hpp"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>

#include <cstdlib>

namespace gdv::paths {

namespace {

namespace fs = std::filesystem;

fs::path g_root;
fs::path g_python;
fs::path g_res;

bool looksLikeRepo(const fs::path& p) {
    std::error_code ec;
    return fs::exists(p / "CMakeLists.txt", ec) &&
           fs::exists(p / "python" / "gdrl" / "train.py", ec);
}

fs::path exePath() {
    wchar_t buf[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof(buf) / sizeof(buf[0])));
    return n ? fs::path(std::wstring(buf, n)) : fs::path();
}

// Walk up from `start` until a repo root is found.
bool findUp(fs::path start, fs::path& out) {
    std::error_code ec;
    start = fs::absolute(start, ec);
    for (int i = 0; i < 8 && !start.empty(); ++i) {
        if (looksLikeRepo(start)) { out = start; return true; }
        fs::path parent = start.parent_path();
        if (parent == start) break;
        start = parent;
    }
    return false;
}

}  // namespace

void init() {
    std::error_code ec;
    fs::path exe = exePath();
    fs::path found;
    if (!exe.empty() && findUp(exe.parent_path(), found)) g_root = found;
    else if (findUp(fs::current_path(ec), found)) g_root = found;
    else g_root = fs::current_path(ec);
    g_python = g_root / ".venv" / "Scripts" / "python.exe";
    g_res = "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources";
}

const fs::path& repoRoot() { return g_root; }
fs::path runsDir() { return g_root / "runs"; }
fs::path buildDir() { return g_root / "build"; }
fs::path pythonCwd() { return g_root / "python"; }
fs::path envCheckExe() { return buildDir() / "env_check.exe"; }
fs::path traceCheckExe() { return buildDir() / "trace_check.exe"; }
fs::path dumpExe() { return buildDir() / "dump.exe"; }
fs::path viewerExe() { return buildDir() / "viewer.exe"; }

fs::path modDir() {
    const char* base = std::getenv("LOCALAPPDATA");
    if (!base) return {};
    return fs::path(base) / "GeometryDash" / "geode" / "mods" / "fang.rl-replay";
}
fs::path modSolutionsDir() { return modDir().empty() ? fs::path() : modDir() / "solutions"; }
fs::path modTracesDir() { return modDir().empty() ? fs::path() : modDir() / "traces"; }

fs::path pythonExe() { return g_python; }
void setPythonExe(const fs::path& p) { if (!p.empty()) g_python = p; }
fs::path resourcesDir() { return g_res; }
void setResourcesDir(const fs::path& p) { if (!p.empty()) g_res = p; }

fs::path levelPath(int id) { return g_res / "levels" / (std::to_string(id) + ".txt"); }

fs::path songPath(int id) {
    const LevelInfo* l = levelInfo(id);
    if (!l) return {};
    return g_res / (std::string(l->song) + ".mp3");
}

int levelIdOf(const fs::path& file) {
    std::string stem = file.stem().string();
    if (stem.empty()) return -1;
    for (char c : stem)
        if (c < '0' || c > '9') return -1;
    return std::atoi(stem.c_str());
}

void openInExplorer(const fs::path& p) {
    if (p.empty()) return;
    ShellExecuteW(nullptr, L"open", p.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

std::string quoteArg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string out = "\"";
    size_t bs = 0;
    for (char c : a) {
        if (c == '\\') { ++bs; continue; }
        if (c == '"') {
            out.append(bs * 2 + 1, '\\');
            out += '"';
            bs = 0;
            continue;
        }
        out.append(bs, '\\');
        bs = 0;
        out += c;
    }
    out.append(bs * 2, '\\');
    out += '"';
    return out;
}

}  // namespace gdv::paths
