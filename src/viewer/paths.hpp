// Repo-relative path discovery for the viewer GUI: repo root from the exe
// location (build/viewer.exe -> parent of build/), the Python venv, the tool
// exes, the GD install and the Geode replay mod's save dir. No raylib here —
// paths.cpp includes windows.h, which clashes with raylib's names.
#pragma once

#include <filesystem>
#include <string>

namespace gdv::paths {

// Discover the repo root and set defaults. Call once before anything else.
void init();

const std::filesystem::path& repoRoot();
std::filesystem::path runsDir();
std::filesystem::path buildDir();
std::filesystem::path pythonCwd();  // <repo>/python (gdrl is not installed)
std::filesystem::path envCheckExe();
std::filesystem::path traceCheckExe();
std::filesystem::path dumpExe();
std::filesystem::path viewerExe();

// Geode mod save dir: %LOCALAPPDATA%/GeometryDash/geode/mods/fang.rl-replay
std::filesystem::path modDir();
std::filesystem::path modSolutionsDir();
std::filesystem::path modTracesDir();

// Overridable (prefs): python interpreter and the GD Resources dir.
std::filesystem::path pythonExe();
void setPythonExe(const std::filesystem::path& p);
std::filesystem::path resourcesDir();
void setResourcesDir(const std::filesystem::path& p);

std::filesystem::path levelPath(int id);           // <res>/levels/<id>.txt
std::filesystem::path songPath(int id);            // <res>/<song>.mp3 ("" if unknown id)
int levelIdOf(const std::filesystem::path& file);  // numeric stem else -1

// Explorer / default handler.
void openInExplorer(const std::filesystem::path& p);

// Windows-CRT quoting for one argv element.
std::string quoteArg(const std::string& a);

}  // namespace gdv::paths
