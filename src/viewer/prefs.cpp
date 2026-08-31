#include "viewer/prefs.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace gdv {

void Prefs::load(const std::filesystem::path& file) {
    file_ = file;
    kv_.clear();
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        kv_[line.substr(0, eq)] = line.substr(eq + 1);
    }
}

void Prefs::save() const {
    if (file_.empty()) return;
    std::ofstream out(file_, std::ios::trunc);
    out << "# gd-rl viewer prefs (auto-written)\n";
    for (const auto& [k, v] : kv_) out << k << '=' << v << '\n';
}

std::string Prefs::get(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second;
}
int Prefs::getInt(const std::string& key, int def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : std::atoi(it->second.c_str());
}
float Prefs::getFloat(const std::string& key, float def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : std::strtof(it->second.c_str(), nullptr);
}
bool Prefs::getBool(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    return it->second == "1" || it->second == "true";
}
void Prefs::set(const std::string& key, const std::string& value) { kv_[key] = value; }
void Prefs::setInt(const std::string& key, int v) { kv_[key] = std::to_string(v); }
void Prefs::setFloat(const std::string& key, float v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%g", v);
    kv_[key] = buf;
}
void Prefs::setBool(const std::string& key, bool v) { kv_[key] = v ? "1" : "0"; }

}  // namespace gdv
