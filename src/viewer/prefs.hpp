// Tiny key=value prefs file (<repo>/viewer_prefs.ini, gitignored) for the
// GUI: resources dir, python exe, panel geometry, last level, form values.
#pragma once

#include <filesystem>
#include <map>
#include <string>

namespace gdv {

class Prefs {
public:
    void load(const std::filesystem::path& file);
    void save() const;

    std::string get(const std::string& key, const std::string& def = "") const;
    int getInt(const std::string& key, int def) const;
    float getFloat(const std::string& key, float def) const;
    bool getBool(const std::string& key, bool def) const;
    void set(const std::string& key, const std::string& value);
    void setInt(const std::string& key, int v);
    void setFloat(const std::string& key, float v);
    void setBool(const std::string& key, bool v);

private:
    std::filesystem::path file_;
    std::map<std::string, std::string> kv_;
};

}  // namespace gdv
