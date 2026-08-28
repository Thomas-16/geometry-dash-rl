#include "core/level.hpp"

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gd {

const std::string* Object::prop(int key) const {
    for (const auto& [k, v] : props)
        if (k == key) return &v;
    return nullptr;
}

namespace {

std::vector<unsigned char> base64UrlDecode(const std::string& in) {
    static int table[256];
    static bool init = false;
    if (!init) {
        for (int& t : table) t = -1;
        const char* std64 =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (int i = 0; i < 62; ++i) table[(unsigned char)std64[i]] = i;
        table[(unsigned char)'+'] = 62;
        table[(unsigned char)'-'] = 62; // URL-safe variant
        table[(unsigned char)'/'] = 63;
        table[(unsigned char)'_'] = 63; // URL-safe variant
        init = true;
    }

    std::vector<unsigned char> out;
    out.reserve(in.size() * 3 / 4);
    int buf = 0, bits = 0;
    for (unsigned char c : in) {
        if (c == '=' || c == '\r' || c == '\n' || c == ' ' || c == '\0')
            continue;
        int v = table[c];
        if (v < 0)
            throw std::runtime_error("invalid base64 character in level data");
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((unsigned char)((buf >> bits) & 0xFF));
        }
    }
    return out;
}

std::string gunzip(const std::vector<unsigned char>& in) {
    z_stream strm{};
    // 15 + 16: max window, expect a gzip (not zlib) header.
    if (inflateInit2(&strm, 15 + 16) != Z_OK)
        throw std::runtime_error("inflateInit2 failed");

    std::string out;
    strm.next_in = const_cast<Bytef*>(in.data());
    strm.avail_in = (uInt)in.size();
    char chunk[65536];
    int ret;
    do {
        strm.next_out = (Bytef*)chunk;
        strm.avail_out = sizeof(chunk);
        ret = inflate(&strm, Z_NO_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) {
            inflateEnd(&strm);
            throw std::runtime_error("gzip inflate failed (code " +
                                     std::to_string(ret) + ")");
        }
        out.append(chunk, sizeof(chunk) - strm.avail_out);
    } while (ret != Z_STREAM_END && (strm.avail_in > 0 || strm.avail_out == 0));
    inflateEnd(&strm);
    if (ret != Z_STREAM_END)
        throw std::runtime_error("truncated gzip stream in level data");
    return out;
}

} // namespace

std::string decodeLevelString(const std::string& raw) {
    return gunzip(base64UrlDecode(raw));
}

Level Level::parse(const std::string& decoded) {
    Level level;
    size_t pos = 0;
    bool first = true;
    while (pos <= decoded.size()) {
        size_t end = decoded.find(';', pos);
        if (end == std::string::npos) end = decoded.size();
        std::string_view seg(decoded.data() + pos, end - pos);
        pos = end + 1;

        if (first) {
            level.header.assign(seg);
            first = false;
            continue;
        }
        if (seg.empty()) continue;

        Object obj;
        size_t p = 0;
        while (p < seg.size()) {
            size_t c1 = seg.find(',', p);
            if (c1 == std::string_view::npos) break;
            size_t c2 = seg.find(',', c1 + 1);
            if (c2 == std::string_view::npos) c2 = seg.size();
            int key = std::atoi(std::string(seg.substr(p, c1 - p)).c_str());
            std::string val(seg.substr(c1 + 1, c2 - c1 - 1));
            switch (key) {
                case 1: obj.id = std::atoi(val.c_str()); break;
                case 2: obj.x = std::strtof(val.c_str(), nullptr); break;
                case 3: obj.y = std::strtof(val.c_str(), nullptr); break;
                case 4: obj.flipX = (val == "1"); break;
                case 5: obj.flipY = (val == "1"); break;
                case 6: obj.rotation = std::strtof(val.c_str(), nullptr); break;
            }
            obj.props.emplace_back(key, std::move(val));
            p = c2 + 1;
        }
        if (obj.id != 0) level.objects.push_back(std::move(obj));
    }
    return level;
}

Level Level::loadFromFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open level file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse(decodeLevelString(ss.str()));
}

} // namespace gd
