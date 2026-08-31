#include "viewer/jobs.hpp"

#include "viewer/paths.hpp"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>

namespace gdv {

struct Jobs::Job {
    JobInfo info;
    HANDLE process = nullptr;
    HANDLE jobObject = nullptr;
    HANDLE pipeRead = nullptr;
    std::thread reader;
    mutable std::mutex mu;
    std::deque<std::string> lines;
    size_t base = 0;  // line index of lines.front()
    std::string partial;
    bool reported = false;

    void push(std::string line) {
        std::lock_guard<std::mutex> lk(mu);
        lines.push_back(std::move(line));
        while (lines.size() > Jobs::kMaxLines) {
            lines.pop_front();
            ++base;
        }
    }
};

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

std::wstring upper(std::wstring s) {
    for (auto& c : s) c = (wchar_t)towupper(c);
    return s;
}

// Current environment + overrides, as a CreateProcessW UNICODE block.
std::wstring buildEnvBlock(const std::vector<std::pair<std::string, std::string>>& overrides) {
    std::map<std::wstring, std::pair<std::wstring, std::wstring>> env;  // UPPER -> (name, value)
    wchar_t* block = GetEnvironmentStringsW();
    if (block) {
        for (wchar_t* p = block; *p;) {
            std::wstring entry(p);
            p += entry.size() + 1;
            size_t eq = entry.find(L'=', 1);  // skip the "=C:" drive entries' leading '='
            if (eq == std::wstring::npos) continue;
            std::wstring name = entry.substr(0, eq);
            env[upper(name)] = {name, entry.substr(eq + 1)};
        }
        FreeEnvironmentStringsW(block);
    }
    for (const auto& [k, v] : overrides) {
        std::wstring name = widen(k);
        env[upper(name)] = {name, widen(v)};
    }
    std::wstring out;
    for (const auto& [_, nv] : env) {
        out += nv.first;
        out += L'=';
        out += nv.second;
        out += L'\0';
    }
    out += L'\0';
    return out;
}

}  // namespace

double Jobs::now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

Jobs::Jobs() = default;

Jobs::~Jobs() {
    for (auto& j : jobs_) {
        if (j->info.status == JobStatus::Running && j->jobObject)
            TerminateJobObject(j->jobObject, 1);
        if (j->reader.joinable()) j->reader.join();
        if (j->process) CloseHandle(j->process);
        if (j->jobObject) CloseHandle(j->jobObject);
    }
}

Jobs::Job* Jobs::find(JobId id) {
    for (auto& j : jobs_)
        if (j->info.id == id) return j.get();
    return nullptr;
}
const Jobs::Job* Jobs::find(JobId id) const {
    for (const auto& j : jobs_)
        if (j->info.id == id) return j.get();
    return nullptr;
}

JobId Jobs::start(const JobSpec& spec, std::string& err) {
    auto job = std::make_unique<Job>();
    job->info.spec = spec;
    job->info.id = next_;

    // Command line: quoted exe + quoted args.
    std::string cmd = paths::quoteArg(spec.exe.string());
    for (const auto& a : spec.args) {
        cmd += ' ';
        cmd += paths::quoteArg(a);
    }
    job->info.commandLine = cmd;

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof sa;
    sa.bInheritHandle = TRUE;
    HANDLE readEnd = nullptr, writeEnd = nullptr;
    if (!CreatePipe(&readEnd, &writeEnd, &sa, 1 << 16)) {
        err = "CreatePipe failed";
        return 0;
    }
    SetHandleInformation(readEnd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = writeEnd;
    si.hStdError = writeEnd;
    PROCESS_INFORMATION pi{};

    auto env = spec.env;
    if (spec.python) {
        env.emplace_back("PYTHONUNBUFFERED", "1");
        env.emplace_back("PYTHONUTF8", "1");
        env.emplace_back("PYTHONIOENCODING", "utf-8");
    }
    std::wstring envBlock = buildEnvBlock(env);
    std::wstring wcmd = widen(cmd);
    std::wstring wcwd = spec.cwd.empty() ? std::wstring() : spec.cwd.wstring();

    HANDLE jobObject = CreateJobObjectW(nullptr, nullptr);
    if (jobObject) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
        lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(jobObject, JobObjectExtendedLimitInformation, &lim, sizeof lim);
    }

    BOOL ok = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
                             envBlock.data(), wcwd.empty() ? nullptr : wcwd.c_str(), &si, &pi);
    CloseHandle(writeEnd);  // our copy; the child holds the other reference
    if (!ok) {
        DWORD e = GetLastError();
        char buf[256];
        std::snprintf(buf, sizeof buf, "CreateProcess failed (error %lu): %s",
                      (unsigned long)e, spec.exe.string().c_str());
        err = buf;
        CloseHandle(readEnd);
        if (jobObject) CloseHandle(jobObject);
        return 0;
    }
    if (jobObject) AssignProcessToJobObject(jobObject, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);

    job->process = pi.hProcess;
    job->jobObject = jobObject;
    job->pipeRead = readEnd;
    job->info.startTime = now();
    Job* raw = job.get();
    job->reader = std::thread([raw] {
        char buf[4096];
        DWORD got = 0;
        bool lastCR = false;
        while (ReadFile(raw->pipeRead, buf, sizeof buf, &got, nullptr) && got > 0) {
            for (DWORD i = 0; i < got; ++i) {
                char c = buf[i];
                if (c == '\n') {
                    // "\r\n": the CR already flushed the line
                    if (!lastCR) {
                        raw->push(raw->partial);
                        raw->partial.clear();
                    }
                } else if (c == '\r') {
                    // CR flushes too, so tqdm-style "\r" progress shows up as lines
                    raw->push(raw->partial);
                    raw->partial.clear();
                } else {
                    raw->partial += c;
                }
                lastCR = c == '\r';
            }
        }
        if (!raw->partial.empty()) {
            raw->push(raw->partial);
            raw->partial.clear();
        }
        CloseHandle(raw->pipeRead);
        raw->pipeRead = nullptr;
    });
    jobs_.push_back(std::move(job));
    return next_++;
}

void Jobs::stop(JobId id, double timeoutSec) {
    Job* j = find(id);
    if (!j || j->info.status != JobStatus::Running) return;
    if (j->info.spec.kind == JobKind::Train && !j->info.spec.runDir.empty()) {
        if (!j->info.stopRequested) {
            std::error_code ec;
            std::filesystem::create_directories(j->info.spec.runDir, ec);
            FILE* f = std::fopen((j->info.spec.runDir / "STOP").string().c_str(), "wb");
            if (f) std::fclose(f);
            j->info.stopRequested = true;
            j->info.stopRequestTime = now();
            j->info.stopTimeoutSec = timeoutSec;
            j->push("[viewer] STOP requested; the trainer saves ckpt_latest at the end of "
                    "this update (kill after " + std::to_string((int)timeoutSec) + "s)");
        } else {
            kill(id);
        }
        return;
    }
    kill(id);
}

void Jobs::kill(JobId id) {
    Job* j = find(id);
    if (!j || j->info.status != JobStatus::Running) return;
    if (j->jobObject) TerminateJobObject(j->jobObject, 1);
    else if (j->process) TerminateProcess(j->process, 1);
    j->push("[viewer] killed");
    j->info.status = JobStatus::Killed;
}

void Jobs::poll() {
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.info.status == JobStatus::Running) {
            if (WaitForSingleObject(j.process, 0) == WAIT_OBJECT_0) {
                DWORD code = 0;
                GetExitCodeProcess(j.process, &code);
                j.info.exitCode = (int)code;
                j.info.status = JobStatus::Exited;
                j.info.endTime = now();
            } else if (j.info.stopRequested &&
                       now() - j.info.stopRequestTime > j.info.stopTimeoutSec) {
                kill(j.info.id);  // trainer ignored the STOP file
            }
        }
        if (j.info.status != JobStatus::Running && !j.reported) {
            if (j.reader.joinable()) j.reader.join();
            if (j.info.status == JobStatus::Killed) j.info.endTime = now();
            j.reported = true;
            if (onExit) onExit(j.info);
        }
    }
}

bool Jobs::running(JobId id) const {
    const Job* j = find(id);
    return j && j->info.status == JobStatus::Running;
}

const JobInfo* Jobs::info(JobId id) const {
    const Job* j = find(id);
    return j ? &j->info : nullptr;
}

std::vector<JobId> Jobs::ids() const {
    std::vector<JobId> out;
    for (const auto& j : jobs_) out.push_back(j->info.id);
    return out;
}

bool Jobs::anyRunning(JobKind kind) const {
    for (const auto& j : jobs_)
        if (j->info.status == JobStatus::Running && j->info.spec.kind == kind) return true;
    return false;
}

bool Jobs::anyPythonRunning() const {
    for (const auto& j : jobs_)
        if (j->info.status == JobStatus::Running && j->info.spec.python) return true;
    return false;
}

bool Jobs::runInUse(const std::string& runName) const {
    for (const auto& j : jobs_)
        if (j->info.status == JobStatus::Running && j->info.spec.runName == runName) return true;
    return false;
}

void Jobs::clearFinished() {
    for (auto it = jobs_.begin(); it != jobs_.end();) {
        if ((*it)->info.status != JobStatus::Running && (*it)->reported) {
            if ((*it)->process) CloseHandle((*it)->process);
            if ((*it)->jobObject) CloseHandle((*it)->jobObject);
            it = jobs_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t Jobs::lineCount(JobId id) const {
    const Job* j = find(id);
    if (!j) return 0;
    std::lock_guard<std::mutex> lk(j->mu);
    return j->base + j->lines.size();
}

size_t Jobs::firstLine(JobId id) const {
    const Job* j = find(id);
    if (!j) return 0;
    std::lock_guard<std::mutex> lk(j->mu);
    return j->base;
}

void Jobs::lines(JobId id, size_t from, std::vector<std::string>& out) const {
    const Job* j = find(id);
    if (!j) return;
    std::lock_guard<std::mutex> lk(j->mu);
    size_t start = from > j->base ? from - j->base : 0;
    for (size_t i = start; i < j->lines.size(); ++i) out.push_back(j->lines[i]);
}

std::string Jobs::joined(JobId id) const {
    const Job* j = find(id);
    if (!j) return {};
    std::lock_guard<std::mutex> lk(j->mu);
    std::string out;
    for (const auto& l : j->lines) {
        out += l;
        out += '\n';
    }
    return out;
}

}  // namespace gdv
