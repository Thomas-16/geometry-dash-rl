// Subprocess runner for the GUI: launches python/tool processes with merged
// stdout+stderr captured into a per-job line buffer, inside a Win32 Job
// Object (KILL_ON_JOB_CLOSE) so nothing outlives the viewer. Several jobs may
// run at once (a long trainer plus short tools). No raylib here — jobs.cpp
// includes windows.h.
#pragma once

#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gdv {

using JobId = int;

enum class JobKind { Train, Eval, Deploy, Tool };
enum class JobStatus { Running, Exited, Killed };

struct JobSpec {
    std::string name;             // shown in the Jobs list
    JobKind kind = JobKind::Tool;
    std::filesystem::path exe;
    std::vector<std::string> args;
    std::filesystem::path cwd;
    std::vector<std::pair<std::string, std::string>> env;  // overrides
    std::string runName;              // train/eval/deploy: the run this belongs to
    std::filesystem::path runDir;     // train: where the STOP file goes
    bool python = false;              // adds PYTHONUNBUFFERED/UTF8
};

struct JobInfo {
    JobId id = 0;
    JobSpec spec;
    JobStatus status = JobStatus::Running;
    int exitCode = -1;
    double startTime = 0.0;   // seconds, steady clock
    double endTime = 0.0;
    bool stopRequested = false;
    double stopRequestTime = 0.0;
    double stopTimeoutSec = 120.0;
    std::string commandLine;  // as launched
};

class Jobs {
public:
    Jobs();
    ~Jobs();
    Jobs(const Jobs&) = delete;
    Jobs& operator=(const Jobs&) = delete;

    // Returns 0 and fills err on failure.
    JobId start(const JobSpec& spec, std::string& err);
    // Graceful stop: Train jobs get a STOP file in runDir and are killed only
    // after timeoutSec; everything else is killed immediately.
    void stop(JobId id, double timeoutSec);
    void kill(JobId id);
    // Main-thread housekeeping: reap exits, escalate overdue stops, fire onExit.
    void poll();

    bool running(JobId id) const;
    const JobInfo* info(JobId id) const;
    std::vector<JobId> ids() const;          // in launch order
    bool anyRunning(JobKind kind) const;
    bool anyPythonRunning() const;
    bool runInUse(const std::string& runName) const;
    void clearFinished();

    // Line access. Lines are numbered from 0 for the job's lifetime; only the
    // most recent kMaxLines are retained (firstLine() tells where they start).
    size_t lineCount(JobId id) const;
    size_t firstLine(JobId id) const;
    void lines(JobId id, size_t from, std::vector<std::string>& out) const;
    std::string joined(JobId id) const;

    std::function<void(const JobInfo&)> onExit;

    static double now();
    static constexpr size_t kMaxLines = 6000;

private:
    struct Job;
    Job* find(JobId id);
    const Job* find(JobId id) const;

    std::vector<std::unique_ptr<Job>> jobs_;
    JobId next_ = 1;
};

}  // namespace gdv
