// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "common/base/status.h"
#include "cuda_alike.h"
#include "transfer_engine.h"
#include "transport/transport.h"

const static int NR_SOCKETS =
    numa_available() == 0 ? numa_num_configured_nodes() : 1;

DEFINE_string(local_server_name, mooncake::getHostname(),
              "Local server name for P2P segment discovery");
DEFINE_string(metadata_server, P2PHANDSHAKE,
              "Metadata server address; this tool is intended for P2PHANDSHAKE");
DEFINE_string(mode, "initiator", "Running mode: initiator or target");
DEFINE_string(operation, "write", "Operation type: read or write");
DEFINE_string(target_segments, "",
              "Comma-separated target P2P segment names, one per target TE");
DEFINE_string(segment_id, "",
              "Deprecated alias for --target_segments with one target TE");
DEFINE_uint64(buffer_size, 1ull << 28, "Size of each local registered buffer");
DEFINE_uint64(block_size, 65536, "Block size for each transfer request");
DEFINE_int32(batch_size, 64, "Batch size");
DEFINE_int32(duration, 10, "Test duration in seconds");
DEFINE_int32(threads_per_source_te, 4,
             "Worker threads inside each source TE process");
DEFINE_int32(expected_workers_per_source_te, 4,
             "Target-side expected workers inside each source TE process");
DEFINE_int32(buffers, 0, "Local buffer count; 0 uses NUMA node count");
DEFINE_int32(source_rank, 0, "Source TE rank in [0, source_world_size)");
DEFINE_int32(source_world_size, 1, "Number of source TE processes");
DEFINE_int32(target_rank, 0, "Target TE rank in [0, target_world_size)");
DEFINE_int32(target_world_size, 1, "Number of target TE processes");
DEFINE_string(target_selection, "mod",
              "Target selection policy: mod or all_to_all");
DEFINE_string(target_memory, "cpu", "Target memory in target mode: cpu or cuda");
DEFINE_int32(cuda_device_id, 0, "CUDA device ID for target_memory=cuda");
DEFINE_int32(initiator_threads, 0,
             "Deprecated alias for --threads_per_source_te");
DEFINE_int32(expected_threads_per_initiator, 0,
             "Deprecated alias for --expected_workers_per_source_te");
DEFINE_int32(rank, -1, "Deprecated alias for --source_rank");
DEFINE_int32(world_size, 0, "Deprecated alias for --source_world_size");
DEFINE_bool(init_pattern, true,
            "Initialize initiator source buffers with deterministic data");
DEFINE_bool(verify_on_exit, false,
            "Target verifies all source regions after SIGINT/SIGTERM");
DEFINE_string(report_unit, "Gb", "Report unit: GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb");
DEFINE_uint32(report_precision, 2, "Report precision");

using namespace mooncake;

static std::atomic<bool> running(true);
static std::atomic<size_t> total_batch_count(0);
static std::atomic<bool> target_running(true);

const static std::unordered_map<std::string, uint64_t> RATE_UNIT_MP = {
    {"GB", 1000ull * 1000ull * 1000ull},
    {"GiB", 1ull << 30},
    {"Gb", 1000ull * 1000ull * 1000ull / 8},
    {"MB", 1000ull * 1000ull},
    {"MiB", 1ull << 20},
    {"Mb", 1000ull * 1000ull / 8},
    {"KB", 1000ull},
    {"KiB", 1ull << 10},
    {"Kb", 1000ull / 8}};

static std::string calculateRate(uint64_t data_bytes, double duration) {
    if (std::fabs(duration) < 1e-10) return "";
    if (!RATE_UNIT_MP.count(FLAGS_report_unit)) FLAGS_report_unit = "Gb";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(FLAGS_report_precision)
        << 1.0 * data_bytes / duration / RATE_UNIT_MP.at(FLAGS_report_unit)
        << " " << FLAGS_report_unit << "/s";
    return oss.str();
}

static void signalHandler(int /* signum */) {
    target_running.store(false, std::memory_order_relaxed);
}

static void setupSignalHandler() {
    struct sigaction sa;
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

static int localBufferCount() {
    if (FLAGS_buffers > 0) return FLAGS_buffers;
    return std::max(1, NR_SOCKETS);
}

static void normalizeDeprecatedFlags() {
    if (FLAGS_initiator_threads > 0)
        FLAGS_threads_per_source_te = FLAGS_initiator_threads;
    if (FLAGS_expected_threads_per_initiator > 0)
        FLAGS_expected_workers_per_source_te =
            FLAGS_expected_threads_per_initiator;
    if (FLAGS_rank >= 0) FLAGS_source_rank = FLAGS_rank;
    if (FLAGS_world_size > 0) FLAGS_source_world_size = FLAGS_world_size;
    if (!FLAGS_segment_id.empty() && FLAGS_target_segments.empty())
        FLAGS_target_segments = FLAGS_segment_id;
}

static std::string trim(std::string value) {
    auto begin = value.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) return "";
    auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(begin, end - begin + 1);
}

static std::vector<std::string> splitCommaList(const std::string& input) {
    std::vector<std::string> values;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) values.push_back(item);
    }
    return values;
}

static bool isAllToAll() { return FLAGS_target_selection == "all_to_all"; }

static bool sourceWritesTarget(int source_rank, int target_rank) {
    if (isAllToAll()) return true;
    return source_rank % FLAGS_target_world_size == target_rank;
}

static int sourceRegionCountForTarget(int target_rank) {
    if (isAllToAll()) return FLAGS_source_world_size;
    if (target_rank >= FLAGS_source_world_size) return 0;
    return (FLAGS_source_world_size + FLAGS_target_world_size - 1 -
            target_rank) /
           FLAGS_target_world_size;
}

static int sourceLocalIndexForTarget(int source_rank, int target_rank) {
    if (isAllToAll()) return source_rank;
    if (!sourceWritesTarget(source_rank, target_rank)) return -1;
    return source_rank / FLAGS_target_world_size;
}

static int sourceRankForTargetLocalIndex(int target_rank, int local_index) {
    if (isAllToAll()) return local_index;
    return target_rank + local_index * FLAGS_target_world_size;
}

static uint64_t sourceOffset(int request_index, int source_target_ordinal,
                             int thread_id, int target_count_for_source) {
    return FLAGS_block_size *
           uint64_t((request_index * target_count_for_source +
                     source_target_ordinal) *
                        FLAGS_threads_per_source_te +
                    thread_id);
}

static uint64_t sourceSpanBytes(int target_count_for_source) {
    return FLAGS_block_size * uint64_t(FLAGS_batch_size) *
           uint64_t(FLAGS_threads_per_source_te) *
           uint64_t(target_count_for_source);
}

static int threadsForTargetBuffer(int thread_count, int buffer_index,
                                  int buffer_count) {
    if (buffer_index >= thread_count) return 0;
    return (thread_count + buffer_count - 1 - buffer_index) / buffer_count;
}

static uint64_t targetSpanForBuffer(int thread_count, int buffer_index,
                                    int buffer_count) {
    return FLAGS_block_size * uint64_t(FLAGS_batch_size) *
           uint64_t(threadsForTargetBuffer(thread_count, buffer_index,
                                           buffer_count));
}

static uint64_t targetOffsetInRankRegion(int thread_count, int thread_id,
                                         int request_index,
                                         int target_buffer_count) {
    int buffer_index = thread_id % target_buffer_count;
    int thread_ordinal = thread_id / target_buffer_count;
    int threads_in_buffer = threadsForTargetBuffer(thread_count, buffer_index,
                                                  target_buffer_count);
    return FLAGS_block_size *
           uint64_t(request_index * threads_in_buffer + thread_ordinal);
}

static uint8_t patternByte(int rank, int thread_id, int request_index,
                           uint64_t byte_index) {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    x ^= uint64_t(rank + 1) * 0xbf58476d1ce4e5b9ull;
    x ^= uint64_t(thread_id + 1) * 0x94d049bb133111ebull;
    x ^= uint64_t(request_index + 1) * 0x2545f4914f6cdd1dull;
    x ^= byte_index * 0x100000001b3ull;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return static_cast<uint8_t>(x & 0xff);
}

static void fillPattern(uint8_t* base, int rank, int thread_id,
                        int request_index) {
    for (uint64_t j = 0; j < FLAGS_block_size; ++j)
        base[j] = patternByte(rank, thread_id, request_index, j);
}

static bool verifyPattern(const uint8_t* base, int rank, int thread_id,
                          int request_index) {
    for (uint64_t j = 0; j < FLAGS_block_size; ++j) {
        uint8_t expected = patternByte(rank, thread_id, request_index, j);
        if (base[j] != expected) {
            LOG(ERROR) << "Verify mismatch: rank=" << rank
                       << " thread=" << thread_id
                       << " request=" << request_index << " byte=" << j
                       << " expected=" << uint32_t(expected)
                       << " actual=" << uint32_t(base[j]);
            return false;
        }
    }
    return true;
}

static std::vector<void*> allocateBuffers(uint64_t size, int count) {
    std::vector<void*> buffers(count);
    for (int i = 0; i < count; ++i) {
        int numa_node = i % std::max(1, NR_SOCKETS);
        buffers[i] = numa_alloc_onnode(size, numa_node);
        CHECK(buffers[i]) << "numa_alloc_onnode failed";
    }
    return buffers;
}

static void freeBuffers(std::vector<void*>& buffers, uint64_t size) {
    for (void* addr : buffers) numa_free(addr, size);
    buffers.clear();
}

static bool targetUsesCuda() { return FLAGS_target_memory == "cuda"; }

static std::vector<void*> allocateTargetBuffers(uint64_t size, int count) {
    if (!targetUsesCuda()) return allocateBuffers(size, count);

#ifdef USE_CUDA
    cudaError_t err = cudaSetDevice(FLAGS_cuda_device_id);
    CHECK(err == cudaSuccess) << "cudaSetDevice failed: "
                              << cudaGetErrorString(err)
                              << ", device=" << FLAGS_cuda_device_id;

    std::vector<void*> buffers(count);
    for (int i = 0; i < count; ++i) {
        err = cudaMalloc(&buffers[i], size);
        CHECK(err == cudaSuccess && buffers[i])
            << "cudaMalloc failed: " << cudaGetErrorString(err)
            << ", size=" << size << ", device=" << FLAGS_cuda_device_id;
    }
    return buffers;
#else
    LOG(FATAL) << "target_memory=cuda requires USE_CUDA";
    return {};
#endif
}

static void initializeTargetBuffers(std::vector<void*>& buffers,
                                    uint64_t size) {
    if (!targetUsesCuda()) {
        for (void* addr : buffers) memset(addr, 0xa5, size);
        return;
    }

#ifdef USE_CUDA
    cudaError_t err = cudaSetDevice(FLAGS_cuda_device_id);
    CHECK(err == cudaSuccess) << "cudaSetDevice failed: "
                              << cudaGetErrorString(err)
                              << ", device=" << FLAGS_cuda_device_id;
    for (void* addr : buffers) {
        err = cudaMemset(addr, 0xa5, size);
        CHECK(err == cudaSuccess) << "cudaMemset failed: "
                                  << cudaGetErrorString(err)
                                  << ", addr=" << addr << ", size=" << size;
    }
    err = cudaDeviceSynchronize();
    CHECK(err == cudaSuccess) << "cudaDeviceSynchronize failed after memset: "
                              << cudaGetErrorString(err);
#else
    LOG(FATAL) << "target_memory=cuda requires USE_CUDA";
#endif
}

static void freeTargetBuffers(std::vector<void*>& buffers, uint64_t size) {
    if (!targetUsesCuda()) {
        freeBuffers(buffers, size);
        return;
    }

#ifdef USE_CUDA
    cudaError_t err = cudaSetDevice(FLAGS_cuda_device_id);
    if (err != cudaSuccess) {
        LOG(WARNING) << "cudaSetDevice failed during free: "
                     << cudaGetErrorString(err)
                     << ", device=" << FLAGS_cuda_device_id;
    }
    for (void* addr : buffers) {
        err = cudaFree(addr);
        if (err != cudaSuccess) {
            LOG(WARNING) << "cudaFree failed: " << cudaGetErrorString(err)
                         << ", addr=" << addr;
        }
    }
    buffers.clear();
#else
    LOG(FATAL) << "target_memory=cuda requires USE_CUDA";
#endif
}

static void fillInitiatorSourceBuffers(std::vector<void*>& source_buffers,
                                       int target_count_for_source) {
    for (int thread_id = 0; thread_id < FLAGS_threads_per_source_te;
         ++thread_id) {
        auto* base = static_cast<uint8_t*>(
            source_buffers[thread_id % source_buffers.size()]);
        for (int i = 0; i < FLAGS_batch_size; ++i) {
            for (int target_ordinal = 0;
                 target_ordinal < target_count_for_source; ++target_ordinal) {
                uint64_t local_offset =
                    sourceOffset(i, target_ordinal, thread_id,
                                 target_count_for_source);
                fillPattern(base + local_offset, FLAGS_source_rank, thread_id,
                            i);
            }
        }
    }
}

static bool validateFlags() {
    if (FLAGS_batch_size <= 0 || FLAGS_block_size == 0 ||
        FLAGS_buffer_size == 0 || FLAGS_source_world_size <= 0 ||
        FLAGS_target_world_size <= 0) {
        LOG(ERROR) << "Invalid non-positive benchmark flags";
        return false;
    }
    if (FLAGS_mode != "initiator" && FLAGS_mode != "target") {
        LOG(ERROR) << "--mode must be initiator or target";
        return false;
    }
    if (FLAGS_target_selection != "mod" &&
        FLAGS_target_selection != "all_to_all") {
        LOG(ERROR) << "--target_selection must be mod or all_to_all";
        return false;
    }
    if (FLAGS_mode == "initiator" && FLAGS_threads_per_source_te <= 0) {
        LOG(ERROR) << "--threads_per_source_te must be positive";
        return false;
    }
    if (FLAGS_mode == "target" && FLAGS_expected_workers_per_source_te <= 0) {
        LOG(ERROR) << "--expected_workers_per_source_te must be positive";
        return false;
    }
    if (FLAGS_mode == "initiator" &&
        (FLAGS_source_rank < 0 ||
         FLAGS_source_rank >= FLAGS_source_world_size)) {
        LOG(ERROR) << "--source_rank must be in [0, source_world_size)";
        return false;
    }
    if (FLAGS_mode == "target" &&
        (FLAGS_target_rank < 0 ||
         FLAGS_target_rank >= FLAGS_target_world_size)) {
        LOG(ERROR) << "--target_rank must be in [0, target_world_size)";
        return false;
    }
    if (FLAGS_mode == "initiator" && FLAGS_target_segments.empty()) {
        LOG(ERROR) << "--target_segments is required in initiator mode";
        return false;
    }
    if (FLAGS_mode == "initiator") {
        auto target_segments = splitCommaList(FLAGS_target_segments);
        if (int(target_segments.size()) != FLAGS_target_world_size) {
            LOG(ERROR) << "--target_segments count (" << target_segments.size()
                       << ") must equal --target_world_size ("
                       << FLAGS_target_world_size << ")";
            return false;
        }
    }
    if (FLAGS_target_selection == "all_to_all" && FLAGS_operation == "read") {
        LOG(ERROR) << "--target_selection=all_to_all currently supports "
                      "--operation=write only";
        return false;
    }
    if (FLAGS_operation != "read" && FLAGS_operation != "write") {
        LOG(ERROR) << "--operation must be read or write";
        return false;
    }
    if (FLAGS_target_memory != "cpu" && FLAGS_target_memory != "cuda") {
        LOG(ERROR) << "--target_memory must be cpu or cuda";
        return false;
    }
    if (FLAGS_mode == "initiator" && FLAGS_target_memory != "cpu") {
        LOG(ERROR) << "--target_memory is target-mode only";
        return false;
    }
#ifndef USE_CUDA
    if (FLAGS_mode == "target" && targetUsesCuda()) {
        LOG(ERROR) << "--target_memory=cuda requires USE_CUDA build";
        return false;
    }
#endif
    return true;
}

struct WorkerTargetPlan {
    SegmentID segment_id;
    int target_rank;
    int target_buffer_count;
    uint64_t remote_base;
    int source_target_ordinal;
};

struct TargetPlan {
    std::string segment_name;
    SegmentID segment_id;
    int target_rank;
    int source_local_index;
    int target_buffer_count;
    std::vector<uint64_t> thread_remote_base;
};

static Status initiatorWorker(TransferEngine* engine,
                              std::vector<WorkerTargetPlan> target_plans,
                              int thread_id, void* source_buffer) {
    bindToSocket(thread_id % NR_SOCKETS);
    TransferRequest::OpCode opcode =
        FLAGS_operation == "read" ? TransferRequest::READ
                                  : TransferRequest::WRITE;

    size_t batch_count = 0;
    while (running.load(std::memory_order_relaxed)) {
        auto batch_id =
            engine->allocateBatchID(FLAGS_batch_size * target_plans.size());
        std::vector<TransferRequest> requests;
        requests.reserve(FLAGS_batch_size * target_plans.size());
        for (const auto& target : target_plans) {
            for (int i = 0; i < FLAGS_batch_size; ++i) {
                TransferRequest entry;
                entry.opcode = opcode;
                entry.length = FLAGS_block_size;
                entry.source =
                    static_cast<uint8_t*>(source_buffer) +
                    sourceOffset(i, target.source_target_ordinal, thread_id,
                                 target_plans.size());
                entry.target_id = target.segment_id;
                entry.target_offset =
                    target.remote_base +
                    targetOffsetInRankRegion(FLAGS_threads_per_source_te,
                                             thread_id, i,
                                             target.target_buffer_count);
                requests.emplace_back(entry);
            }
        }

        Status s = engine->submitTransfer(batch_id, requests);
        LOG_ASSERT(s.ok()) << s.ToString();
        for (int task_id = 0; task_id < int(requests.size()); ++task_id) {
            TransferStatus status;
            while (true) {
                s = engine->getTransferStatus(batch_id, task_id, status);
                LOG_ASSERT(s.ok()) << s.ToString();
                if (status.s == TransferStatusEnum::COMPLETED) break;
                if (status.s == TransferStatusEnum::FAILED) {
                    LOG(ERROR) << "Transfer failed";
                    exit(EXIT_FAILURE);
                }
            }
        }
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok()) << s.ToString();
        ++batch_count;
    }
    total_batch_count.fetch_add(batch_count);
    LOG(INFO) << "Worker " << thread_id << " stopped";
    return Status::OK();
}

static int initiator() {
    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name.c_str(),
                 hostname_port.first.c_str(), hostname_port.second);
    auto* xport = engine->installTransport("tcp", nullptr);
    LOG_ASSERT(xport);

    auto target_segment_names = splitCommaList(FLAGS_target_segments);
    std::vector<TargetPlan> target_plans;
    target_plans.reserve(target_segment_names.size());
    for (int target_rank = 0; target_rank < int(target_segment_names.size());
         ++target_rank) {
        if (!sourceWritesTarget(FLAGS_source_rank, target_rank)) continue;

        TargetPlan plan;
        plan.segment_name = target_segment_names[target_rank];
        plan.target_rank = target_rank;
        plan.source_local_index =
            sourceLocalIndexForTarget(FLAGS_source_rank, target_rank);
        LOG_ASSERT(plan.source_local_index >= 0);
        plan.segment_id = engine->openSegment(plan.segment_name.c_str());
        auto segment_desc =
            engine->getMetadata()->getSegmentDescByID(plan.segment_id);
        LOG_ASSERT(segment_desc);
        LOG_ASSERT(!segment_desc->buffers.empty())
            << "Target segment " << plan.segment_name
            << " has no registered buffers";

        plan.target_buffer_count = segment_desc->buffers.size();
        int source_regions = sourceRegionCountForTarget(target_rank);
        LOG_ASSERT(source_regions > 0)
            << "Target rank " << target_rank << " has no source regions";
        plan.thread_remote_base.resize(FLAGS_threads_per_source_te);
        for (int thread_id = 0; thread_id < FLAGS_threads_per_source_te;
             ++thread_id) {
            int target_buffer = thread_id % plan.target_buffer_count;
            const auto& remote_buffer = segment_desc->buffers[target_buffer];
            uint64_t source_region_size =
                remote_buffer.length / uint64_t(source_regions);
            uint64_t required_span =
                targetSpanForBuffer(FLAGS_threads_per_source_te, target_buffer,
                                    plan.target_buffer_count);
            LOG_ASSERT(source_region_size >= required_span)
                << "Remote target rank " << target_rank << " buffer "
                << target_buffer
                << " source region too small: source_region_size="
                << source_region_size << " required_span=" << required_span;
            plan.thread_remote_base[thread_id] =
                remote_buffer.addr +
                uint64_t(plan.source_local_index) * source_region_size;
        }
        target_plans.emplace_back(std::move(plan));
    }
    LOG_ASSERT(!target_plans.empty())
        << "No target TE selected for source_rank=" << FLAGS_source_rank
        << " with target_selection=" << FLAGS_target_selection;

    int source_buffer_count = localBufferCount();
    uint64_t min_local_span = sourceSpanBytes(target_plans.size());
    if (FLAGS_buffer_size < min_local_span) {
        LOG(WARNING) << "Adjusting local buffer size to " << min_local_span;
        FLAGS_buffer_size = min_local_span;
    }

    auto source_buffers = allocateBuffers(FLAGS_buffer_size, source_buffer_count);
    for (int i = 0; i < source_buffer_count; ++i) {
        int rc = engine->registerLocalMemory(
            source_buffers[i], FLAGS_buffer_size,
            "cpu:" + std::to_string(i % std::max(1, NR_SOCKETS)));
        LOG_ASSERT(!rc);
    }
    if (FLAGS_operation == "write" && FLAGS_init_pattern)
        fillInitiatorSourceBuffers(source_buffers, target_plans.size());

    LOG(INFO) << "Incast initiator layout: source_rank=" << FLAGS_source_rank
              << "/" << FLAGS_source_world_size
              << ", target_world_size=" << FLAGS_target_world_size
              << ", selected_targets=" << target_plans.size()
              << ", target_selection=" << FLAGS_target_selection
              << ", source_buffers=" << source_buffer_count
              << ", threads_per_source_te=" << FLAGS_threads_per_source_te
              << ", block_size=" << FLAGS_block_size
              << ", batch_size=" << FLAGS_batch_size;
    for (const auto& plan : target_plans) {
        LOG(INFO) << "Selected target: target_rank=" << plan.target_rank
                  << ", source_local_index=" << plan.source_local_index
                  << ", segment=" << plan.segment_name
                  << ", target_buffers=" << plan.target_buffer_count;
    }

    std::vector<std::thread> workers(FLAGS_threads_per_source_te);
    struct timeval start_tv, stop_tv;
    gettimeofday(&start_tv, nullptr);
    for (int i = 0; i < FLAGS_threads_per_source_te; ++i) {
        std::vector<WorkerTargetPlan> worker_targets;
        worker_targets.reserve(target_plans.size());
        for (int target_ordinal = 0; target_ordinal < int(target_plans.size());
             ++target_ordinal) {
            const auto& target = target_plans[target_ordinal];
            WorkerTargetPlan worker_target;
            worker_target.segment_id = target.segment_id;
            worker_target.target_rank = target.target_rank;
            worker_target.target_buffer_count = target.target_buffer_count;
            worker_target.remote_base = target.thread_remote_base[i];
            worker_target.source_target_ordinal = target_ordinal;
            worker_targets.push_back(worker_target);
        }
        workers[i] = std::thread(initiatorWorker, engine.get(),
                                 std::move(worker_targets), i,
                                 source_buffers[i % source_buffer_count]);
    }

    sleep(FLAGS_duration);
    running.store(false, std::memory_order_relaxed);
    for (auto& worker : workers) worker.join();

    gettimeofday(&stop_tv, nullptr);
    double duration = (stop_tv.tv_sec - start_tv.tv_sec) +
                      (stop_tv.tv_usec - start_tv.tv_usec) / 1000000.0;
    auto batch_count = total_batch_count.load();
    LOG(INFO) << "Test completed: duration " << std::fixed
              << std::setprecision(2) << duration << ", batch count "
              << batch_count << ", throughput "
              << calculateRate(
                     batch_count * FLAGS_batch_size * FLAGS_block_size *
                         target_plans.size(),
                     duration);

    for (void* addr : source_buffers) engine->unregisterLocalMemory(addr);
    freeBuffers(source_buffers, FLAGS_buffer_size);
    return 0;
}

static bool verifyTargetBuffers(const std::vector<void*>& target_buffers,
                                uint64_t buffer_size) {
    bool ok = true;
    int target_buffer_count = target_buffers.size();
    int source_regions = sourceRegionCountForTarget(FLAGS_target_rank);
    if (source_regions == 0) {
        LOG(INFO) << "Target rank " << FLAGS_target_rank
                  << " has no assigned source regions; skip verification";
        return true;
    }
    for (int b = 0; b < target_buffer_count; ++b) {
        std::vector<uint8_t> cuda_host_copy;
        const uint8_t* buffer_base = nullptr;
        if (targetUsesCuda()) {
#ifdef USE_CUDA
            cuda_host_copy.resize(buffer_size);
            cudaError_t err = cudaSetDevice(FLAGS_cuda_device_id);
            CHECK(err == cudaSuccess) << "cudaSetDevice failed before verify: "
                                      << cudaGetErrorString(err)
                                      << ", device=" << FLAGS_cuda_device_id;
            err = cudaMemcpy(cuda_host_copy.data(), target_buffers[b],
                             buffer_size, cudaMemcpyDeviceToHost);
            CHECK(err == cudaSuccess) << "cudaMemcpy D2H verify failed: "
                                      << cudaGetErrorString(err)
                                      << ", buffer_index=" << b
                                      << ", size=" << buffer_size;
            buffer_base = cuda_host_copy.data();
#else
            LOG(FATAL) << "target_memory=cuda requires USE_CUDA";
#endif
        } else {
            buffer_base = static_cast<const uint8_t*>(target_buffers[b]);
        }
        uint64_t source_region_size = buffer_size / uint64_t(source_regions);
        for (int local_index = 0; local_index < source_regions; ++local_index) {
            int source_rank =
                sourceRankForTargetLocalIndex(FLAGS_target_rank, local_index);
            const uint8_t* source_base =
                buffer_base + uint64_t(local_index) * source_region_size;
            for (int thread_id = 0;
                 thread_id < FLAGS_expected_workers_per_source_te;
                 ++thread_id) {
                if (thread_id % target_buffer_count != b) continue;
                for (int i = 0; i < FLAGS_batch_size; ++i) {
                    uint64_t off = targetOffsetInRankRegion(
                        FLAGS_expected_workers_per_source_te, thread_id, i,
                        target_buffer_count);
                    ok = verifyPattern(source_base + off, source_rank,
                                       thread_id, i) &&
                         ok;
                }
            }
        }
    }
    return ok;
}

static int target() {
    setupSignalHandler();
    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name.c_str(),
                 hostname_port.first.c_str(), hostname_port.second);
    auto* xport = engine->installTransport("tcp", nullptr);
    LOG_ASSERT(xport);

    int target_buffer_count = localBufferCount();
    int source_regions = sourceRegionCountForTarget(FLAGS_target_rank);
    for (int b = 0; b < target_buffer_count; ++b) {
        if (source_regions == 0) break;
        uint64_t required =
            targetSpanForBuffer(FLAGS_expected_workers_per_source_te, b,
                                target_buffer_count);
        uint64_t source_region_size =
            FLAGS_buffer_size / uint64_t(source_regions);
        LOG_ASSERT(source_region_size >= required)
            << "Target rank " << FLAGS_target_rank << " buffer " << b
            << " is too small for source_regions=" << source_regions
            << ": source_region_size=" << source_region_size
            << " required=" << required;
    }

    auto target_buffers =
        allocateTargetBuffers(FLAGS_buffer_size, target_buffer_count);
    initializeTargetBuffers(target_buffers, FLAGS_buffer_size);

    for (int i = 0; i < target_buffer_count; ++i) {
        std::string location =
            targetUsesCuda()
                ? "cuda:" + std::to_string(FLAGS_cuda_device_id)
                : "cpu:" + std::to_string(i % std::max(1, NR_SOCKETS));
        int rc = engine->registerLocalMemory(
            target_buffers[i], FLAGS_buffer_size, location);
        LOG_ASSERT(!rc);
    }

    LOG(INFO) << "PCLAB_TARGET_SESSION target_rank=" << FLAGS_target_rank
              << " session_id=" << engine->getLocalIpAndPort();
    LOG(INFO) << "Incast target layout: target_rank=" << FLAGS_target_rank
              << "/" << FLAGS_target_world_size
              << ", source_world_size=" << FLAGS_source_world_size
              << ", source_regions=" << source_regions
              << ", target_selection=" << FLAGS_target_selection
              << ", target_buffers=" << target_buffer_count
              << ", target_memory=" << FLAGS_target_memory
              << ", cuda_device_id=" << FLAGS_cuda_device_id
              << ", buffer_size=" << FLAGS_buffer_size
              << ", source_region_size="
              << (source_regions > 0
                      ? FLAGS_buffer_size / uint64_t(source_regions)
                      : uint64_t(0))
              << ", expected_workers_per_source_te="
              << FLAGS_expected_workers_per_source_te
              << ", block_size=" << FLAGS_block_size
              << ", batch_size=" << FLAGS_batch_size
              << ", verify_on_exit=" << FLAGS_verify_on_exit;

    while (target_running.load(std::memory_order_relaxed)) sleep(1);

    int exit_code = EXIT_SUCCESS;
    if (FLAGS_verify_on_exit) {
        bool ok = verifyTargetBuffers(target_buffers, FLAGS_buffer_size);
        LOG(INFO) << "Target verification " << (ok ? "PASSED" : "FAILED");
        if (!ok) exit_code = EXIT_FAILURE;
    }

    for (void* addr : target_buffers) engine->unregisterLocalMemory(addr);
    freeTargetBuffers(target_buffers, FLAGS_buffer_size);
    return exit_code;
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    normalizeDeprecatedFlags();
    if (!validateFlags()) return EXIT_FAILURE;

    if (FLAGS_mode == "initiator") return initiator();
    if (FLAGS_mode == "target") return target();
    LOG(ERROR) << "--mode must be initiator or target";
    return EXIT_FAILURE;
}
