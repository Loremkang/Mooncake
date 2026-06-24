// Copyright 2026 KVCache.AI
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

#include "acl/acl.h"
#include "common.h"
#include "common/base/status.h"
#include "transfer_engine.h"
#include "transport/transport.h"

const static int NR_SOCKETS =
    numa_available() == 0 ? numa_num_configured_nodes() : 1;

DEFINE_string(local_server_name, mooncake::getHostname(),
              "Local server name for P2P segment discovery");
DEFINE_string(metadata_server, P2PHANDSHAKE,
              "Metadata server address; this tool is intended for P2PHANDSHAKE");
DEFINE_string(mode, "initiator",
              "Running mode. Only initiator is supported; use "
              "pclab_host_tcp_incast_bench for target mode");
DEFINE_string(operation, "write", "Operation type. Only write is supported");
DEFINE_string(source_memory, "ascend", "Source memory: ascend or cpu");
DEFINE_string(target_segments, "",
              "Comma-separated target P2P segment names, one per target TE");
DEFINE_string(segment_id, "",
              "Deprecated alias for --target_segments with one target TE");
DEFINE_uint64(buffer_size, 1ull << 28, "Size of each local source buffer");
DEFINE_uint64(block_size, 65536, "Block size for each transfer request");
DEFINE_int32(batch_size, 64, "Batch size");
DEFINE_int32(duration, 10, "Test duration in seconds");
DEFINE_int32(threads_per_source_te, 4,
             "Worker threads inside this source TE process");
DEFINE_int32(buffers, 1, "Local source buffer count");
DEFINE_int32(source_rank, 0, "Source TE rank in [0, source_world_size)");
DEFINE_int32(source_world_size, 1, "Number of source TE processes");
DEFINE_int32(target_world_size, 1, "Number of target TE processes");
DEFINE_string(target_selection, "mod",
              "Target selection policy: mod or all_to_all");
DEFINE_uint64(device_id, 65536,
              "Deprecated alias that sets both device_logicid and device_phyid");
DEFINE_uint64(device_logicid, 0, "Ascend device logical ID");
DEFINE_uint64(device_phyid, 0,
              "Ascend device physical ID, kept for log symmetry");
DEFINE_int32(initiator_threads, 0,
             "Deprecated alias for --threads_per_source_te");
DEFINE_int32(rank, -1, "Deprecated alias for --source_rank");
DEFINE_int32(world_size, 0, "Deprecated alias for --source_world_size");
DEFINE_bool(init_pattern, true,
            "Initialize source buffers with deterministic data");
DEFINE_string(report_unit, "Gb", "Report unit: GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb");
DEFINE_uint32(report_precision, 2, "Report precision");

using namespace mooncake;

static std::atomic<bool> running(true);
static std::atomic<size_t> total_batch_count(0);

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

static void normalizeDeprecatedFlags() {
    if (FLAGS_initiator_threads > 0)
        FLAGS_threads_per_source_te = FLAGS_initiator_threads;
    if (FLAGS_rank >= 0) FLAGS_source_rank = FLAGS_rank;
    if (FLAGS_world_size > 0) FLAGS_source_world_size = FLAGS_world_size;
    if (!FLAGS_segment_id.empty() && FLAGS_target_segments.empty())
        FLAGS_target_segments = FLAGS_segment_id;
    if (FLAGS_device_id != 65536) {
        FLAGS_device_logicid = FLAGS_device_id;
        FLAGS_device_phyid = FLAGS_device_id;
    }
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

static std::vector<void*> allocateCpuBuffers(uint64_t size, int count) {
    std::vector<void*> buffers(count);
    for (int i = 0; i < count; ++i) {
        int numa_node = i % std::max(1, NR_SOCKETS);
        buffers[i] = numa_alloc_onnode(size, numa_node);
        CHECK(buffers[i]) << "numa_alloc_onnode failed";
    }
    return buffers;
}

static void freeCpuBuffers(std::vector<void*>& buffers, uint64_t size) {
    for (void* addr : buffers) numa_free(addr, size);
    buffers.clear();
}

static std::vector<void*> allocateAclHostBuffers(uint64_t size, int count) {
    std::vector<void*> buffers(count);
    for (int i = 0; i < count; ++i) {
        void* addr = nullptr;
        aclError ret = aclrtMallocHost(&addr, size);
        CHECK(ret == ACL_ERROR_NONE && addr)
            << "aclrtMallocHost failed, ret=" << ret;
        std::memset(addr, 0, size);
        buffers[i] = addr;
    }
    return buffers;
}

static void freeAclHostBuffers(std::vector<void*>& buffers) {
    for (void* addr : buffers) {
        aclError ret = aclrtFreeHost(addr);
        if (ret != ACL_ERROR_NONE) {
            LOG(WARNING) << "aclrtFreeHost failed, ret=" << ret
                         << ", addr=" << addr;
        }
    }
    buffers.clear();
}

static std::vector<void*> allocateAscendDeviceBuffers(uint64_t size,
                                                      int count) {
    std::vector<void*> buffers(count);
    for (int i = 0; i < count; ++i) {
        void* addr = nullptr;
        aclError ret = aclrtMalloc(&addr, size, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK(ret == ACL_ERROR_NONE && addr)
            << "aclrtMalloc failed, ret=" << ret << ", size=" << size;
        buffers[i] = addr;
    }
    return buffers;
}

static void freeAscendDeviceBuffers(std::vector<void*>& buffers) {
    for (void* addr : buffers) {
        aclError ret = aclrtFree(addr);
        if (ret != ACL_ERROR_NONE) {
            LOG(WARNING) << "aclrtFree failed, ret=" << ret
                         << ", addr=" << addr;
        }
    }
    buffers.clear();
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

static bool copyHostPatternToAscendDevice(std::vector<void*>& device_buffers,
                                          int target_count_for_source) {
    auto host_buffers =
        allocateAclHostBuffers(FLAGS_buffer_size, device_buffers.size());
    if (FLAGS_init_pattern)
        fillInitiatorSourceBuffers(host_buffers, target_count_for_source);

    for (size_t i = 0; i < device_buffers.size(); ++i) {
        aclError ret =
            aclrtMemcpy(device_buffers[i], FLAGS_buffer_size, host_buffers[i],
                        FLAGS_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "aclrtMemcpy H2D failed, ret=" << ret
                       << ", buffer_index=" << i
                       << ", size=" << FLAGS_buffer_size;
            freeAclHostBuffers(host_buffers);
            return false;
        }
    }

    freeAclHostBuffers(host_buffers);
    return true;
}

static bool validateFlags() {
    if (FLAGS_batch_size <= 0 || FLAGS_block_size == 0 ||
        FLAGS_buffer_size == 0 || FLAGS_source_world_size <= 0 ||
        FLAGS_target_world_size <= 0 || FLAGS_threads_per_source_te <= 0 ||
        FLAGS_buffers <= 0) {
        LOG(ERROR) << "Invalid non-positive benchmark flags";
        return false;
    }
    if (FLAGS_mode != "initiator") {
        LOG(ERROR) << "--mode must be initiator for this source-side bench";
        return false;
    }
    if (FLAGS_operation != "write") {
        LOG(ERROR) << "--operation=write is required for Ascend TCP staging";
        return false;
    }
    if (FLAGS_source_memory != "ascend" && FLAGS_source_memory != "cpu") {
        LOG(ERROR) << "--source_memory must be ascend or cpu";
        return false;
    }
    if (FLAGS_target_selection != "mod" &&
        FLAGS_target_selection != "all_to_all") {
        LOG(ERROR) << "--target_selection must be mod or all_to_all";
        return false;
    }
    if (FLAGS_source_rank < 0 ||
        FLAGS_source_rank >= FLAGS_source_world_size) {
        LOG(ERROR) << "--source_rank must be in [0, source_world_size)";
        return false;
    }
    if (FLAGS_target_segments.empty()) {
        LOG(ERROR) << "--target_segments is required";
        return false;
    }
    auto target_segments = splitCommaList(FLAGS_target_segments);
    if (int(target_segments.size()) != FLAGS_target_world_size) {
        LOG(ERROR) << "--target_segments count (" << target_segments.size()
                   << ") must equal --target_world_size ("
                   << FLAGS_target_world_size << ")";
        return false;
    }
    return true;
}

class AclRuntimeGuard {
   public:
    bool init(int device_logic_id) {
        device_logic_id_ = device_logic_id;
        aclError ret = aclInit(nullptr);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "aclInit failed, ret=" << ret;
            return false;
        }
        initialized_ = true;

        ret = aclrtSetDevice(device_logic_id_);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "aclrtSetDevice failed, ret=" << ret
                       << ", device_logicid=" << device_logic_id_;
            return false;
        }
        device_set_ = true;

        ret = aclrtCreateContext(&context_, device_logic_id_);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "aclrtCreateContext failed, ret=" << ret
                       << ", device_logicid=" << device_logic_id_;
            return false;
        }
        return true;
    }

    ~AclRuntimeGuard() {
        if (context_) {
            aclError ret = aclrtDestroyContext(context_);
            if (ret != ACL_ERROR_NONE)
                LOG(WARNING) << "aclrtDestroyContext failed, ret=" << ret;
        }
        if (device_set_) {
            aclError ret = aclrtResetDevice(device_logic_id_);
            if (ret != ACL_ERROR_NONE)
                LOG(WARNING) << "aclrtResetDevice failed, ret=" << ret;
        }
        if (initialized_) {
            aclError ret = aclFinalize();
            if (ret != ACL_ERROR_NONE)
                LOG(WARNING) << "aclFinalize failed, ret=" << ret;
        }
    }

   private:
    bool initialized_{};
    bool device_set_{};
    int device_logic_id_{};
    aclrtContext context_{};
};

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

    size_t batch_count = 0;
    while (running.load(std::memory_order_relaxed)) {
        auto batch_id =
            engine->allocateBatchID(FLAGS_batch_size * target_plans.size());
        std::vector<TransferRequest> requests;
        requests.reserve(FLAGS_batch_size * target_plans.size());
        for (const auto& target : target_plans) {
            for (int i = 0; i < FLAGS_batch_size; ++i) {
                TransferRequest entry;
                entry.opcode = TransferRequest::WRITE;
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
    auto device_id = std::to_string(FLAGS_device_logicid);
    setenv("MC_ASCEND_TCP_DEVICE_ID", device_id.c_str(), 1);

    auto engine = std::make_unique<TransferEngine>(false);
    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name);
    int ret = engine->init(FLAGS_metadata_server, FLAGS_local_server_name.c_str(),
                           hostname_port.first.c_str(), hostname_port.second);
    LOG_ASSERT(!ret) << "TransferEngine init failed, ret=" << ret;
    auto* xport = engine->installTransport("ascend_tcp", nullptr);
    LOG_ASSERT(xport) << "Failed to install ascend_tcp transport";

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

    uint64_t min_local_span = sourceSpanBytes(target_plans.size());
    if (FLAGS_buffer_size < min_local_span) {
        LOG(WARNING) << "Adjusting local buffer size to " << min_local_span;
        FLAGS_buffer_size = min_local_span;
    }

    std::vector<void*> source_buffers;
    bool source_is_ascend = FLAGS_source_memory == "ascend";
    if (source_is_ascend) {
        source_buffers =
            allocateAscendDeviceBuffers(FLAGS_buffer_size, FLAGS_buffers);
        if (!copyHostPatternToAscendDevice(source_buffers,
                                           target_plans.size())) {
            freeAscendDeviceBuffers(source_buffers);
            return EXIT_FAILURE;
        }
    } else {
        source_buffers = allocateCpuBuffers(FLAGS_buffer_size, FLAGS_buffers);
        for (void* addr : source_buffers) std::memset(addr, 0, FLAGS_buffer_size);
        if (FLAGS_init_pattern)
            fillInitiatorSourceBuffers(source_buffers, target_plans.size());
    }

    LOG(INFO) << "PCLAB Ascend TCP HBM initiator layout: source_rank="
              << FLAGS_source_rank << "/" << FLAGS_source_world_size
              << ", target_world_size=" << FLAGS_target_world_size
              << ", selected_targets=" << target_plans.size()
              << ", target_selection=" << FLAGS_target_selection
              << ", source_memory=" << FLAGS_source_memory
              << ", device_logicid=" << FLAGS_device_logicid
              << ", device_phyid=" << FLAGS_device_phyid
              << ", source_buffers=" << source_buffers.size()
              << ", threads_per_source_te=" << FLAGS_threads_per_source_te
              << ", block_size=" << FLAGS_block_size
              << ", batch_size=" << FLAGS_batch_size
              << ", buffer_size=" << FLAGS_buffer_size;
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
                                 source_buffers[i % source_buffers.size()]);
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

    if (source_is_ascend) {
        freeAscendDeviceBuffers(source_buffers);
    } else {
        freeCpuBuffers(source_buffers, FLAGS_buffer_size);
    }
    return 0;
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    normalizeDeprecatedFlags();
    if (!validateFlags()) return EXIT_FAILURE;

    AclRuntimeGuard acl_guard;
    if (!acl_guard.init(static_cast<int>(FLAGS_device_logicid)))
        return EXIT_FAILURE;

    return initiator();
}
