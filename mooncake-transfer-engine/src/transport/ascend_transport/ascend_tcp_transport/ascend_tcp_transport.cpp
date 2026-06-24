#include "transport/ascend_transport/ascend_tcp_transport.h"

#include <algorithm>
#include <cstdlib>

namespace mooncake {

namespace {

// ACL memory location types for aclrtPtrAttributes::location.type:
//   0 = ACL host memory, 1 = device memory, 2 = regular CPU memory (malloc)
static constexpr int kDeviceMemoryLocationType = 1;
static constexpr size_t kHostStagingAlignment = 64;

bool isAscendDeviceMemory(void *addr) {
    aclrtPtrAttributes attributes{};
    if (int ret = aclrtPointerGetAttributes(addr, &attributes)) {
        VLOG(1) << "aclrtPointerGetAttributes failed for addr " << addr
                << ", ret: " << ret << ". Treating as CPU memory.";
        return false;
    }
    return attributes.location.type == kDeviceMemoryLocationType;
}

std::shared_ptr<void> allocateHostStagingBuffer(size_t length) {
    const size_t alloc_size =
        ((std::max<size_t>(length, 1) + kHostStagingAlignment - 1) /
         kHostStagingAlignment) *
        kHostStagingAlignment;

    void *ptr = nullptr;
    if (posix_memalign(&ptr, kHostStagingAlignment, alloc_size) != 0) {
        return nullptr;
    }
    return std::shared_ptr<void>(ptr, [](void *p) { std::free(p); });
}

bool isFinishedStatus(TransferStatusEnum status) {
    return status == TransferStatusEnum::COMPLETED ||
           status == TransferStatusEnum::FAILED ||
           status == TransferStatusEnum::TIMEOUT ||
           status == TransferStatusEnum::CANCELED ||
           status == TransferStatusEnum::INVALID;
}

}  // namespace

AscendTcpTransport::AscendTcpTransport() {
    const char *device_id_env = std::getenv("MC_ASCEND_TCP_DEVICE_ID");
    if (device_id_env && *device_id_env) {
        logic_device_id_ = std::atoi(device_id_env);
        logic_device_initialized_ = true;
    } else if (int ret = aclrtGetDevice(&logic_device_id_)) {
        VLOG(1) << "AscendTcpTransport: aclrtGetDevice during construction "
                << "failed, ret: " << ret
                << ". Falling back to logic device 0.";
        logic_device_id_ = 0;
        logic_device_initialized_ = true;
    } else {
        logic_device_initialized_ = true;
    }

    if (int ret = aclrtGetCurrentContext(&context_)) {
        VLOG(1) << "AscendTcpTransport: aclrtGetCurrentContext during "
                << "construction failed, ret: " << ret
                << ". Worker threads will use aclrtSetDevice.";
        context_ = nullptr;
        context_initialized_ = false;
    } else {
        context_initialized_ = context_ != nullptr;
    }
}

AscendTcpTransport::~AscendTcpTransport() {
    if (stream_copy_created_) {
        activateAclContext();
        aclrtDestroyStream(stream_copy_);
        stream_copy_ = nullptr;
        stream_copy_created_ = false;
    }
}

int AscendTcpTransport::activateAclContext() {
    if (context_initialized_) {
        int ret = aclrtSetCurrentContext(context_);
        if (ret) {
            LOG(ERROR) << "AscendTcpTransport: aclrtSetCurrentContext failed, "
                       << "ret: " << ret;
            return ret;
        }
        return 0;
    }

    if (!logic_device_initialized_) {
        const char *device_id_env = std::getenv("MC_ASCEND_TCP_DEVICE_ID");
        if (device_id_env && *device_id_env) {
            logic_device_id_ = std::atoi(device_id_env);
        } else {
            logic_device_id_ = 0;
        }
        logic_device_initialized_ = true;
    }

    int ret = aclrtSetDevice(logic_device_id_);
    if (ret) {
        LOG(ERROR) << "AscendTcpTransport: aclrtSetDevice failed, ret: "
                   << ret << ", device: " << logic_device_id_;
        return ret;
    }
    return 0;
}

int AscendTcpTransport::checkAndCreateStreamCopy() {
    int ret = activateAclContext();
    if (ret) return ret;

    if (!stream_copy_created_) {
        ret = aclrtCreateStream(&stream_copy_);
        if (ret) {
            LOG(ERROR) << "AscendTcpTransport: aclrtCreateStream failed, ret: "
                       << ret;
            return ret;
        }
        stream_copy_created_ = true;
    }
    return 0;
}

Status AscendTcpTransport::copyDeviceToHost(void *host_dst,
                                            const void *device_src,
                                            size_t length) {
    std::lock_guard<std::mutex> lock(copy_mutex_);

    int ret = checkAndCreateStreamCopy();
    if (ret) {
        return Status::InvalidArgument(
            "AscendTcpTransport: failed to create ACL copy stream");
    }

    ret = aclrtMemcpyAsync(host_dst, length, device_src, length,
                           ACL_MEMCPY_DEVICE_TO_HOST, stream_copy_);
    if (ret) {
        LOG(ERROR) << "AscendTcpTransport: aclrtMemcpyAsync D2H failed, ret: "
                   << ret << ", host_dst: " << host_dst
                   << ", device_src: " << device_src << ", length: " << length;
        return Status::InvalidArgument(
            "AscendTcpTransport: aclrtMemcpyAsync D2H failed");
    }

    ret = aclrtSynchronizeStream(stream_copy_);
    if (ret) {
        LOG(ERROR) << "AscendTcpTransport: aclrtSynchronizeStream failed, ret: "
                   << ret;
        return Status::InvalidArgument(
            "AscendTcpTransport: aclrtSynchronizeStream failed");
    }
    return Status::OK();
}

Status AscendTcpTransport::stageTransferRequest(
    const TransferRequest &request, TransferRequest &staged_request,
    std::shared_ptr<void> &staging_buffer) {
    staged_request = request;

    if (!isAscendDeviceMemory(request.source)) {
        return Status::OK();
    }

    if (request.opcode != TransferRequest::WRITE) {
        return Status::NotImplemented(
            "AscendTcpTransport only supports WRITE from Ascend device memory");
    }

    staging_buffer = allocateHostStagingBuffer(request.length);
    if (!staging_buffer) {
        return Status::Memory(
            "AscendTcpTransport: failed to allocate host staging buffer");
    }

    auto status =
        copyDeviceToHost(staging_buffer.get(), request.source, request.length);
    if (!status.ok()) {
        staging_buffer.reset();
        return status;
    }

    staged_request.source = staging_buffer.get();
    return Status::OK();
}

void AscendTcpTransport::retainStagingBuffer(
    TransferTask *task, std::shared_ptr<void> staging_buffer) {
    if (!task || !staging_buffer) return;

    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_buffers_[task] = std::move(staging_buffer);
}

void AscendTcpTransport::releaseStagingBuffer(TransferTask *task) {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_buffers_.erase(task);
}

Status AscendTcpTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest> &entries) {
    if (entries.empty()) {
        return TcpTransport::submitTransfer(batch_id, entries);
    }

    auto &batch_desc = Transport::toBatchDesc(batch_id);
    const size_t first_task_id = batch_desc.task_list.size();
    std::vector<TransferRequest> staged_entries(entries.size());
    std::vector<std::pair<size_t, std::shared_ptr<void>>> staged_buffers;

    for (size_t i = 0; i < entries.size(); ++i) {
        std::shared_ptr<void> staging_buffer;
        auto status =
            stageTransferRequest(entries[i], staged_entries[i], staging_buffer);
        if (!status.ok()) return status;
        if (staging_buffer) {
            staged_buffers.emplace_back(i, std::move(staging_buffer));
        }
    }

    auto status = TcpTransport::submitTransfer(batch_id, staged_entries);
    if (!status.ok()) return status;

    for (auto &[entry_index, staging_buffer] : staged_buffers) {
        auto task_index = first_task_id + entry_index;
        if (task_index < batch_desc.task_list.size()) {
            retainStagingBuffer(&batch_desc.task_list[task_index],
                                std::move(staging_buffer));
        }
    }

    return Status::OK();
}

Status AscendTcpTransport::submitTransferTask(
    const std::vector<TransferTask *> &task_list) {
    if (task_list.empty()) {
        return TcpTransport::submitTransferTask(task_list);
    }

    std::vector<TransferRequest> staged_requests(task_list.size());
    std::vector<const TransferRequest *> original_requests(task_list.size(),
                                                           nullptr);
    std::vector<std::pair<TransferTask *, std::shared_ptr<void>>> staged_buffers;

    for (size_t i = 0; i < task_list.size(); ++i) {
        auto *task = task_list[i];
        if (!task || !task->request) {
            return Status::InvalidArgument(
                "AscendTcpTransport: transfer task or request is null");
        }

        original_requests[i] = task->request;

        std::shared_ptr<void> staging_buffer;
        auto status = stageTransferRequest(*task->request, staged_requests[i],
                                           staging_buffer);
        if (!status.ok()) {
            for (size_t j = 0; j < i; ++j) {
                task_list[j]->request = original_requests[j];
            }
            return status;
        }

        if (staging_buffer) {
            task->request = &staged_requests[i];
            staged_buffers.emplace_back(task, std::move(staging_buffer));
        }
    }

    for (auto &[task, staging_buffer] : staged_buffers) {
        retainStagingBuffer(task, std::move(staging_buffer));
    }

    auto status = TcpTransport::submitTransferTask(task_list);

    for (size_t i = 0; i < task_list.size(); ++i) {
        task_list[i]->request = original_requests[i];
    }

    if (!status.ok()) {
        for (auto *task : task_list) {
            releaseStagingBuffer(task);
        }
    }

    return status;
}

Status AscendTcpTransport::getTransferStatus(BatchID batch_id, size_t task_id,
                                             TransferStatus &status) {
    auto result = TcpTransport::getTransferStatus(batch_id, task_id, status);
    if (!result.ok()) return result;

    auto &batch_desc = Transport::toBatchDesc(batch_id);
    if (task_id < batch_desc.task_list.size() && isFinishedStatus(status.s)) {
        releaseStagingBuffer(&batch_desc.task_list[task_id]);
    }

    return Status::OK();
}

int AscendTcpTransport::registerLocalMemory(void *addr, size_t length,
                                            const std::string &location,
                                            bool remote_accessible,
                                            bool update_metadata) {
    (void)location;
    (void)remote_accessible;

    if (isAscendDeviceMemory(addr)) {
        return 0;
    }

    BufferDesc buffer_desc;
    buffer_desc.name = local_server_name_;
    buffer_desc.addr = reinterpret_cast<uint64_t>(addr);
    buffer_desc.length = length;
#ifdef ENABLE_MULTI_PROTOCOL
    buffer_desc.protocol = "tcp";
#endif
    return metadata_->addLocalMemoryBuffer(buffer_desc, update_metadata);
}

int AscendTcpTransport::unregisterLocalMemory(void *addr,
                                              bool update_metadata) {
    if (isAscendDeviceMemory(addr)) {
        return 0;
    }
    return metadata_->removeLocalMemoryBuffer(addr, update_metadata);
}

int AscendTcpTransport::registerLocalMemoryBatch(
    const std::vector<Transport::BufferEntry> &buffer_list,
    const std::string &location) {
    for (auto &buffer : buffer_list) {
        int ret = registerLocalMemory(buffer.addr, buffer.length, location, true,
                                      false);
        if (ret) return ret;
    }
    return metadata_->updateLocalSegmentDesc();
}

int AscendTcpTransport::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    for (auto &addr : addr_list) {
        int ret = unregisterLocalMemory(addr, false);
        if (ret) return ret;
    }
    return metadata_->updateLocalSegmentDesc();
}

}  // namespace mooncake
