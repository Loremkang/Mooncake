// Copyright 2026
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

#ifndef ASCEND_TCP_TRANSPORT_H_
#define ASCEND_TCP_TRANSPORT_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "transport/tcp_transport/tcp_transport.h"

namespace mooncake {

using TransferRequest = Transport::TransferRequest;
using TransferStatus = Transport::TransferStatus;
using TransferStatusEnum = Transport::TransferStatusEnum;
using SegmentID = Transport::SegmentID;
using BatchID = Transport::BatchID;

class AscendTcpTransport : public TcpTransport {
   public:
    AscendTcpTransport();
    ~AscendTcpTransport() override;

    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest> &entries) override;

    Status submitTransferTask(
        const std::vector<TransferTask *> &task_list) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus &status) override;

   private:
    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location, bool remote_accessible,
                            bool update_metadata) override;

    int unregisterLocalMemory(void *addr,
                              bool update_metadata = true) override;

    int registerLocalMemoryBatch(
        const std::vector<Transport::BufferEntry> &buffer_list,
        const std::string &location) override;

    int unregisterLocalMemoryBatch(
        const std::vector<void *> &addr_list) override;

    const char *getName() const override { return "ascend_tcp"; }

    int checkAndCreateStreamCopy();
    int activateAclContext();
    Status copyDeviceToHost(void *host_dst, const void *device_src,
                            size_t length);
    Status stageTransferRequest(const TransferRequest &request,
                                TransferRequest &staged_request,
                                std::shared_ptr<void> &staging_buffer);
    void retainStagingBuffer(TransferTask *task,
                             std::shared_ptr<void> staging_buffer);
    void releaseStagingBuffer(TransferTask *task);

   private:
    aclrtStream stream_copy_{};
    aclrtContext context_{};
    bool stream_copy_created_{};
    bool context_initialized_{};
    bool logic_device_initialized_{};
    int logic_device_id_{};
    std::mutex copy_mutex_;

    std::mutex staging_mutex_;
    std::unordered_map<TransferTask *, std::shared_ptr<void>> staging_buffers_;
};

}  // namespace mooncake

#endif  // ASCEND_TCP_TRANSPORT_H_
