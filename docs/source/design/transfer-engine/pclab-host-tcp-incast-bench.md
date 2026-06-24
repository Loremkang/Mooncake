# PCLab Host TCP Incast Benchmark 设计记录

## 背景

`pclab_host_tcp_incast_bench` 是为 PCLab 910C-H20 异构 PD 分离验证准备的专用 Mooncake Transfer Engine 微基准。它的第一阶段目标不是模拟完整 KV cache 语义，而是在没有 RDMA 的 910C-H20 中间网络上，验证多台 pca initiator 到一台 pcn target 的 host DRAM TCP incast 路径。

当前路径为：

```text
pca initiator source DRAM -> Mooncake TCP -> pcn target DRAM
```

该工具只安装 `tcp` transport，使用 `P2PHANDSHAKE` 打开 target segment。后续 910C NPU HBM 到 host DRAM、pcn host DRAM 到 H20 CUDA 的搬运，应在此 host TCP incast 验证稳定后再接入。

## 角色

- target：运行在 pcn 侧，分配并注册一组 target host DRAM buffer，等待 initiator 写入或读取。
- initiator：运行在 pca 侧，分配并注册一组 source host DRAM buffer，启动多个 worker thread，持续提交 batch transfer。

target 进程本身不创建 benchmark worker thread。target 侧真正接收 TCP 并写入 registered memory 的工作由 Transfer Engine transport 内部完成。

## 拓扑与线程参数

工具刻意拆分 TE 数量和单个 TE 内部 worker 数，不保留 `--threads` 兼容别名。旧的 `--initiator_threads`、`--expected_threads_per_initiator`、`--rank`、`--world_size`、`--segment_id` 仅作为 deprecated alias 保留。

| 参数 | 使用侧 | 含义 |
| --- | --- | --- |
| `--source_world_size` | initiator/target | source TE 进程总数。 |
| `--source_rank` | initiator | 当前 source TE rank。 |
| `--target_world_size` | initiator/target | target TE 进程总数。 |
| `--target_rank` | target | 当前 target TE rank。 |
| `--target_segments` | initiator | 按 target rank 排列的逗号分隔 target session id 列表。 |
| `--threads_per_source_te` | initiator | 单个 source TE 进程内部 worker 数，决定并发数、source offset 和 target offset。 |
| `--expected_workers_per_source_te` | target | target 侧预期的每个 source TE worker 数，只用于容量预检查和 `--verify_on_exit` 验证布局。 |
| `--target_selection` | initiator/target | target 映射策略，当前支持 `mod` 和写入压测用 `all_to_all`。 |

推荐在 M1 阶段让所有 source TE 使用相同的 `--threads_per_source_te`，并让 target 的 `--expected_workers_per_source_te` 与它一致。

如果二者不一致：

- 传输路径本身由 initiator 的 `--threads_per_source_te` 计算 offset。
- target 的容量预检查只覆盖 `--expected_workers_per_source_te` 描述的布局。
- `--verify_on_exit=true` 时，target 只按 `--expected_workers_per_source_te` 验证，因此不一致会导致漏验或误报。

后续如果需要不同 source rank 使用不同 worker 数，应增加 per-rank 配置，例如 `--expected_workers_by_source=4,4,8,8`，而不是重新复用单个全局 worker 参数。

## Buffer 数量

`--buffers` 表示当前进程本地注册 buffer 数。`--buffers=0` 时使用本机 NUMA node 数。

- initiator 侧：`--buffers` 对应 `source_buffer_count`。
- target 侧：`--buffers` 对应 `target_buffer_count`。

这两个数量可以不同。典型场景是 pca 侧按 pca NUMA 分配 source buffer，pcn 侧按 pcn NUMA 分配 target buffer。

## Source Buffer 布局

每个 initiator worker 使用一个 source buffer：

```text
source_buffer = source_buffers[thread_id % source_buffer_count]
```

同一个 source buffer 内，不同 thread 和 batch request 使用不重叠 slot：

```text
source_offset =
    block_size * (
      (request_index * selected_target_count + target_ordinal)
      * threads_per_source_te + thread_id
    )
```

其中：

- `thread_id` 属于 `[0, threads_per_source_te)`。
- `target_ordinal` 属于当前 source TE 实际写入的 target 列表。
- `request_index` 属于 `[0, batch_size)`。
- 每个 slot 大小为 `block_size`。

因此 initiator 每个 source buffer 至少需要覆盖：

```text
source_span =
    block_size * batch_size * threads_per_source_te
    * selected_target_count
```

当前实现中，如果 `--buffer_size < source_span`，initiator 会把 `buffer_size` 自动调大到 `source_span`。

## Target Buffer 布局

target segment 可以暴露多个 target buffer。initiator 按 thread id 选择远端 target buffer：

```text
target_buffer_index = thread_id % target_buffer_count
```

每个 target buffer 内部先按会写入该 target TE 的 source ranks 切分，避免多个 source TE 互相覆盖：

```text
source_region_size = target_buffer.length / source_regions_for_target
source_base = target_buffer.addr + source_local_index * source_region_size
```

source region 内再按落到该 target buffer 的 worker lane 切分：

```text
threads_in_target_buffer =
    ceil((thread_count - target_buffer_index) / target_buffer_count)

thread_ordinal = thread_id / target_buffer_count

target_offset_in_rank_region =
    block_size * (request_index * threads_in_target_buffer + thread_ordinal)
```

代码中使用整数形式计算：

```text
threads_in_target_buffer =
    (thread_count + target_buffer_count - 1 - target_buffer_index)
    / target_buffer_count
```

initiator 传输时，`thread_count` 取 `--threads_per_source_te`。target 验证时，`thread_count` 取 `--expected_workers_per_source_te`。

## 多打一映射示例

假设：

```text
source_world_size = 8
target_world_size = 2
target_selection = mod
target_buffer_count = 2
threads_per_source_te = 4
batch_size = N
```

单个 source TE 内的 worker 映射为：

| worker | target buffer | source region 内 lane |
| --- | --- | --- |
| 0 | buffer 0 | lane 0 |
| 1 | buffer 1 | lane 0 |
| 2 | buffer 0 | lane 1 |
| 3 | buffer 1 | lane 1 |

不同 pca source TE 必须使用不同 `--source_rank`。在 `mod` 映射下：

```text
pca000 -> source_rank 0 -> target_rank 0
pca001 -> source_rank 1 -> target_rank 1
pca002 -> source_rank 2 -> target_rank 0
pca003 -> source_rank 3 -> target_rank 1
...
pca007 -> source_rank 7 -> target_rank 1
```

这样在每个 target buffer 内，source ranks 之间由 `source_region_size` 隔开，source rank 内 worker/request 又由 lane slot 隔开。

## 容量约束

target 侧每个 target buffer 的每个 source region 至少需要：

```text
target_span_for_buffer =
    block_size * batch_size * threads_in_target_buffer
```

因此必须满足：

```text
buffer_size / source_regions_for_target >= target_span_for_buffer
```

initiator 打开 target segment 后，也会基于自己的 `--threads_per_source_te` 对远端 buffer 做同样检查。这样即使 target 的 `--expected_workers_per_source_te` 设置过小，initiator 也会在自身需要的 source region 不够时失败，而不是提交越界 remote offset。

## 数据校验

`--operation=write` 且 `--init_pattern=true` 时，initiator 会在 source buffer 中写入确定性 pattern。pattern 输入包括：

```text
source_rank, worker_id, request_index, byte_index
```

target 收到退出信号后，如果设置 `--verify_on_exit=true`，会按 `source_world_size`、`target_world_size`、`target_selection`、`expected_workers_per_source_te`、`batch_size` 和 target buffer 布局逐 slot 验证。

验证的前提是：

- 所有 source TE 使用唯一且正确的 `--source_rank`。
- 所有 source TE 使用相同 `--threads_per_source_te`。
- target 的 `--expected_workers_per_source_te` 等于 initiator 的 `--threads_per_source_te`。
- 测试期间每个 slot 最终内容来自对应 source rank/worker/request 的最后一次 WRITE。

## 示例命令

target：

```bash
./pclab_host_tcp_incast_bench \
    --mode=target \
    --metadata_server=P2PHANDSHAKE \
    --local_server_name=<target-host-ip> \
    --source_world_size=8 \
    --target_world_size=2 \
    --target_rank=<0-or-1> \
    --target_selection=mod \
    --buffers=2 \
    --buffer_size=268435456 \
    --block_size=65536 \
    --batch_size=64 \
    --expected_workers_per_source_te=4 \
    --verify_on_exit=true
```

initiator：

```bash
./pclab_host_tcp_incast_bench \
    --mode=initiator \
    --metadata_server=P2PHANDSHAKE \
    --local_server_name=<source-host-ip> \
    --target_segments=<target0-host:rpc-port>,<target1-host:rpc-port> \
    --source_rank=<rank> \
    --source_world_size=8 \
    --target_world_size=2 \
    --target_selection=mod \
    --buffers=4 \
    --buffer_size=268435456 \
    --block_size=65536 \
    --batch_size=64 \
    --threads_per_source_te=4 \
    --duration=10 \
    --operation=write
```

## 当前限制

- 该工具只覆盖 host DRAM TCP incast，不覆盖 Ascend HBM、CUDA DRAM/VRAM 或完整 KV cache block 元数据。
- `--expected_workers_per_source_te` 是全局值，尚不支持不同 source rank 使用不同 worker 数。
- `--buffers` 是本地进程语义，initiator 和 target 可分别设置，但同一进程内 source/target buffer 数没有进一步拆成不同 flag。
- target verify 只验证确定性 pattern，不验证吞吐、延迟分布或乱序完成行为。
- 如果多个 source TE 复用同一个 `--source_rank`，它们会写入同一 source region，结果属于配置错误。
