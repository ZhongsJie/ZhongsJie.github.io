---
title: "vLLM 分布式通信"
date: 2026-04-28T11:03:49+08:00
draft: false
subtitle: ""
author: "ZhongsJie"
tags:
  - vllm
  - ai
  - inference
categories: []
showToc: false
TocOpen: false
disableShare: true
cover:
    image: ""
    caption: ""
    alt: ""
    relative: false
---

```python
class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None
	@instrument(span_name="Worker init")
    def __init__(...):
	    self.rank = rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        ...
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper
        ...
        self.worker.init_device()
	    if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            self.worker.load_model()
        。。
```

## Worker `init_device`

>worker的init_device函数负责初始化设备的相关信息

1. **根据当前worker的rank找到它所属的device**，将它绑到指定的卡上，以及清空该device的显存，获取该device的显存大小等
2. 对当前的worker做分布式环境初始化，也就是初始化当前worker的各种**进程组**（如模型并行、流水线并行、数据并行等）
3. 构造当前worker的GPUModelRunner对象。维护着模型权重分片，还维护模型运行过程中所需要的一些数据结构，比如**kv cache**等，**负责模型权重的加载(load_model)，以及实际的推理执行过程等逻辑。**

```python
@instrument(span_name="Init device")
def init_device(self):
	if self.device_config.device_type == "cuda":
		...
		# Ray 会设置 NCCL_ASYNC_ERROR_HANDLING，但这个环境变量会导致 CUDA graph
		# 构建时出现异常。CUDA graph 需要同步执行，而该变量会引入异步错误处理，
		# 两者可能引起冲突。
		os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
		...
		# - Ray/external_launcher 场景：这些分布式执行器自己管理GPU映射
        # - 多节点场景(nnodes_within_dp > 1)：每个节点有独立的GPU集合，映射逻辑不同
        # - Ray作为DP后端：Ray的resource pool处理GPU分配
		if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1 # 单节点场景
            ):
            # local DP rank 表示在当前节点内的数据并行编号，而 global rank 可能
            # 跨节点。在单节点场景下，GPU映射只看节点内的local rank。
            dp_local_rank = self.parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = self.parallel_config.data_parallel_index
            # DP副本 0 (dp_local_rank=0)
            # original_local_rank=0 → GPU 0 + 0×2 = GPU 0
            # original_local_rank=0 → GPU 0 + 1×2 = GPU 2 ← 偏移
            # 偏移是为了计算出实际的rank信息，后续用来初始化device
            self.local_rank += dp_local_rank * tp_pp_world_size
            。。
            
		self.device = torch.device(f"cuda:{self.local_rank}")
		# PyTorch的设备API演进：从 torch.cuda.set_device() 到
        # torch.accelerator.set_device_index()。 后续可以不用再手动指定，
		torch.accelerator.set_device_index(self.device)
		...
		# 初始化分布式推理所需的所有环境，包括通信组信息
		# 优先于内存快照处理逻辑，NCCL在初始化会分配内部缓存区，提前初始化用于保证显存计算的准确性
		init_worker_distributed_environment(
			self.vllm_config,
			self.rank,
			self.distributed_init_method,
			self.local_rank,
			current_platform.dist_backend,
		)
		# 1. gc.collect() 回收Python层的垃圾对象，释放可能的GPU引用
        # 2. empty_cache() 释放PyTorch缓存的显存（包括NCCL缓冲区）
        # 得到一个"干净"的显存状态，作为基准线
        gc.collect()
        torch.accelerator.empty_cache()
        # 用于后续计算 KV cache 可用显存。
        # init_snapshot 记录当前可用显存，request_memory 根据模型配置
        # 计算需要预留的 KV cache 大小。
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
    
	# 最后初始化modelrunner，需要依赖设备、分布式通信环境，同时通过快照可以计算出可以分配的kv cache显存大小
	if self.use_v2_model_runner:
		...
		self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
			self.vllm_config, self.device
		)
	else:
		...
		self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
	...
```


## `init_worker_distributed_environment`

> 负责初始化分布式推理所需的所有环境组件

```python
def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    attention_config = vllm_config.attention_config
    parallel_config = vllm_config.parallel_config

    # Batch invariance 确保模型输出与 batch size 和请求顺序无关。
    # 用于保证在不同批次相同输入的情况下，有相同的输出，可以应用于以下场景：
    #   - 框架调试：相同输入在不同 batch 配置下产生相同输出
    #   - 模型调试：跨不同 batch 配置验证模型行为一致性
    #   - 强化学习 (RL)：可重现的 rollout 对训练稳定性至关重要
    #   - 大规模推理系统：测试、验证和一致性保证
    #
    # Batch invariance 使用特殊的注意力内核，让不同 batch size
    # 共享相同的计算内核和 CUDA graph，从而消除 batch size 变化带来的数值差异
    # 目前应该只有Nvidia支持
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    init_batch_invariance(attention_config.backend)
    
	...
    override_envs_for_eplb(parallel_config)
    # 自定义all-reduce是一种通信优化，在某些场景下可以加速集合通信。
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # "env://" 是PyTorch推荐的分布式初始化方式，通过环境变量传递连接信息
    init_method = distributed_init_method or "env://"

    # 设置超时可以避免无限等待，加速故障检测和恢复。
    timeout = None
    if parallel_config.distributed_timeout_seconds is not None:
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)

    # ========== 核心：初始化分布式环境 ==========
    # 初始化PyTorch的world group，这是所有其他并行组(TP/PP/DP)的基础。
    # 没有world group，就无法建立进程间的通信连接。 ！！！
    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        backend,
        timeout,
    )

    # ========== 初始化模型并行组 (TP/PP/DP/CP) ==========
    # World group 只是建立了"所有进程都能互相通信"的基础设施。
    # 但在实际推理中，不同的进程组需要执行不同的协作：
    # - TP组内需要all-reduce同步隐藏状态
    # - PP组内需要broadcast传递layers
    # - DP组内需要all-gather收集输出
    # 这些都是通过不同的进程组实现的。
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
        parallel_config.prefetch_context_parallel_size,
    )

	...
    ensure_ec_transfer_initialized(vllm_config)
```

初始化过程主要分两个步骤：
1. 调用`init_distributed_environment`函数初始化分布式环境
2. 调用`ensure_model_parallel_initialized`函数初始化模型并行（比如张量并行、数据并行、流水行并行）的进程组

### `env://`

> `env://` ：**避免写死配置或依赖外部服务，通过环境变量统一注入这些信息**

在使用过程中需要包括以下变量
- MASTER_ADDR=主节点IP
- MASTER_PORT=端口
- WORLD_SIZE=总进程数
- RANK=当前进程编号
实际处理逻辑
1. 从环境变量读取：
    - MASTER_ADDR / PORT
    - RANK / WORLD_SIZE
2. 构造一个 TCP rendezvous 地址：`tcp://MASTER_ADDR:MASTER_PORT`
	- rendezvous = 初始化 process group 之前的**成员发现（peer discovery）和信息交换机制**
3. 所有进程连接到这个地址完成初始化
4. 建立通信组（process group）

## `init_distributed_environment`

> vLLM中的分布式主要在 vllm/distributed/parallel_state.py 文件中实现，它抽象了 PyTorch 原生的分布式操作，并提供了针对不同并行策略（张量并行、流水线并行、数据并行）的特定通信组和接口。

```python
def init_distributed_environment(
    world_size: int = -1, # TP*PP，用于创建 worker 进程数（影响进程数量）
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: timedelta | None = None,
):
	...
    if (
        config is not None
        and config.parallel_config.distributed_executor_backend != "external_launcher"
        and (
            config.parallel_config.nnodes > 1
            or config.parallel_config.data_parallel_size > 1
        )
        and not enable_elastic_ep
    ):
	    ...
	    # 和之前的逻辑一致，计算偏移量之后获取到全局的rank信息
	    rank = parallel_config.data_parallel_rank * world_size + rank
	    # 单节点：所有进程在同一台机器上，可以通过共享内存高效通信
        # 多节点：需要通过网络TCP通信，需要知道master节点的地址和端口
        if parallel_config.nnodes > 1:
            # 多节点：使用master_addr和master_port
            # 这些通常是rank 0所在节点的地址
            ip = parallel_config.master_addr
            port = parallel_config.master_port
            distributed_init_method = get_distributed_init_method(ip, port)
        else:
            # 单节点数据并行：使用data_parallel_master_ip
            # 为了高效，DP组内的通信使用共享内存而不是TCP
            ip = parallel_config.data_parallel_master_ip
            port = parallel_config.get_next_dp_init_port()
            distributed_init_method = get_distributed_init_method(ip, port)
    if not torch.distributed.is_initialized():
	    if not torch.distributed.is_backend_available(backend):
			# 如果不支持backend会回退到gloo    
		    backend = "gloo"
		# pytorch分布式模块的入口点和核心初始化步骤
		# 1. 建立进程间的通信通道（NCCL/Goo）
	    # 2. 确定每个进程的rank（全局唯一标识）
	    # 3. 设置同步原语（barrier等）
	    # 之后每个进程都知道自己是谁，以及如何与其他进程通信。
		torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
    global _WORLD, _NODE_COUNT, _INNER_DP_WORLD
    ...
	    _WORLD = init_world_group(ranks, local_rank, backend)
    
```

调用torch.distributed.init_process_group，这个函数**是pytorch分布式模块的入口点和核心初始化步骤，在执行后续的分布式操作之前必须调用此函数，这是所有后续分布式操作（包括创建子ProcessGroup）的前提**
- 通过world_size规定了参与分布式作业的总进程数，会隐式地创建一个包含所有 world_size 个进程的默认进程组，即WORLD组，在此之后，任何不指定 group 参数的 torch.distributed 通信操作（如 `torch.distributed.all_reduce(tensor)`）都会默认在这个全局 WORLD 组上执行。
- 并且根据指定的 backend（例如 "nccl", "gloo", "mpi"），初始化相应的通信库；
- 通过 init_method（例如 `env://` 或 `tcp://address:port`），使得所有参与分布式作业的进程能够建立通信连接；

vLLM的 parallel_state 模块通过 GroupCoordinator 来统一管理不同类型的并行组（全局、张量并行、流水线并行等），为了这种统一性，即使是全局的WORLD组，也需要一个GroupCoordinator 实例来封装它，因此紧接着**调用了init_world_group（内部是GroupCoordinator 的构造函数）来为`_WORLD`变量构造一个`GroupCoordinator`实例**。

## GroupCoordinator
> GroupCoordinator是 vLLM 中 分布式并行通信的核心抽象层， PyTorch 的 ProcessGroup 是底层的分布式通信接口，但使用起来比较繁琐。 

GroupCoordinator 封装了ProcessGroup：
- 更高层次的抽象 - 封装了设备管理、通信操作、rank 计算等
	- `all_reduce`, `all_gather`等
- 统一 CPU 和 GPU 通信 - 同时管理 cpu_group （用于元数据传输）和 device_group （用于张量通信）
- 平台无关性 - 隐藏了 NCCL/Gloo 等后端的差异

```python
rank: int              # 全局 rank
ranks: list[int]       # 组内所有进程的全局 rank
world_size: int        # 组内进程数
local_rank: int        # 本地设备编号（用于绑定 GPU）
rank_in_group: int     # 在组内的编号（从 0 开始）
cpu_group: ProcessGroup  # CPU 通信组（gloo 后端）
device_group: ProcessGroup  # GPU 通信组（nccl 后端）
device_communicator: DeviceCommunicatorBase  # 设备通信器封装
mq_broadcaster: Any | None # 一个可选的基于共享内存的消息队列广播器，用于高效广播对象（特别是当源 rank 固定时）。
```


```python
def __init__(...):
	for ranks in group_ranks:
		device_group = torch.distributed.new_group(
			ranks, backend=torch_distributed_backend
		)
		# a group with `gloo` backend, to allow direct coordination between
		# processes through the CPU.
		with suppress_stdout():
			cpu_group = torch.distributed.new_group(ranks, backend="gloo")
```
传入`GroupCoordinator`的 group_ranks 参数实际上就是 `[[0, 1, ..., world_size-1]]`，即包含了在 `init_process_group` 中定义的所有进程的列表。然后调用 `torch.distributed.new_group`创建了两个显式的ProcessGroup对象，device_group用于GPU通信，cpu_group用于CPU通信。
**vLLM 的选择**：
- **GPU 通信**：NCCL（大张量通信）
- **CPU 通信**：Gloo（元数据、小数据）

### 集合通信操作

| 方法 | 作用 | 使用场景 |
|------|------|----------|
| `all_reduce(input_)` | 所有进程同时执行归约操作，结果分发给所有进程 | TP内同步隐藏状态 |
| `all_gather(input_, dim)` | 收集所有进程的张量并拼接 | DP内收集输出 |
| `reduce_scatter(input_, dim)` | 先归约再分散到各进程 | 负载均衡 |
| `gather(input_, dst, dim)` | 收集到指定进程 | PP内收集layers |
| `broadcast(input_, src)` | 从 src 广播张量到所有进程 | PP内广播layers |

### 对象通信操作

| 方法                           | 作用                          | 使用场景     |
| ---------------------------- | --------------------------- | -------- |
| `broadcast_object(obj, src)` | 广播 Python 对象（通过 pickle 序列化） | 广播配置     |
| `send_object(obj, dst)`      | 发送 Python 对象到指定进程           | 点对点传输元数据 |
| `recv_object(src)`           | 从指定进程接收 Python 对象           | 点对点接收元数据 |

### 张量字典通信

```python
broadcast_tensor_dict(tensor_dict, src)  # 广播包含任意类型值的字典
send_tensor_dict(tensor_dict, dst)         # 发送字典
```

这是 vLLM 中非常常用的功能，因为模型权重、KV cache 等都以字典形式存储。

## `ensure_model_parallel_initialized`

> 通信组的初始化逻辑

```python
def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
	...
	# 要求PyTorch 的分布式环境（即 WORLD 组）已经被初始化
	if not model_parallel_is_initialized():
		initialize_model_parallel(
			tensor_model_parallel_size,
			pipeline_model_parallel_size,
			prefill_context_model_parallel_size,
			decode_context_model_parallel_size,
			backend,
		)
	...
```
调用`initialize_model_parallel`，用于根据用户指定的 tensor_model_parallel_size 和 pipeline_model_parallel_size (以及从配置中获取的 data_parallel_size) 来划分和创建**全局变量**中的 `_TP`、`_PP` 和 `_DP` 组等。

### `initialize_model_parallel`

```python
def initialize_model_parallel(...):
    # Step 1: 构建多维 rank 数组
    # shape: (ExternalDP, DP, PP, PCP, TP)
    # ExternalDP=1, DP=2, PP=2, PCP=1, TP=2（EP=DP*PCP*TP=4)
    # shape = (1, 2, 2, 1, 2)
    all_ranks = torch.arange(world_size).reshape(
	    -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )

    # Step 2: 创建 TP 组（最内层，通信最频繁）
    # TP组内的通信非常频繁（如每层的all-reduce）。
    # MessageQueue通过共享内存实现对象广播，比NCCL更快。`broadcast_object`
    # 它是单writer多reader的，适合TP场景。
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    ...
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,  # TP组需要MQ广播对象
        group_name="tp",
    )

    # Step 3: 创建 PCP/DCP 组（Context Parallel）之前提过一次CP的概念， DCP是复用的TP的GPU资源，将TP组进行细分
    _PCP = init_model_parallel_group(..., group_name="pcp")
    _DCP = init_model_parallel_group(..., group_name="dcp")

    # Step 4: 创建 PP 组
    # 将PP维度移到最后一维，然后展平。
    # 这样每个PP组内的rank在不同TP/PCP间是对角线分布的。
    # 例如：PP=2时，g0,g2,g4,g6在一个PP组，g1,g3,g5,g7在另一个PP组
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )

     # Step 5: 创建 DP 组（最外层，通信最少）
    # 将DP维度移到最后一维，然后展平。
    # 这样每个DP组内的rank在不同TP/PP/PCP间是均匀分布的。
    # 同一DP组内的所有GPU处理不同的数据，但模型相同。
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    _DP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="dp"
    )

    # Step 6: 创建 EP 组（仅 MoE 模型）
    # MoE (Mixture of Experts) 模型有多个专家(expert)，EP允许不同的
    # GPU处理不同的专家。这样可以扩展专家数量而不增加单GPU内存。
    if config.model_config.is_moe:
        group_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(-1, data_parallel_size * prefill_context_parallel_size * tensor_model_parallel_size)
            .unbind(0)
        )
        _EP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="ep"
        )

        # EPLB (EP Load Balancer) 是一种动态负载均衡机制。
        # MoE的专家使用不均匀时，需要重新分配负载。
        # EPLB使用独立的通信组，避免与MoE的集合通信产生死锁。
        if config.parallel_config.enable_eplb:
            _EPLB = init_model_parallel_group(
                group_ranks, get_world_group().local_rank, backend, group_name="eplb"
            )
```
`initialize_model_parallel` 是 vLLM 分布式并行的**核心初始化函数**，负责：
1. **构建 all_ranks 多维数组** - 按照 `(ExternalDP, DP, PP, PCP, TP)` 的顺序组织所有 rank
2. **创建各种进程组** - 通过 `torch.distributed.new_group()` 创建 TP/PP/DP/PCP/EP 组
3. **绑定到 GroupCoordinator** - 封装 ProcessGroup，提供高层次通信 API

布局顺序为：`ExternalDP x DP x PP x PCP x TP`
- **ExternalDP**：外部数据并行组（用于verl集成），不同 ExternalDP 组之间无需任何通信
- **DP**：数据并行组（参与模型生成）
- **PP**：流水线并行组
- **PCP**：`Prefill Context Parallel`组
- **TP**：张量并行组
定义了并行维度之间的嵌套关系，最内层的是TP，最外层的就是ExternalDP，它**定义了不同并行维度下哪些 rank 属于同一个子组。例如，在 TP 维度上相邻的 rank 会被视为同一个张量并行组的成员**。布局顺序的核心设计原则是**通信频率递增**——通信越频繁的并行维度放在越内层


```
all_ranks = [
    [  // ExternalDP = 0
        [  // DP = 0
            [  // PP = 0
                [0, 1]     // PCP = 0, TP = [0, 1]
            ],
            [  // PP = 1
                [2, 3]     // PCP = 0, TP = [0, 1]
            ]
        ],
        [  // DP = 1
            [  // PP = 0
                [4, 5]     // PCP = 0, TP = [0, 1]
            ],
            [  // PP = 1
                [6, 7]     // PCP = 0, TP = [0, 1]
            ]
        ]
    ]
]
```

`group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)`：view将四维张量重新展平成二维，得到的形状为 (N, tensor_model_parallel_size) ，**其中每一行代表一个完整的张量并行组所需的所有 rank**，一共有N个张量并行组。
```
tp_group_ranks = [
    [0, 1],     // ExternalDP=0, DP=0, PP=0, PCP=0
    [2, 3],     // ExternalDP=0, DP=0, PP=1, PCP=0
    [4, 5],     // ExternalDP=0, DP=1, PP=0, PCP=0
    [6, 7]      // ExternalDP=0, DP=1, PP=1, PCP=0
]

GPU0: TP索引=0，拥有数据分片 A
GPU2: TP索引=1，拥有数据分片 B
组合成一个PP组可以互补
pp_group_ranks = [
    [0, 2],     // ExternalDP=0, DP=0, PCP=0, TP索引分别是 0 和 1 互补
    [1, 3],     // ExternalDP=0, DP=0, PCP=0, ...
    [4, 6],     // ExternalDP=0, DP=1, PCP=0, ...
    [5, 7]      // ExternalDP=0, DP=1, PCP=0, ...
]

DP 组内的 GPU 拥有相同的模型（相同的 TP/PP 分片），但处理不同的输入数据。
dp_group_ranks = [
    [0, 4],     // ExternalDP=0, PP=0, PCP=0, TP=[0,1]
    [1, 5],     // ExternalDP=0, PP=0, PCP=0, TP=[1,0]
    [2, 6],     // ExternalDP=0, PP=1, PCP=0, TP=[0,1]
    [3, 7]      // ExternalDP=0, PP=1, PCP=0, TP=[1,0]
]

ep_group_ranks = [
    [0, 1, 2, 3],   // EP组0: DP=0,DP=1 的所有 TP 成员，跨 PP=0
    [4, 5, 6, 7]    // EP组1: DP=0,DP=1 的所有 TP 成员，跨 PP=1
]
```

## `init_ascend_model_parallel(self.parallel_config)`

> Ascend提供一个细粒度的并行策略

```python
def _create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator:
	if group_size is None:
		return None
	if group_size not in _group_cache:
		# 构建 rank_grid: (PP, DP, TP)  这里感觉是没有支持ExternalDP和PCP，因此没有考虑这个维度数据?
		# shape (2, 2, 2)
		rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
		# 将 DP 维度切分成多个 chunks，每个 chunk 包含 group_size 个连续的 DP 实例
		# group_size 必须能整除 DP，按照DP处理，不同的输入数据都在处理
		num_chunks = global_dp_size // group_size
		group_ranks = []
		# 遍历: PP维度 → DP分块 → TP维度 
		for pp_idx in range(global_pp_size):
			stage_ranks = rank_grid[pp_idx]  # (dp, tp)
			for chunk in range(num_chunks):
				for tp_idx in range(global_tp_size):
					group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
					group_ranks.append(group)
		pg = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=group_name)
		_group_cache[group_size] = pg

	return _group_cache[group_size]

otp_size = get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size
lmhead_tp_size = get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size
embedding_tp_size = get_ascend_config().finegrained_tp_config.embedding_tensor_parallel_size
mlp_tp_size = get_ascend_config().finegrained_tp_config.mlp_tensor_parallel_size

global _OTP, _LMTP, _EMBED_TP, _MLP_TP

if otp_size > 0:
	_OTP = _create_or_get_group(otp_size, "otp")
if lmhead_tp_size > 0:
	_LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")
if embedding_tp_size > 0:
	_EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
if mlp_tp_size > 0:
	_MLP_TP = _create_or_get_group(mlp_tp_size, "mlptp")
```

num_chunks = DP / group_size = 2 / 2 = 1
遍历过程:
```python
PP=0, chunk=0, TP=0 -> [0, 2]
PP=0, chunk=0, TP=1 -> [1, 3]
PP=1, chunk=0, TP=0 -> [4, 6]
PP=1, chunk=0, TP=1 -> [5, 7]
```


## 参考
1. https://zhuanlan.zhihu.com/p/1912989074316297299
