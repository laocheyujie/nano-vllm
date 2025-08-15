import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        # 获取一个使用 "spawn" 方式来创建子进程的上下文环境
        # 在多进程编程中，有几种启动子进程的方式，主要有 fork, spawn, forkserver
        # fork (Unix 默认): 子进程几乎是父进程的一个完整拷贝，包括内存和资源。这种方式速度快，但如果父进程中包含像 CUDA 上下文或者文件句柄这类复杂的状态，直接拷贝可能会导致子进程出错或行为异常
        # spawn (Windows 和 macOS 默认): 这是更安全但稍慢的方式，会启动一个全新的、干净的 Python 解释器作为子进程，子进程不会继承父进程的内存空间，只会接收启动它所必需的参数
        # 在 PyTorch 中使用 CUDA 时，推荐使用 "spawn" 或 "forkserver"，可以避免很多由 fork 带来的 CUDA 初始化问题
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            # 创建一个事件对象，用于在主进程和子进程之间进行简单的同步
            event = ctx.Event()
            # 创建一个新的子进程来运行 ModelRunner 初始化
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        # 等待所有工作进程执行完毕
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 把 prompt 转换成 token_ids 并加到 scheduler 中的 waitting 列表
        # '<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
        if isinstance(prompt, str):
            # [151644, 872, 198, 396, 47845, 6133, 151645, 198, 151644, 77091, 198]
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # seqs: 当前 batch 中要执行的序列
        # is_prefill: 当前 batch 是否是 prefill
        seqs, is_prefill = self.scheduler.schedule()
        # token_ids: 当前 batch 中每个 sequence 的下一个 token 的 token_ids
        # len(token_ids): bs
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 1. 把 prompt 转换成 token_ids 并加到 scheduler 中的 waitting 列表
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            # 2. 执行当前 batch 的 prefill 和 decode
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
