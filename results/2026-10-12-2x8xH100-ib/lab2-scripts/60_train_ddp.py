#!/usr/bin/env python3
"""60_train_ddp.py - ResNet-50 DistributedDataParallel throughput on synthetic data.

Usage (via torchrun, see 60_train.sh):
    torchrun --nnodes N --nproc-per-node G ... 60_train_ddp.py [--steps 60] [--warmup 10] [--bs 128]

What it measures: images/second for the whole job (all ranks), after a warm-up. The point is
scaling (1 GPU -> 8 -> 16), not accuracy, so the data is random tensors made on the GPU: no
disk, no data-loader, nothing but compute and the gradient all-reduce.
Each rank prints nothing except rank 0, which prints one JSON line the collector greps.
"""
import argparse, json, os, time
import torch, torch.distributed as dist, torchvision

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=60)    # timed steps
ap.add_argument("--warmup", type=int, default=10)   # untimed steps (cuDNN autotune, NCCL init)
ap.add_argument("--bs", type=int, default=128)      # per-GPU batch
a = ap.parse_args()

rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
local = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local)
if world > 1:
    dist.init_process_group("nccl")                  # NCCL picks IB/NVLink on its own

torch.backends.cudnn.benchmark = True
model = torchvision.models.resnet50().cuda().to(memory_format=torch.channels_last)
if world > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
x = torch.randn(a.bs, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
y = torch.randint(0, 1000, (a.bs,), device="cuda")

def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):   # bf16: what an H100 is used for
        loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()

for _ in range(a.warmup):
    step()
torch.cuda.synchronize()
if world > 1:
    dist.barrier()
t0 = time.time()
for _ in range(a.steps):
    step()
torch.cuda.synchronize()
dt = time.time() - t0
if rank == 0:
    ips = a.steps * a.bs * world / dt
    print("RESULT " + json.dumps({"world_size": world, "per_gpu_batch": a.bs, "steps": a.steps,
          "seconds": round(dt, 2), "images_per_sec": round(ips, 1),
          "images_per_sec_per_gpu": round(ips / world, 1),
          "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version()))}))
if world > 1:
    dist.destroy_process_group()
