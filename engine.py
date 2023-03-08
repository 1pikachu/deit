# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import os
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import time


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    if args.jit:
        try:
            img = None
            for image, target in data_loader:
                img = image
                break
            img = img.to(device)
            model = torch.jit.trace(model, img, check_trace=False)
            for i in range(3):
                model(img)
            print("[INFO] JIT enabled.")
        except:
            print("[WARN] JIT disabled.")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    total_time = 0.0
    total_sample = 0
    i = 0

    profile_len = min(len(data_loader), args.num_iter) // 2
    if args.profile and args.device == "xpu":
        for images, target in data_loader:
            if i >= args.num_iter:
                break
            i += 1
            
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            batch_size = images.shape[0]
            # compute output
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                elapsed = time.time()
                images = images.to(device, non_blocking=True)
                output = model(images)
                torch.xpu.synchronize()
                elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec, batchSize: {}".format(i, elapsed, batch_size), flush=True)

            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == profile_len:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
                torch.save(prof.table(sort_by="id", row_limit=100000),
                    timeline_dir+'profile_detail_withId.pt')
                prof.export_chrome_trace(timeline_dir+"trace.json")
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for images, target in data_loader:
                if i >= args.num_iter:
                    break
                i += 1
                
                if args.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                batch_size = images.shape[0]
                # compute output
                elapsed = time.time()
                images = images.to(device, non_blocking=True)
                with torch.jit.fuser(fuser_mode):
                    output = model(images)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec, batchSize: {}".format(i, elapsed, batch_size), flush=True)

                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for images, target in data_loader:
                if i >= args.num_iter:
                    break
                i += 1
                
                if args.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                batch_size = images.shape[0]
                # compute output
                elapsed = time.time()
                images = images.to(device, non_blocking=True)
                output = model(images)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec, batchSize: {}".format(i, elapsed, batch_size), flush=True)

                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for images, target in data_loader:
            if i >= args.num_iter:
                break
            i += 1
            
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            batch_size = images.shape[0]
            # compute output
            elapsed = time.time()
            images = images.to(device, non_blocking=True)
            with torch.jit.fuser(fuser_mode):
                output = model(images)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec, batchSize: {}".format(i, elapsed, batch_size), flush=True)

            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for images, target in data_loader:
            if i >= args.num_iter:
                break
            i += 1
            
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            batch_size = images.shape[0]
            # compute output
            elapsed = time.time()
            images = images.to(device, non_blocking=True)
            output = model(images)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec, batchSize: {}".format(i, elapsed, batch_size), flush=True)

            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

    return {}

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)
