"""Resume the existing experiment without changing its trainer or configuration."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
import traceback

import torch

import run_training as base


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main():
    original = json.loads((base.RUN / 'execution_metadata.json').read_text())
    for name, expected in original['source_sha256'].items():
        assert hashlib.sha256((base.ROOT / name).read_bytes()).hexdigest() == expected, name
    assert not base.command_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
    existing = subprocess.run(['pgrep', '-af', 'spherical_bev_fusion[.]train'],
                              capture_output=True, text=True)
    assert existing.returncode == 1, 'A training process may already be running'
    latest = torch.load(base.RUN / 'latest.path', map_location='cpu', weights_only=False)
    checkpoint = base.RUN / 'best_val.path'
    best = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for key in ('epoch', 'model', 'optim', 'scheduler', 'metrics'):
        assert same(latest[key], best[key]), f'Best is no longer latest: {key}'
    initial_records = base.epoch_records()
    completed = int(best['epoch']) + 1
    assert initial_records[-1]['epoch'] == completed == 2
    assert best['best_miou'] == max(row['valid']['miou'] for row in initial_records)
    assert all(torch.isfinite(tensor).all() for tensor in best['model'].values())
    assert (base.RUN / 'arch_cfg.yaml').read_bytes() == base.ARCH.read_bytes()
    assert (base.RUN / 'data_cfg.yaml').read_bytes() == base.LABELS.read_bytes()
    command = [str(base.PYTHON), '-u', '-B', '-m', 'spherical_bev_fusion.train',
               '--dataset', str(base.DATASET), '--arch_cfg', str(base.RUN / 'arch_cfg.yaml'),
               '--data_cfg', str(base.RUN / 'data_cfg.yaml'), '--log', str(base.RUN),
               '--resume', str(checkpoint)]
    info = dict(resumed_at=base.timestamp(), completed_epochs=completed,
                next_display_epoch=completed + 1, branch=base.command_output(
                    ['git', 'branch', '--show-current']), command=shlex.join(command),
                resume_checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(
                    checkpoint.read_bytes()).hexdigest(), best_miou=best['best_miou'],
                restored_lr=[group['lr'] for group in best['optim']['param_groups']],
                scheduler_last_epoch=best['scheduler']['last_epoch'],
                checkpoint_matches_latest_model_optimizer_scheduler=True,
                source_sha256=original['source_sha256'],
                stop_reason='Unknown: no terminal traceback; former processes no longer exist')
    original_start = datetime.fromisoformat(original['started_at']).timestamp()
    seconds_per_epoch = ((base.RUN / 'metrics.jsonl').stat().st_mtime - original_start) / completed
    del best, latest
    start_wall, start = time.time(), time.monotonic()
    previous_time, previous_processes = start, {}
    log_offset = (base.RUN / 'log.txt').stat().st_size
    metadata_path = base.RUN / f'resume_{datetime.now():%Y%m%d_%H%M%S}.json'
    with base.TERMINAL.open('a', buffering=1) as terminal:
        terminal.write(f'\n{base.timestamp()} RESUMING at displayed epoch {completed + 1}/100\n')
        terminal.write(json.dumps(info, indent=2) + '\n')
        child = subprocess.Popen(command, cwd=base.ROOT, stdin=subprocess.DEVNULL,
                                 stdout=terminal, stderr=subprocess.STDOUT)
        info.update(training_pid=child.pid, monitor_pid=os.getpid())
        base.save_json(metadata_path, info)
        with (base.RUN / 'used_files.text').open('a') as stream:
            stream.write('\nRESUME\n' + json.dumps(info, indent=2) + '\n')
        print(json.dumps(info, indent=2), flush=True)
        try:
            while True:
                now = time.monotonic()
                with (base.RUN / 'log.txt').open() as stream:
                    stream.seek(log_offset)
                    fresh = stream.read()
                    log_offset = stream.tell()
                if fresh:
                    terminal.write(fresh)
                    print(fresh, end='', flush=True)
                records = base.epoch_records()
                epochs = records[-1]['epoch']
                elapsed = now - start
                if epochs > completed:
                    seconds_per_epoch = ((base.RUN / 'metrics.jsonl').stat().st_mtime
                                         - start_wall) / (epochs - completed)
                remaining = max(0, seconds_per_epoch * (100 - completed) - elapsed)
                status = dict(timestamp=base.timestamp(), training_pid=child.pid,
                              monitor_pid=os.getpid(), running=child.poll() is None,
                              resumed_at=info['resumed_at'], session_elapsed_seconds=elapsed,
                              epochs_evaluated=epochs, planned_epochs=100,
                              remaining_seconds_estimate=remaining,
                              estimate_basis='completed epoch timings; excludes stopped time')
                processes = base.process_sample(child.pid)
                ticks = sum(max(0, row[1] - previous_processes[pid][1])
                            for pid, row in processes.items() if pid in previous_processes)
                cpu = ticks / os.sysconf('SC_CLK_TCK') / max(now - previous_time, .001) * 100
                sample = dict(timestamp=base.timestamp(), session_elapsed_seconds=elapsed,
                              process_tree_cpu_percent=cpu if previous_processes else None,
                              process_tree_rss_bytes=sum(row[2] for row in processes.values()))
                try:
                    sample['gpu_util_memory'] = base.command_output(['nvidia-smi',
                        '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                        '--format=csv,noheader,nounits'])
                except (OSError, subprocess.SubprocessError) as exc:
                    sample['gpu_query_error'] = str(exc)
                base.save_json(base.RUN / 'runtime_status.json', status)
                for name in ('resource_usage.jsonl', 'used_files.text'):
                    with (base.RUN / name).open('a') as stream:
                        stream.write(json.dumps(dict(**sample, remaining_seconds_estimate=remaining)) + '\n')
                previous_time, previous_processes = now, processes
                if child.poll() is not None:
                    break
                time.sleep(30)
        except Exception:
            traceback.print_exc()
            child.wait()
        returncode = child.wait()
        base.save_json(base.CONTROL / 'resume_exit_status.json',
                       dict(returncode=returncode, finished_at=base.timestamp()))
        report = base.write_report(original, returncode)
        terminal.write(json.dumps(report, indent=2) + '\n')
        print(f'{base.timestamp()} Training exit code: {returncode}', flush=True)
        return returncode


if __name__ == '__main__':
    with (base.CONTROL / 'resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main())
