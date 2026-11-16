"""Launch the unchanged trainer and record this run's status outside research code."""
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import yaml


CONTROL = Path(__file__).resolve().parent
ROOT = CONTROL.parents[1]
NAME = CONTROL.name.removesuffix('_control')
RUN = ROOT / 'hirobe' / NAME
TERMINAL = ROOT / 'hirobe' / f'terminal_{NAME}.log'
PYTHON = ROOT / '.venv/bin/python'
DATASET = ROOT / 'SemanticKitti'
ARCH = ROOT / 'spherical_bev_fusion/config_nocopypaste.yaml'
LABELS = ROOT / 'spherical_bev_fusion/labels.yaml'
COMMAND = [str(PYTHON), '-u', '-B', '-m', 'spherical_bev_fusion.train',
           '--dataset', str(DATASET), '--arch_cfg', str(ARCH),
           '--data_cfg', str(LABELS), '--log', str(RUN)]


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def command_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True, timeout=15).strip()


def metadata():
    assert not RUN.exists(), f'Fresh run required: {RUN}'
    assert not TERMINAL.exists(), f'Terminal log already exists: {TERMINAL}'
    assert 'PASS:' in (CONTROL / 'check_model.log').read_text()
    check_log = (CONTROL / 'check_nocopypaste.log').read_text()
    check_path = Path(check_log.split('Preflight result: ')[-1].strip())
    check = json.loads(check_path.read_text())
    assert check['status'] == 'passed' and check['device'] == 'cuda'
    assert check['forward_loss_backward'] == 'passed'
    assert check['copy_paste_called'] is False
    assert check['dataset'] == str(DATASET)
    arch = yaml.safe_load(ARCH.read_text())
    labels = yaml.safe_load(LABELS.read_text())
    assert arch['dataset']['copy_paste'] is False
    assert arch['train']['batch_size'] == 2 and arch['train']['max_epochs'] == 100
    assert arch['train']['lr'] == .001 and arch['train']['w_decay'] == .01
    assert labels['split']['train'] == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    assert labels['split']['valid'] == [8]
    compute = command_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                              '--format=csv,noheader'])
    assert not compute, f'GPU compute processes appeared; inspect before starting: {compute}'
    sources = ['train.py', 'trainer.py', 'model.py', 'dataset.py', 'projection.py',
               'config_nocopypaste.yaml', 'labels.yaml']
    hashes = {f'spherical_bev_fusion/{name}': hashlib.sha256(
        (ROOT / 'spherical_bev_fusion' / name).read_bytes()).hexdigest() for name in sources}
    assert hashes['spherical_bev_fusion/model.py'] == check['model_sha256']
    counts = {split: {f'{seq:02d}': len(list(
        (DATASET / 'sequences' / f'{seq:02d}' / 'velodyne').glob('*.bin')))
        for seq in labels['split'][split]} for split in ('train', 'valid')}
    return dict(started_at=timestamp(), branch=command_output(['git', 'branch', '--show-current']),
                commit=command_output(['git', 'rev-parse', 'HEAD']), dataset=str(DATASET),
                run_dir=str(RUN), terminal_log=str(TERMINAL), command=shlex.join(COMMAND),
                initialization='scratch; no resume or pretrained checkpoint', config=arch,
                sequence_counts=counts, source_sha256=hashes, preflight=check,
                preflight_result=str(check_path), python=sys.version,
                libraries={name: version(name) for name in ('torch', 'numpy', 'PyYAML', 'tensorboardX')},
                metric_domain='polar_bev_pixels (not official SemanticKITTI point-wise mIoU)')


def process_sample(pid):
    # Include this training process and its DataLoader workers, without reading command lines.
    table = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            table[int(path.parent.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]),
                                           int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError):
            continue
    members = {pid}
    while True:
        expanded = members | {p for p, row in table.items() if row[0] in members}
        if expanded == members:
            break
        members = expanded
    return {p: table[p] for p in members if p in table}


def epoch_records():
    path = RUN / 'metrics.jsonl'
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines(keepends=True):
        if line.endswith('\n'):
            records.append(json.loads(line))
    return records


def write_report(info, returncode):
    rows = epoch_records()
    result_path = RUN / 'results.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    complete = returncode == 0 and result is not None and result['epochs_completed'] == 100
    report = dict(status='completed' if complete else 'failed_or_incomplete',
                  returncode=returncode, finished_at=timestamp(), branch=info['branch'],
                  dataset=str(DATASET), run_dir=str(RUN), epochs_evaluated=len(rows),
                  metric_domain=info['metric_domain'], trainer_results=result,
                  best_checkpoint=str(RUN / 'best_val.path'))
    labels = yaml.safe_load((RUN / 'data_cfg.yaml').read_text())
    if rows:
        best = max(rows, key=lambda row: (row['valid']['miou'], row['epoch']))
        report.update(best_epoch=best['epoch'], best_validation_miou=best['valid']['miou'],
                      best_validation_accuracy=best['valid']['accuracy'],
                      best_class_iou={labels['labels'][labels['learning_map_inv'][cid]]: iou
                                      for cid, iou in enumerate(best['valid']['class_iou'], 1)})
    with (RUN / 'loss_history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_loss', 'validation_loss', 'train_miou', 'validation_miou'])
        writer.writerows([row['epoch'], row['train']['loss'], row['valid']['loss'],
                          row['train']['miou'], row['valid']['miou']] for row in rows)
    report['terminal_error_warning_lines'] = [line for line in TERMINAL.read_text().splitlines()
        if any(word in line.lower() for word in ('warning', 'traceback', 'error'))]
    save_json(RUN / 'training_report.json', report)
    return report


def main():
    info = metadata()
    save_json(CONTROL / 'execution_metadata.json', info)
    start = time.monotonic()
    previous_time, previous_processes = start, {}
    log_offset = 0
    copied_metadata = False
    with TERMINAL.open('x', buffering=1) as terminal:
        terminal.write(f"{timestamp()} Starting NEW training\n{info['command']}\n")
        terminal.write(f"Branch: {info['branch']} | Dataset: {DATASET} | Output: {RUN}\n")
        child = subprocess.Popen(COMMAND, cwd=ROOT, stdout=terminal, stderr=subprocess.STDOUT)
        info['training_pid'] = child.pid
        info['monitor_pid'] = os.getpid()
        save_json(CONTROL / 'execution_metadata.json', info)
        print(f'{timestamp()} Training PID={child.pid}, monitor PID={os.getpid()}', flush=True)
        try:
            while True:
                now = time.monotonic()
                if (RUN / 'data_cfg.yaml').exists() and not copied_metadata:
                    save_json(RUN / 'execution_metadata.json', info)
                    (RUN / 'used_files.text').write_text(
                        json.dumps(info, indent=2) + '\n\n'
                        'GPU/CPU samples: resource_usage.jsonl\n'
                        'ETA after the first full epoch: runtime_status.json\n'
                        'CPU percent uses one core = 100%; GPU utilization is device-wide.\n'
                        'No bottleneck attribution is made without profiling evidence.\n')
                    copied_metadata = True
                log = RUN / 'log.txt'
                if log.exists():
                    with log.open() as stream:
                        stream.seek(log_offset)
                        fresh = stream.read()
                        log_offset = stream.tell()
                    if fresh:
                        terminal.write(fresh)
                        print(fresh, end='', flush=True)
                records = epoch_records()
                elapsed = now - start
                epochs = len(records)
                status = dict(timestamp=timestamp(), training_pid=child.pid, monitor_pid=os.getpid(),
                              running=child.poll() is None, elapsed_seconds=elapsed,
                              epochs_evaluated=epochs, planned_epochs=100,
                              remaining_seconds_estimate=None)
                if epochs:
                    # Use completion time, not time spent in the currently running epoch.
                    measured = (RUN / 'metrics.jsonl').stat().st_mtime - info_start_wall
                    status['remaining_seconds_estimate'] = max(0, measured / epochs * 100 - elapsed)
                processes = process_sample(child.pid)
                ticks = sum(max(0, row[1] - previous_processes[p][1])
                            for p, row in processes.items() if p in previous_processes)
                cpu = ticks / os.sysconf('SC_CLK_TCK') / max(now - previous_time, .001) * 100
                sample = dict(timestamp=timestamp(), elapsed_seconds=elapsed,
                              process_tree_cpu_percent=cpu if previous_processes else None,
                              process_tree_rss_bytes=sum(row[2] for row in processes.values()))
                try:
                    sample['gpu_util_memory'] = command_output(['nvidia-smi',
                        '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                        '--format=csv,noheader,nounits'])
                except (OSError, subprocess.SubprocessError) as exc:
                    sample['gpu_query_error'] = str(exc)
                previous_time, previous_processes = now, processes
                target = RUN if copied_metadata else CONTROL
                save_json(target / 'runtime_status.json', status)
                with (target / 'resource_usage.jsonl').open('a') as stream:
                    stream.write(json.dumps(sample) + '\n')
                if copied_metadata:
                    with (RUN / 'used_files.text').open('a') as stream:
                        stream.write(json.dumps(dict(**sample, remaining_seconds_estimate=
                            status['remaining_seconds_estimate'])) + '\n')
                if child.poll() is not None:
                    break
                time.sleep(30)
        except Exception:
            # Monitoring failure must not kill or reconfigure the user's training.
            traceback.print_exc()
            child.wait()
        returncode = child.wait()
        if (RUN / 'data_cfg.yaml').exists():
            report = write_report(info, returncode)
            terminal.write(json.dumps(report, indent=2) + '\n')
        print(f'{timestamp()} Training exit code: {returncode}', flush=True)
        return returncode


if __name__ == '__main__':
    info_start_wall = time.time()
    raise SystemExit(main())
