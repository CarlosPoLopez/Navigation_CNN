"""Dispatcher for the 8-GPU A100 node.

Reserves a full node (--gres=gpu:a100:8) and trains the FULL GRID with k-fold,
distributing the jobs across the 8 GPUs: it keeps 8 train.py processes running at a
time (one per GPU via CUDA_VISIBLE_DEVICES) and pulls jobs from a queue until it is
empty.

Grid: 3 lr x 4 bs x 5 folds = 60 trainings of 140 epochs at 256x256, with the
fixed input config (goal-conditioning) and the fixed loss (BCE + Tversky + clDice).
At 256 each one is ~1.5 h -> 60/5 = 12 rounds ~= 18 h makespan (5 GPUs); resumable.
Selection only needs to rank lr x bs; the winner is retrained separately. bs2 and bs4
are for the robustness section of the report (small batch), not for selecting; bs8 vs
bs16 is the selection question. bs16 is ranked FIRST.

Results -> $RUNS_ROOT/lr<LR>_bs<BS>/fold<F>/best.pt.

Resumable: if the job is resubmitted, it skips the (config, fold) that already have best.pt.
"""
import os
import sys
import time
import subprocess
from collections import deque

# Paths. All overridable via environment variables so the same script runs on your PC and
# on an HPC cluster. Defaults are relative to the repository root (this file lives in
# analysis/), so with a data/ and runs/ layout it works out of the box; on a cluster,
# export PYTHON / NET_DIR / DATA_DIR / TEST_IDS / RUNS_ROOT to point at your paths.
_REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY      = os.environ.get('PYTHON', sys.executable)             # python that launches train.py
NET     = os.environ.get('NET_DIR', _REPO)                     # folder where train.py lives
DATA    = os.environ.get('DATA_DIR', os.path.join(_REPO, 'data', 'Dataset_Matrices'))
TESTIDS = os.environ.get('TEST_IDS', os.path.join(_REPO, 'data', 'test_ids.txt'))
ROOT    = os.environ.get('RUNS_ROOT', os.path.join(_REPO, 'runs'))
LOGDIR  = os.path.join(ROOT, '_logs')

# LRS/BSS overridable via environment (comma-separated list) for partial sweeps
# without touching the grid: e.g. BSS=2 resumes only the bs2 of robustness (the lr
# are the same as the grid) without re-launching bs4/8/16, which are already done.
LRS    = os.environ.get('LRS', '3e-4,1e-4,3e-5').split(',')   # log-spaced around the winning lr1e-4
BSS    = os.environ.get('BSS', '16,8,4,2').split(',')         # bs16 FIRST; bs2/bs4 -> robustness section
FOLDS  = [0, 1, 2, 3, 4]
EPOCHS = '140'
NGPU   = int(os.environ.get('NGPU', '8'))
# The input (3-channel, 256px), the loss (BCE+Tversky+clDice) and the 5-level U-Net are
# all fixed in train.py; only lr x batch x fold are swept here.

os.makedirs(LOGDIR, exist_ok=True)


def best_exists(lr, bs, f):
    return os.path.exists(os.path.join(ROOT, f'lr{lr}_bs{bs}', f'fold{f}', 'best.pt'))


def verify_spread(running):
    """Checks with nvidia-smi that each process landed on a different GPU.
    Only logs (does not abort): if something is wrong it shows in the .o and is fixed by hand."""
    pids = {info[0].pid for info in running.values()}
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
            text=True, timeout=30)
    except Exception as e:
        print(f'[VERIFY] could not run nvidia-smi: {e}', flush=True)
        return
    pid2uuid = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 2:
            try:
                pid2uuid[int(parts[0])] = parts[1]
            except ValueError:
                pass
    distinct = {pid2uuid[p] for p in pids if p in pid2uuid}
    print(f'[VERIFY] processes in flight={len(pids)}  distinct GPUs={len(distinct)}', flush=True)
    for p in sorted(pids):
        print(f'[VERIFY]   pid {p} -> {pid2uuid.get(p, "(not showing up yet)")}', flush=True)
    if len(distinct) >= len(pids) and pids:
        print('[VERIFY] OK: each process on a different GPU. All 8 in use.', flush=True)
    else:
        print('[VERIFY] *** WARNING: processes sharing a GPU. Check the pinning. ***', flush=True)


def launch(gpu, task):
    lr, bs, f = task
    ckpt = os.path.join(ROOT, f'lr{lr}_bs{bs}')
    log = open(os.path.join(LOGDIR, f'lr{lr}_bs{bs}_fold{f}.log'), 'w')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, os.path.join(NET, 'train.py'),
           '--data_dir', DATA, '--test_ids', TESTIDS,
           '--ckpt_dir', ckpt, '--k', '5', '--fold', str(f),
           '--batch_size', bs, '--epochs', EPOCHS, '--lr', lr]
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=NET)
    print(f'[GPU {gpu}] START  lr={lr} bs={bs} fold={f}', flush=True)
    return (p, task, log, time.time())


def main():
    tasks = [(lr, bs, f) for lr in LRS for bs in BSS for f in FOLDS]
    pending = deque(t for t in tasks if not best_exists(*t))
    total = len(pending)
    print(f'Tasks: {len(tasks)} | already done: {len(tasks)-total} | pending: {total} '
          f'| GPUs: {NGPU}', flush=True)

    running = {}            # gpu -> (proc, task, log, start)
    free = list(range(NGPU))
    done = 0
    verified = False

    while pending or running:
        # Fill the free GPUs (with a small stagger to not saturate the FS at start-up).
        while free and pending:
            gpu = free.pop()
            running[gpu] = launch(gpu, pending.popleft())
            time.sleep(3)
        # One-off verification of the GPU spread, once the first round is launched.
        if not verified and len(running) >= min(NGPU, total):
            time.sleep(90)      # give each process time to reserve its GPU
            verify_spread(running)
            verified = True
        time.sleep(10)
        # Collect the ones that have finished.
        for gpu, (p, task, log, st) in list(running.items()):
            if p.poll() is not None:
                log.close()
                done += 1
                lr, bs, f = task
                status = 'OK  ' if p.returncode == 0 else f'FAIL(rc={p.returncode})'
                print(f'[GPU {gpu}] {status} lr={lr} bs={bs} fold={f}  '
                      f'{(time.time()-st)/3600:.2f}h  ({done}/{total})', flush=True)
                del running[gpu]
                free.append(gpu)

    # Final balance: if MANY best.pt are missing (e.g. the node went down) it exits with
    # a code !=0 so the old sweep (chained with afternotok) kicks in as a fallback. If
    # only a few are missing (isolated failures) it exits OK and they are patched by
    # hand, without triggering a redundant full re-run.
    expected = [(lr, bs, f) for lr in LRS for bs in BSS for f in FOLDS]
    missing = [t for t in expected if not best_exists(*t)]
    if len(missing) > len(expected) // 2:
        print(f'SEVERE FAILURE: {len(missing)}/{len(expected)} best.pt missing -> exit 1 '
              f'(fallback activated)', flush=True)
        sys.exit(1)
    if missing:
        print(f'WARNING: {len(missing)} best.pt missing (isolated failures, patch by hand): '
              f'{missing}', flush=True)
    print(f'DISPATCHER COMPLETE: {len(expected)-len(missing)}/{len(expected)} best.pt', flush=True)


if __name__ == '__main__':
    main()

