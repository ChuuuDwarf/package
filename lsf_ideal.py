#!/usr/bin/env python3
"""
lsf_ideal.py — DAG 排程的理想狀態分析

計算 Work / Span 下界、理想資源量 P*，並用 list scheduling
產生各資源量下的排程與加速曲線。

輸入 CSV 欄位:
    job_id      必要
    step        必要 (整數)
    runtime_sec 必要 (浮點)
    depends_on  選填, 分號分隔的 job_id; 全部留空則視為 step barrier 模式
    slots       選填, 預設 1
"""

import argparse
import csv
import sys
from collections import defaultdict


class Job:
    __slots__ = ("jid", "step", "runtime", "deps", "slots", "rank", "start", "end")

    def __init__(self, jid, step, runtime, deps, slots):
        self.jid = jid
        self.step = step
        self.runtime = runtime
        self.deps = deps
        self.slots = slots
        self.rank = 0.0
        self.start = None
        self.end = None


def load_jobs(path):
    jobs = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            jid = row["job_id"].strip()
            if not jid:
                continue
            if jid in jobs:
                sys.exit(f"重複的 job_id: {jid}")
            raw_deps = (row.get("depends_on") or "").strip()
            deps = [d.strip() for d in raw_deps.split(";") if d.strip()]
            jobs[jid] = Job(
                jid=jid,
                step=int(row["step"]),
                runtime=float(row["runtime_sec"]),
                deps=deps,
                slots=int(row.get("slots") or 1),
            )
    if not jobs:
        sys.exit("CSV 沒有任何 job")
    return jobs


def apply_barrier(jobs):
    """無顯式相依時，用 step 當 barrier: step N 的 job 依賴 step N-1 全部。"""
    by_step = defaultdict(list)
    for j in jobs.values():
        by_step[j.step].append(j.jid)
    steps = sorted(by_step)
    for prev, cur in zip(steps, steps[1:]):
        for jid in by_step[cur]:
            jobs[jid].deps = list(by_step[prev])
    return jobs


def build_succ(jobs):
    succ = defaultdict(list)
    for j in jobs.values():
        for d in j.deps:
            if d not in jobs:
                sys.exit(f"{j.jid} 依賴不存在的 job: {d}")
            succ[d].append(j.jid)
    return succ


def topo_order(jobs, succ):
    indeg = {jid: len(j.deps) for jid, j in jobs.items()}
    stack = [jid for jid, d in indeg.items() if d == 0]
    order = []
    while stack:
        jid = stack.pop()
        order.append(jid)
        for s in succ[jid]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    if len(order) != len(jobs):
        sys.exit("DAG 有環，無法排程")
    return order


def compute_rank(jobs, succ, order):
    """upward rank: 該 job 到終點的最長路徑長度。用作 list scheduling 優先序。"""
    for jid in reversed(order):
        j = jobs[jid]
        j.rank = j.runtime + max((jobs[s].rank for s in succ[jid]), default=0.0)


def critical_path(jobs, succ, order):
    compute_rank(jobs, succ, order)
    span = max(j.rank for j in jobs.values())
    # 回推關鍵路徑上的 job
    path, cur = [], None
    for jid, j in jobs.items():
        if not j.deps and abs(j.rank - span) < 1e-9:
            cur = jid
            break
    while cur is not None:
        path.append(cur)
        nxt, best = None, -1.0
        for s in succ[cur]:
            if jobs[s].rank > best:
                nxt, best = s, jobs[s].rank
        cur = nxt
    return span, path


def simulate(jobs, succ, capacity):
    """事件驅動 list scheduling。回傳 (makespan, schedule)。"""
    remaining = {jid: len(j.deps) for jid, j in jobs.items()}
    ready = [jid for jid, d in remaining.items() if d == 0]
    running = []          # (end_time, jid)
    free = capacity
    now = 0.0
    sched = {}
    done = 0
    n = len(jobs)

    while done < n:
        # 依 upward rank 由大到小嘗試派送
        ready.sort(key=lambda x: -jobs[x].rank)
        launched = True
        while launched:
            launched = False
            for i, jid in enumerate(ready):
                if jobs[jid].slots <= free:
                    j = jobs[jid]
                    free -= j.slots
                    end = now + j.runtime
                    sched[jid] = (now, end)
                    running.append((end, jid))
                    ready.pop(i)
                    launched = True
                    break

        if not running:
            sys.exit("無法派送: 有 job 的 slots 需求超過總資源量")

        # 推進到下一個完成事件
        running.sort()
        now = running[0][0]
        while running and abs(running[0][0] - now) < 1e-9:
            _, jid = running.pop(0)
            free += jobs[jid].slots
            done += 1
            for s in succ[jid]:
                remaining[s] -= 1
                if remaining[s] == 0:
                    ready.append(s)

    makespan = max(e for _, e in sched.values())
    return makespan, sched


def main():
    ap = argparse.ArgumentParser(description="DAG 排程理想狀態分析")
    ap.add_argument("csv", help="輸入的 job CSV")
    ap.add_argument("--max-slots", type=int, default=0,
                    help="掃描的最大資源量, 預設自動取 2*P*")
    ap.add_argument("--schedule-at", type=int, default=0,
                    help="輸出此資源量下的排程明細, 預設用 P*")
    ap.add_argument("-o", "--out", default="schedule.csv",
                    help="排程輸出檔")
    ap.add_argument("--curve", default="speedup.csv",
                    help="加速曲線輸出檔")
    args = ap.parse_args()

    jobs = load_jobs(args.csv)
    if not any(j.deps for j in jobs.values()):
        print("[i] 未偵測到顯式相依，採用 step barrier 模式")
        jobs = apply_barrier(jobs)
    else:
        print("[i] 偵測到顯式相依，採用細粒度 DAG 模式")

    succ = build_succ(jobs)
    order = topo_order(jobs, succ)

    work = sum(j.runtime * j.slots for j in jobs.values())
    span, cpath = critical_path(jobs, succ, order)
    p_star = work / span if span else 1.0
    max_slot_req = max(j.slots for j in jobs.values())

    print(f"\n{'='*52}")
    print(f"job 數量            : {len(jobs)}")
    print(f"Work  W (slot-sec)  : {work:,.1f}")
    print(f"Span  L (sec)       : {span:,.1f}   <- makespan 的絕對下界")
    print(f"平均平行度 W/L      : {p_star:,.2f}")
    print(f"理想資源量 P*       : {max(int(p_star + 0.999), max_slot_req)}")
    print(f"關鍵路徑長度        : {len(cpath)} 個 job")
    print(f"關鍵路徑            : {' -> '.join(cpath[:8])}"
          + (" ..." if len(cpath) > 8 else ""))
    print(f"{'='*52}\n")

    p_ideal = max(int(p_star + 0.999), max_slot_req)
    p_max = args.max_slots or max(p_ideal * 2, max_slot_req + 1)

    rows = []
    print(f"{'slots':>6} {'makespan':>12} {'加速比':>8} {'效率':>8} {'距下界':>8}")
    print("-" * 48)
    for p in range(max_slot_req, p_max + 1):
        mk, _ = simulate(jobs, succ, p)
        lb = max(work / p, span)
        rows.append((p, mk, work / mk, work / mk / p, mk / lb))
        if p <= 4 or p % max(1, p_max // 12) == 0 or p == p_ideal:
            mark = "  <- P*" if p == p_ideal else ""
            print(f"{p:>6} {mk:>12,.1f} {work/mk:>8.2f} "
                  f"{work/mk/p:>7.1%} {mk/lb:>7.2f}x{mark}")

    with open(args.curve, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slots", "makespan_sec", "speedup", "efficiency", "vs_lower_bound"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.3f}"])

    p_sched = args.schedule_at or p_ideal
    mk, sched = simulate(jobs, succ, p_sched)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "step", "start_sec", "end_sec", "runtime_sec",
                    "slots", "is_critical"])
        cset = set(cpath)
        for jid in sorted(sched, key=lambda x: sched[x][0]):
            s, e = sched[jid]
            j = jobs[jid]
            w.writerow([jid, j.step, f"{s:.2f}", f"{e:.2f}",
                        f"{j.runtime:.2f}", j.slots, int(jid in cset)])

    print(f"\n[✓] {args.curve}  加速曲線")
    print(f"[✓] {args.out}  P={p_sched} 的排程 (makespan {mk:,.1f}s)")


if __name__ == "__main__":
    main()
