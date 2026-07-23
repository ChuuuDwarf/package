#!/usr/bin/env python3
"""
lsf_scheduler_analysis.py — LSF 排程分析框架

對每個 step 內部的 job 進行排程分析：
  1. ILP 求數學最優解
  2. 啟發式演算法比較 (LPT / SPT / FIFO)
  3. Array vs Session trade-off 量化
  4. 資源量掃描 (邊際效益)
  5. 特徵萃取 → Decision Tree 規則產出

輸入 CSV 欄位 (最少需要):
    job_id       必要
    step         必要 (整數或字串, 代表所屬 step)
    start_time   必要 (秒或時間戳)
    end_time     必要 (秒或時間戳)
    submit_time  選填 (有的話可精確算 dispatch overhead)
    slots        選填 (每個 job 用幾個 slot, 預設 1)

輸出:
    step_analysis.csv   每個 step 的分析結果
    rules.txt           Decision Tree 萃取的規則
    resource_sweep.csv   各 step 不同資源量下的表現
"""

import argparse
import csv
import sys
import math
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# 1. 資料載入
# ---------------------------------------------------------------------------

def parse_time(val):
    """嘗試把 val 轉成秒數 (float)。支援純數字或常見時間格式。"""
    val = val.strip()
    # 純數字
    try:
        return float(val)
    except ValueError:
        pass
    # 常見時間格式
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(val, fmt).timestamp()
        except ValueError:
            continue
    sys.exit(f"無法解析時間: {val}")


def load_csv(path):
    """載入 CSV, 回傳 {step: [Job, ...]}"""
    steps = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        if "job_id" not in cols or "step" not in cols:
            sys.exit("CSV 必須包含 job_id 和 step 欄位")
        has_submit = "submit_time" in cols
        has_slots = "slots" in cols

        # 支援 runtime_sec 直接給定 或 start/end 計算
        has_runtime = "runtime_sec" in cols
        has_times = "start_time" in cols and "end_time" in cols

        if not has_runtime and not has_times:
            sys.exit("CSV 必須包含 runtime_sec 或 (start_time + end_time)")

        for row in reader:
            jid = row["job_id"].strip()
            if not jid:
                continue
            step = row["step"].strip()

            if has_runtime:
                runtime = float(row["runtime_sec"])
                start = parse_time(row["start_time"]) if has_times else 0.0
                end = start + runtime
            else:
                start = parse_time(row["start_time"])
                end = parse_time(row["end_time"])
                runtime = end - start
                if runtime <= 0:
                    print(f"[warn] job {jid} runtime <= 0, 跳過")
                    continue

            submit = parse_time(row["submit_time"]) if has_submit and row.get("submit_time", "").strip() else None
            slots = int(row["slots"]) if has_slots and row.get("slots", "").strip() else 1

            steps[step].append({
                "jid": jid,
                "runtime": runtime,
                "start": start,
                "end": end,
                "submit": submit,
                "slots": slots,
            })
    if not steps:
        sys.exit("CSV 沒有任何有效 job")
    return steps


# ---------------------------------------------------------------------------
# 2. ILP 最優解
# ---------------------------------------------------------------------------

def solve_ilp(runtimes, m, time_limit=60):
    """
    P||Cmax ILP: N 個 job 分到 M 個 slot, 最小化 makespan。
    runtimes: list of float
    m: slot 數
    回傳 (optimal_makespan, status, assignment)
    """
    import pulp

    n = len(runtimes)
    if n == 0:
        return 0.0, "Optimal", {}

    prob = pulp.LpProblem("Pllcmax", pulp.LpMinimize)

    # 決策變數: x[i][j] = 1 表示 job i 分到 slot j
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    # Cmax
    cmax = pulp.LpVariable("Cmax", lowBound=0)

    # 目標: minimize Cmax
    prob += cmax

    # 約束 1: 每個 job 恰好分到一個 slot
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(m)) == 1

    # 約束 2: Cmax >= 每個 slot 的總負載
    for j in range(m):
        prob += cmax >= pulp.lpSum(runtimes[i] * x[i, j] for i in range(n))

    # 求解
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    opt = pulp.value(cmax) if status == "Optimal" else None

    assignment = {}
    if status in ("Optimal", "Feasible"):
        opt = pulp.value(cmax)
        for i in range(n):
            for j in range(m):
                if pulp.value(x[i, j]) > 0.5:
                    assignment[i] = j
                    break

    return opt, status, assignment


# ---------------------------------------------------------------------------
# 3. 啟發式排程
# ---------------------------------------------------------------------------

def schedule_heuristic(runtimes, m, strategy="LPT"):
    """
    用啟發式把 N 個 job 排到 M 個 slot。
    strategy: LPT / SPT / FIFO
    回傳 (makespan, slot_loads)
    """
    n = len(runtimes)
    if n == 0:
        return 0.0, []

    indices = list(range(n))
    if strategy == "LPT":
        indices.sort(key=lambda i: -runtimes[i])
    elif strategy == "SPT":
        indices.sort(key=lambda i: runtimes[i])
    # FIFO = 原始順序

    slot_loads = [0.0] * m
    for i in indices:
        # 分配到目前負載最小的 slot
        j = min(range(m), key=lambda k: slot_loads[k])
        slot_loads[j] += runtimes[i]

    return max(slot_loads), slot_loads


# ---------------------------------------------------------------------------
# 4. Array vs Session 模擬
# ---------------------------------------------------------------------------

def simulate_array(runtimes, m, t_dispatch):
    """Array 模式: 每個 job 加上 dispatch overhead"""
    effective = [r + t_dispatch for r in runtimes]
    makespan, loads = schedule_heuristic(effective, m, "LPT")
    total_work = sum(effective)
    utilization = total_work / (m * makespan) if makespan > 0 else 0
    return makespan, utilization, total_work


def simulate_session(runtimes, m):
    """Session 模式: 不加 dispatch, 但 slot 持續佔用直到全部跑完"""
    makespan, loads = schedule_heuristic(runtimes, m, "LPT")
    actual_work = sum(runtimes)
    occupied = m * makespan  # session 佔用的總 slot-秒
    utilization = actual_work / occupied if occupied > 0 else 0
    return makespan, utilization, occupied


# ---------------------------------------------------------------------------
# 5. 特徵萃取
# ---------------------------------------------------------------------------

def extract_features(jobs, actual_makespan=None):
    """萃取一個 step 的統計特徵, 供 ML 使用。"""
    runtimes = [j["runtime"] for j in jobs]
    n = len(runtimes)
    if n == 0:
        return {}

    arr = np.array(runtimes)
    mean_rt = np.mean(arr)
    std_rt = np.std(arr)
    cov = std_rt / mean_rt if mean_rt > 0 else 0  # 變異係數
    skew = float(np.mean(((arr - mean_rt) / std_rt) ** 3)) if std_rt > 0 else 0

    total_work = np.sum(arr)

    # Dispatch overhead 估算
    dispatches = []
    for j in jobs:
        if j["submit"] is not None:
            d = j["start"] - j["submit"]
            if d >= 0:
                dispatches.append(d)
    if dispatches:
        mean_dispatch = np.mean(dispatches)
        dispatch_ratio = mean_dispatch / mean_rt if mean_rt > 0 else 0
    else:
        # 無 submit_time 時, 從同 step job 的 start time 間距估算
        starts = sorted(j["start"] for j in jobs)
        if len(starts) > 1:
            gaps = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
            mean_dispatch = np.median(gaps)  # 用中位數避免極端值
            dispatch_ratio = mean_dispatch / mean_rt if mean_rt > 0 else 0
        else:
            mean_dispatch = 0.0
            dispatch_ratio = 0.0

    return {
        "n_jobs": n,
        "total_work": float(total_work),
        "mean_runtime": float(mean_rt),
        "std_runtime": float(std_rt),
        "cov": float(cov),
        "skewness": float(skew),
        "min_runtime": float(np.min(arr)),
        "max_runtime": float(np.max(arr)),
        "range_ratio": float(np.max(arr) / np.min(arr)) if np.min(arr) > 0 else float("inf"),
        "mean_dispatch": float(mean_dispatch),
        "dispatch_ratio": float(dispatch_ratio),
    }


# ---------------------------------------------------------------------------
# 6. 資源掃描
# ---------------------------------------------------------------------------

def resource_sweep(runtimes, m_range, t_dispatch=0.0):
    """對不同 slot 數跑 LPT, 回傳 [(m, makespan, speedup, efficiency)]"""
    if not runtimes:
        return []
    baseline = sum(runtimes)  # 1 slot 的 makespan (串列)
    results = []
    for m in m_range:
        mk, _ = schedule_heuristic(runtimes, m, "LPT")
        speedup = baseline / mk if mk > 0 else 0
        efficiency = speedup / m if m > 0 else 0
        results.append((m, mk, speedup, efficiency))
    return results


# ---------------------------------------------------------------------------
# 7. Decision Tree 規則萃取
# ---------------------------------------------------------------------------

def train_decision_tree(features_list, labels):
    """
    用 Decision Tree 從特徵中萃取 array/session 選擇規則。
    features_list: list of dict (每個 step 的特徵)
    labels: list of str ("array" 或 "session")
    回傳 (tree_model, feature_names, rules_text)
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    if len(features_list) < 2:
        return None, [], "樣本太少, 無法訓練 Decision Tree (至少需要 2 個 step)"

    feature_names = ["n_jobs", "mean_runtime", "cov", "skewness",
                     "range_ratio", "dispatch_ratio"]

    X = []
    for f in features_list:
        row = [f.get(k, 0.0) for k in feature_names]
        X.append(row)
    X = np.array(X)

    # label 轉數字
    label_map = {"session": 1, "array": 0}
    y = np.array([label_map.get(l, 0) for l in labels])

    if len(np.unique(y)) < 2:
        return None, feature_names, "所有 step 的最佳策略相同, 無法產出分類規則"

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1, random_state=42)
    clf.fit(X, y)

    rules = export_text(clf, feature_names=feature_names,
                        class_names=["array", "session"])
    return clf, feature_names, rules


# ---------------------------------------------------------------------------
# 8. 主流程
# ---------------------------------------------------------------------------

def analyze_step(step_name, jobs, m_actual, ilp_time_limit=60):
    """對單一 step 進行完整分析。"""
    runtimes = [j["runtime"] for j in jobs]
    n = len(runtimes)

    if n == 0:
        return None

    # --- 特徵 ---
    feat = extract_features(jobs)
    t_dispatch = feat["mean_dispatch"]

    # --- Lower Bound ---
    lb = max(max(runtimes), sum(runtimes) / m_actual)

    # --- ILP 最優解 ---
    ilp_mk, ilp_status, _ = solve_ilp(runtimes, m_actual, ilp_time_limit)

    # --- 啟發式 ---
    lpt_mk, _ = schedule_heuristic(runtimes, m_actual, "LPT")
    spt_mk, _ = schedule_heuristic(runtimes, m_actual, "SPT")
    fifo_mk, _ = schedule_heuristic(runtimes, m_actual, "FIFO")

    # --- Array vs Session ---
    arr_mk, arr_util, arr_work = simulate_array(runtimes, m_actual, t_dispatch)
    ses_mk, ses_util, ses_occupied = simulate_session(runtimes, m_actual)

    # 判定: 哪個策略在這個 step 表現較好
    better = "session" if ses_mk < arr_mk else "array"
    diff_sec = abs(arr_mk - ses_mk)
    diff_pct = diff_sec / max(arr_mk, ses_mk) * 100 if max(arr_mk, ses_mk) > 0 else 0

    result = {
        "step": step_name,
        "n_jobs": n,
        "m_slots": m_actual,
        # 下界
        "lower_bound": round(lb, 2),
        # ILP
        "ilp_makespan": round(ilp_mk, 2) if ilp_mk else None,
        "ilp_status": ilp_status,
        # 啟發式
        "lpt_makespan": round(lpt_mk, 2),
        "spt_makespan": round(spt_mk, 2),
        "fifo_makespan": round(fifo_mk, 2),
        # 啟發式 vs ILP 差距
        "lpt_vs_ilp": f"{(lpt_mk / ilp_mk - 1) * 100:.1f}%" if ilp_mk else "N/A",
        # Array vs Session
        "array_makespan": round(arr_mk, 2),
        "array_utilization": round(arr_util, 4),
        "session_makespan": round(ses_mk, 2),
        "session_utilization": round(ses_util, 4),
        "better_strategy": better,
        "diff_sec": round(diff_sec, 2),
        "diff_pct": round(diff_pct, 1),
        # 特徵
        "dispatch_overhead": round(t_dispatch, 4),
        "dispatch_ratio": round(feat["dispatch_ratio"], 4),
        "cov": round(feat["cov"], 4),
        "skewness": round(feat["skewness"], 4),
        "range_ratio": round(feat["range_ratio"], 2),
    }
    return result, feat, better


def main():
    ap = argparse.ArgumentParser(description="LSF 排程分析框架")
    ap.add_argument("csv", help="輸入的 job CSV 檔")
    ap.add_argument("--slots", type=int, default=0,
                    help="每個 step 使用的 slot 數 (預設: 從資料推斷或用 10)")
    ap.add_argument("--ilp-timeout", type=int, default=60,
                    help="ILP solver 時間限制 (秒, 預設 60)")
    ap.add_argument("--sweep", action="store_true",
                    help="是否執行資源量掃描")
    ap.add_argument("-o", "--output", default="step_analysis.csv",
                    help="分析結果輸出 (預設 step_analysis.csv)")
    ap.add_argument("--rules", default="rules.txt",
                    help="Decision Tree 規則輸出 (預設 rules.txt)")
    ap.add_argument("--sweep-out", default="resource_sweep.csv",
                    help="資源掃描輸出 (預設 resource_sweep.csv)")
    args = ap.parse_args()

    # --- 載入 ---
    steps_data = load_csv(args.csv)
    print(f"[i] 載入 {sum(len(v) for v in steps_data.values())} 個 job, "
          f"{len(steps_data)} 個 step\n")

    # --- 推斷 slot 數 ---
    def infer_slots(jobs):
        """從同時間重疊的 job 數推斷 slot 數。"""
        events = []
        for j in jobs:
            events.append((j["start"], 1))
            events.append((j["end"], -1))
        events.sort()
        peak, cur = 0, 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)
        return peak

    m_global = args.slots

    # --- 逐 step 分析 ---
    all_results = []
    all_features = []
    all_labels = []

    step_keys = sorted(steps_data.keys(), key=lambda s: (s.isdigit() and int(s), s))

    print(f"{'step':>10} {'jobs':>6} {'slots':>6} {'LB':>10} {'ILP':>10} "
          f"{'LPT':>10} {'LPT/ILP':>8} {'Array':>10} {'Session':>10} {'Better':>8}")
    print("-" * 100)

    for step_name in step_keys:
        jobs = steps_data[step_name]
        m = m_global if m_global > 0 else infer_slots(jobs)
        m = max(m, 1)

        result, feat, better = analyze_step(step_name, jobs, m, args.ilp_timeout)
        if result is None:
            continue

        all_results.append(result)
        all_features.append(feat)
        all_labels.append(better)

        print(f"{step_name:>10} {result['n_jobs']:>6} {m:>6} "
              f"{result['lower_bound']:>10.1f} "
              f"{result['ilp_makespan'] or 'N/A':>10} "
              f"{result['lpt_makespan']:>10.1f} "
              f"{result['lpt_vs_ilp']:>8} "
              f"{result['array_makespan']:>10.1f} "
              f"{result['session_makespan']:>10.1f} "
              f"{result['better_strategy']:>8}")

    # --- 輸出分析結果 CSV ---
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
        print(f"\n[✓] {args.output}")

    # --- Decision Tree ---
    if len(all_features) >= 2:
        clf, fnames, rules_text = train_decision_tree(all_features, all_labels)
        print(f"\n{'='*60}")
        print("Decision Tree 規則:")
        print(f"{'='*60}")
        print(rules_text)
        with open(args.rules, "w", encoding="utf-8") as f:
            f.write("LSF Scheduler Analysis - Decision Tree Rules\n")
            f.write("=" * 60 + "\n\n")
            f.write("Features:\n")
            f.write("  n_jobs        : job 數量\n")
            f.write("  mean_runtime  : 平均 runtime (秒)\n")
            f.write("  cov           : 變異係數 (std/mean), 越大表示 job 長度差異越大\n")
            f.write("  skewness      : 偏度, 正值表示少數 job 特別長\n")
            f.write("  range_ratio   : 最長/最短 job 的比值\n")
            f.write("  dispatch_ratio: dispatch overhead / mean runtime\n\n")
            f.write("Rules:\n")
            f.write(rules_text)
        print(f"[✓] {args.rules}")
    else:
        print("\n[i] 只有 1 個 step, 跳過 Decision Tree")

    # --- 資源掃描 ---
    if args.sweep and all_results:
        print(f"\n{'='*60}")
        print("資源量掃描 (每個 step 的邊際效益)")
        print(f"{'='*60}")
        sweep_rows = []
        for step_name in step_keys:
            jobs = steps_data[step_name]
            runtimes = [j["runtime"] for j in jobs]
            m_inferred = m_global if m_global > 0 else infer_slots(jobs)
            m_max = min(len(runtimes), max(m_inferred * 3, 20))
            m_range = range(1, m_max + 1)
            results = resource_sweep(runtimes, m_range)
            for m, mk, sp, eff in results:
                sweep_rows.append({
                    "step": step_name,
                    "slots": m,
                    "makespan": round(mk, 2),
                    "speedup": round(sp, 3),
                    "efficiency": round(eff, 4),
                })
                # 找拐點
            # 印出摘要
            if results:
                baseline = results[0][1]
                for m, mk, sp, eff in results:
                    if mk <= results[0][1] * 0.5:
                        print(f"  step {step_name}: slot={m} 時 makespan 降到 "
                              f"{mk:.0f}s ({sp:.1f}x 加速, {eff:.0%} 效率)")
                        break

        with open(args.sweep_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "slots", "makespan",
                                               "speedup", "efficiency"])
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"[✓] {args.sweep_out}")

    print("\n完成。")


if __name__ == "__main__":
    main()
