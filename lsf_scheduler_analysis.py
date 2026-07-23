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



另一個版本
#!/usr/bin/env python3
"""
lsf_scheduler_analysis.py — LSF 排程分析框架 v2

對每個 step 內部的 job 進行排程分析：
  1. ILP 求數學最優解 (理想無 overhead 的天花板)
  2. 啟發式演算法比較 (LPT / SPT / FIFO)
  3. Array vs Session trade-off: 用資料中的 dispatch 分佈模擬, 非固定值
  4. 資源量掃描 (邊際效益)
  5. 特徵萃取 → Decision Tree 規則產出

核心改進:
  - Array 模擬: 每個 job 的 dispatch overhead 從資料的實際分佈中取樣
  - Session 模擬: 一次 dispatch 拿 slot, task 直接餵入, 但 slot 持續佔用到結束
  - 長短 job 的切分: 自動找出 runtime 閾值, 驗證「長 job 適合 array, 短 job 適合 session」

輸入 CSV 欄位:
    job_id       必要
    step         必要
    start_time   必要
    end_time     必要
    submit_time  選填 (精確計算 dispatch overhead)
    slots        選填 (預設 1)
    runtime_sec  選填 (若有則忽略 start/end 算 runtime)
"""

import argparse
import csv
import sys
import math
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")


# ===================================================================
# 1. 資料載入
# ===================================================================

def parse_time(val):
    val = val.strip()
    try:
        return float(val)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(val, fmt).timestamp()
        except ValueError:
            continue
    sys.exit(f"無法解析時間: {val}")


def load_csv(path):
    steps = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        if "job_id" not in cols or "step" not in cols:
            sys.exit("CSV 必須包含 job_id 和 step 欄位")

        has_submit = "submit_time" in cols
        has_slots = "slots" in cols
        has_runtime = "runtime_sec" in cols
        has_times = "start_time" in cols and "end_time" in cols

        if not has_runtime and not has_times:
            sys.exit("CSV 必須包含 runtime_sec 或 (start_time + end_time)")

        for row in reader:
            jid = row["job_id"].strip()
            if not jid:
                continue
            step = row["step"].strip()

            if has_runtime and row.get("runtime_sec", "").strip():
                runtime = float(row["runtime_sec"])
                start = parse_time(row["start_time"]) if has_times else 0.0
                end = start + runtime
            else:
                start = parse_time(row["start_time"])
                end = parse_time(row["end_time"])
                runtime = end - start
                if runtime <= 0:
                    continue

            submit = None
            if has_submit and row.get("submit_time", "").strip():
                submit = parse_time(row["submit_time"])

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


# ===================================================================
# 2. Dispatch 分佈估算
# ===================================================================

def estimate_dispatch_distribution(jobs):
    """
    從資料中萃取 dispatch overhead 的分佈。
    有 submit_time → dispatch = start - submit
    沒有 → 用同 step job 之間的 start time 間距估算
    回傳 dispatch 值的 list (可能為空)
    """
    dispatches = []
    for j in jobs:
        if j["submit"] is not None:
            d = j["start"] - j["submit"]
            if d >= 0:
                dispatches.append(d)

    if not dispatches:
        starts = sorted(j["start"] for j in jobs)
        if len(starts) > 1:
            gaps = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
            dispatches = [g for g in gaps if g > 0]

    return dispatches


# ===================================================================
# 3. ILP 最優解
# ===================================================================

def solve_ilp(runtimes, m, time_limit=60):
    import pulp

    n = len(runtimes)
    if n == 0:
        return 0.0, "Optimal", {}

    prob = pulp.LpProblem("Pllcmax", pulp.LpMinimize)

    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    cmax = pulp.LpVariable("Cmax", lowBound=0)
    prob += cmax

    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(m)) == 1

    for j in range(m):
        prob += cmax >= pulp.lpSum(runtimes[i] * x[i, j] for i in range(n))

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    opt = None
    assignment = {}
    if status in ("Optimal", "Feasible"):
        opt = pulp.value(cmax)
        for i in range(n):
            for j in range(m):
                if pulp.value(x[i, j]) > 0.5:
                    assignment[i] = j
                    break

    return opt, status, assignment


# ===================================================================
# 4. 啟發式排程
# ===================================================================

def schedule_heuristic(runtimes, m, strategy="LPT"):
    n = len(runtimes)
    if n == 0:
        return 0.0, []

    indices = list(range(n))
    if strategy == "LPT":
        indices.sort(key=lambda i: -runtimes[i])
    elif strategy == "SPT":
        indices.sort(key=lambda i: runtimes[i])

    slot_loads = [0.0] * m
    for i in indices:
        j = min(range(m), key=lambda k: slot_loads[k])
        slot_loads[j] += runtimes[i]

    return max(slot_loads), slot_loads


# ===================================================================
# 5. Array vs Session 多維度比較
# ===================================================================

def simulate_array_distribution(runtimes, m, dispatch_samples, n_trials=200):
    """
    Array 模式模擬:
    - 每個 job 獨立排隊, dispatch 從實際分佈取樣
    - job 跑完立刻釋放 slot
    - 失敗的 job 只需重跑自己
    """
    if not dispatch_samples:
        dispatch_samples = [0.0]

    d_arr = np.array(dispatch_samples)
    r_arr = np.array(runtimes)
    n = len(runtimes)

    makespans = []
    slot_seconds_list = []
    for _ in range(n_trials):
        d = np.random.choice(d_arr, size=n, replace=True)
        effective = r_arr + d
        mk, loads = schedule_heuristic(effective.tolist(), m, "LPT")
        makespans.append(mk)
        # Array: 每個 job 用完就還, 實際佔用 = sum(effective)
        slot_seconds_list.append(float(np.sum(effective)))

    mk_median = float(np.median(makespans))
    mk_p25 = float(np.percentile(makespans, 25))
    mk_p75 = float(np.percentile(makespans, 75))
    actual_work = float(np.sum(r_arr))
    avg_slot_sec = float(np.mean(slot_seconds_list))

    return {
        "makespan_median": mk_median,
        "makespan_p25": mk_p25,
        "makespan_p75": mk_p75,
        "actual_work": actual_work,
        "slot_seconds": avg_slot_sec,            # 實際佔用 (job 用完即還)
        "wasted_slot_seconds": avg_slot_sec - actual_work,  # 浪費 = dispatch 等待
        "utilization": actual_work / avg_slot_sec if avg_slot_sec > 0 else 0,
    }


def simulate_session(runtimes, m):
    """
    Session 模式:
    - 一次 dispatch 拿到所有 slot, 持續佔用到最後一個 task 結束
    - 無 per-job dispatch overhead
    - 尾端 idle: 短 job 跑完的 slot 空等最長的 job
    """
    mk, loads = schedule_heuristic(runtimes, m, "LPT")
    actual_work = sum(runtimes)
    occupied = m * mk                            # session 佔住的總 slot-秒
    tail_idle = occupied - actual_work            # 尾端浪費
    utilization = actual_work / occupied if occupied > 0 else 0

    # 每個 slot 的 idle 時間
    slot_idles = [mk - load for load in loads]
    max_slot_idle = max(slot_idles) if slot_idles else 0

    return {
        "makespan": mk,
        "actual_work": actual_work,
        "slot_seconds": occupied,                # 佔用 = m × makespan (全鎖住)
        "wasted_slot_seconds": tail_idle,         # 浪費 = 尾端 idle
        "tail_idle": tail_idle,
        "max_slot_idle": max_slot_idle,            # 最閒那個 slot 浪費多少秒
        "utilization": utilization,
    }


def compare_array_session(runtimes, m, dispatch_samples, fail_rate=0.0):
    """
    多維度比較 array vs session, 不只看 makespan。

    維度:
      1. makespan          誰跑得快
      2. slot_seconds      誰佔叢集資源少 (影響 fairshare)
      3. tail_idle_penalty  CoV 造成的尾端浪費
      4. retry_cost        失敗重跑的代價差異
      5. dispatch_variance  dispatch 不穩定帶來的風險
      6. granularity       per-job 控制力 (log, 重跑, 資源)
    """
    arr = simulate_array_distribution(runtimes, m, dispatch_samples)
    ses = simulate_session(runtimes, m)

    r_arr = np.array(runtimes)
    n = len(runtimes)
    cov = float(np.std(r_arr) / np.mean(r_arr)) if np.mean(r_arr) > 0 else 0
    mean_rt = float(np.mean(r_arr))
    d_arr = np.array(dispatch_samples) if dispatch_samples else np.array([0.0])
    mean_dispatch = float(np.mean(d_arr))

    # --- 維度 1: Makespan ---
    mk_diff = arr["makespan_median"] - ses["makespan"]
    mk_winner = "session" if mk_diff > 0 else "array"

    # --- 維度 2: 叢集資源佔用 (slot-seconds) ---
    # Array: 用完即還, 佔用 ≈ sum(runtime + dispatch)
    # Session: 鎖住 m × makespan
    slot_diff = arr["slot_seconds"] - ses["slot_seconds"]
    slot_winner = "session" if slot_diff < 0 else "array"
    # 通常 session 佔更多, 因為它鎖住所有 slot

    # --- 維度 3: CoV 造成的尾端浪費 ---
    # Session 受 CoV 影響大: job 長度差異越大, 短 job 跑完的 slot 空等越久
    # Array 不受影響: 每個 job 獨立, 跑完就還
    # 量化: session 的 tail_idle / actual_work
    tail_idle_ratio = ses["tail_idle"] / ses["actual_work"] if ses["actual_work"] > 0 else 0
    cov_penalty = "session" if tail_idle_ratio > 0.2 else "neutral"

    # --- 維度 4: 失敗重跑代價 ---
    # Array: 只重跑失敗的那個 job, 代價 = mean_runtime × fail_rate × n
    # Session: 失敗處理更複雜, 最差要重跑整個 session
    #          保守估計: 重跑失敗 job + 重新分配的 overhead
    if fail_rate > 0 and n > 0:
        n_failures = max(1, int(n * fail_rate))
        # Array: 重跑 n_failures 個 job, 各自排隊
        array_retry_cost = n_failures * (mean_rt + mean_dispatch)
        # Session: 需要重新起一個小 session 跑失敗的 jobs
        session_retry_cost = n_failures * mean_rt + ses["max_slot_idle"] * 0.5
        retry_winner = "array" if array_retry_cost < session_retry_cost else "session"
        retry_diff = session_retry_cost - array_retry_cost
    else:
        array_retry_cost = 0.0
        session_retry_cost = 0.0
        retry_winner = "neutral"
        retry_diff = 0.0

    # --- 維度 5: Dispatch 變異風險 ---
    # Array 的 makespan 受 dispatch 變異影響, session 不受
    dispatch_cv = float(np.std(d_arr) / np.mean(d_arr)) if np.mean(d_arr) > 0 else 0
    mk_range = arr["makespan_p75"] - arr["makespan_p25"]
    dispatch_risk = "high" if dispatch_cv > 0.5 and mk_range > mean_rt * 0.1 else "low"

    # --- 維度 6: 顆粒度 ---
    # Array: per-job log, per-job 資源需求, per-job 重跑
    # Session: 失去這些, 換取省 dispatch
    # 這是定性指標, 不好量化, 用 flag 標記
    granularity_matters = n > 10 and cov > 0.5  # job 多且差異大時顆粒度更有價值

    return {
        "array": arr,
        "session": ses,
        "dimensions": {
            # 維度 1: Makespan
            "makespan": {
                "array": round(arr["makespan_median"], 2),
                "session": round(ses["makespan"], 2),
                "diff_sec": round(abs(mk_diff), 2),
                "winner": mk_winner,
            },
            # 維度 2: 資源佔用
            "resource_occupation": {
                "array_slot_sec": round(arr["slot_seconds"], 2),
                "session_slot_sec": round(ses["slot_seconds"], 2),
                "diff_slot_sec": round(abs(slot_diff), 2),
                "winner": slot_winner,
                "array_utilization": round(arr["utilization"], 4),
                "session_utilization": round(ses["utilization"], 4),
            },
            # 維度 3: CoV 尾端浪費
            "cov_tail_penalty": {
                "cov": round(cov, 4),
                "session_tail_idle": round(ses["tail_idle"], 2),
                "tail_idle_ratio": round(tail_idle_ratio, 4),
                "penalty_on": cov_penalty,
            },
            # 維度 4: 失敗重跑
            "retry_cost": {
                "fail_rate": fail_rate,
                "array_retry_cost": round(array_retry_cost, 2),
                "session_retry_cost": round(session_retry_cost, 2),
                "winner": retry_winner,
            },
            # 維度 5: Dispatch 風險
            "dispatch_risk": {
                "dispatch_cv": round(dispatch_cv, 4),
                "array_mk_iqr": round(mk_range, 2),
                "risk_level": dispatch_risk,
            },
            # 維度 6: 顆粒度
            "granularity": {
                "matters": granularity_matters,
                "reason": "job 多且 CoV 高, per-job 控制有價值" if granularity_matters
                          else "job 少或 CoV 低, 顆粒度影響小",
                "favors": "array" if granularity_matters else "neutral",
            },
        },
    }


# ===================================================================
# 6. 長短 job 切分分析
# ===================================================================

def analyze_long_short_per_step(runtimes, m, dispatch_samples):
    """
    對單一 step, 量化「長 job 用 array vs session」和「短 job 用 array vs session」
    的差異, 驗證「長 job → array 沒差, 短 job → session 有利」的假說。

    做法: 用 median runtime 把 job 分成長短兩群,
          各自用全部 M 個 slot 模擬 array 和 session, 比較差異。
    """
    if not runtimes or not dispatch_samples:
        return None

    r_arr = np.array(runtimes)
    median_rt = float(np.median(r_arr))

    short_jobs = r_arr[r_arr <= median_rt].tolist()
    long_jobs = r_arr[r_arr > median_rt].tolist()

    if not short_jobs or not long_jobs:
        return None

    # 短 job 群: array vs session
    m_short = min(m, len(short_jobs))
    short_arr = simulate_array_distribution(short_jobs, m_short, dispatch_samples, n_trials=200)
    short_ses = simulate_session(short_jobs, m_short)
    short_diff = short_arr["makespan_median"] - short_ses["makespan"]
    short_better = "session" if short_ses["makespan"] < short_arr["makespan_median"] else "array"

    # 長 job 群: array vs session
    m_long = min(m, len(long_jobs))
    long_arr = simulate_array_distribution(long_jobs, m_long, dispatch_samples, n_trials=200)
    long_ses = simulate_session(long_jobs, m_long)
    long_diff = long_arr["makespan_median"] - long_ses["makespan"]
    long_better = "session" if long_ses["makespan"] < long_arr["makespan_median"] else "array"

    mean_dispatch = float(np.mean(dispatch_samples)) if dispatch_samples else 0.0

    return {
        "median_runtime": round(median_rt, 2),
        "n_short": len(short_jobs),
        "n_long": len(long_jobs),
        "short_mean_rt": round(float(np.mean(short_jobs)), 2),
        "long_mean_rt": round(float(np.mean(long_jobs)), 2),
        "mean_dispatch": round(mean_dispatch, 2),
        # 短 job 群
        "short_array_mk": round(short_arr["makespan_median"], 2),
        "short_session_mk": round(short_ses["makespan"], 2),
        "short_diff_sec": round(short_diff, 2),
        "short_dispatch_ratio": round(mean_dispatch / float(np.mean(short_jobs)), 4)
                                 if np.mean(short_jobs) > 0 else 0,
        "short_better": short_better,
        # 長 job 群
        "long_array_mk": round(long_arr["makespan_median"], 2),
        "long_session_mk": round(long_ses["makespan"], 2),
        "long_diff_sec": round(long_diff, 2),
        "long_dispatch_ratio": round(mean_dispatch / float(np.mean(long_jobs)), 4)
                                if np.mean(long_jobs) > 0 else 0,
        "long_better": long_better,
    }


# ===================================================================
# 7. 特徵萃取
# ===================================================================

def extract_features(jobs, dispatch_samples):
    runtimes = [j["runtime"] for j in jobs]
    n = len(runtimes)
    if n == 0:
        return {}

    arr = np.array(runtimes)
    mean_rt = float(np.mean(arr))
    std_rt = float(np.std(arr))
    cov = std_rt / mean_rt if mean_rt > 0 else 0
    skew = float(np.mean(((arr - mean_rt) / std_rt) ** 3)) if std_rt > 0 else 0

    mean_dispatch = float(np.mean(dispatch_samples)) if dispatch_samples else 0.0
    dispatch_ratio = mean_dispatch / mean_rt if mean_rt > 0 else 0

    return {
        "n_jobs": n,
        "total_work": float(np.sum(arr)),
        "mean_runtime": mean_rt,
        "std_runtime": std_rt,
        "cov": cov,
        "skewness": skew,
        "min_runtime": float(np.min(arr)),
        "max_runtime": float(np.max(arr)),
        "range_ratio": float(np.max(arr) / np.min(arr)) if np.min(arr) > 0 else float("inf"),
        "mean_dispatch": mean_dispatch,
        "median_dispatch": float(np.median(dispatch_samples)) if dispatch_samples else 0.0,
        "dispatch_ratio": dispatch_ratio,
        # 長短 job 比例 (用 median runtime 切)
        "pct_short_jobs": float(np.sum(arr < np.median(arr))) / n,
        "short_mean_rt": float(np.mean(arr[arr <= np.median(arr)])),
        "long_mean_rt": float(np.mean(arr[arr > np.median(arr)])) if np.any(arr > np.median(arr)) else mean_rt,
    }


# ===================================================================
# 8. Decision Tree
# ===================================================================

def train_decision_tree(features_list, labels):
    from sklearn.tree import DecisionTreeClassifier, export_text

    if len(features_list) < 2:
        return None, [], "樣本太少 (至少需要 2 個 step)"

    feature_names = [
        "n_jobs", "mean_runtime", "cov", "skewness",
        "range_ratio", "dispatch_ratio",
        "pct_short_jobs", "mean_dispatch",
    ]

    X = np.array([[f.get(k, 0.0) for k in feature_names] for f in features_list])

    label_map = {"session": 1, "array": 0}
    y = np.array([label_map.get(l, 0) for l in labels])

    if len(np.unique(y)) < 2:
        majority = "session" if np.mean(y) > 0.5 else "array"
        return None, feature_names, f"所有 step 的最佳策略皆為 {majority}, 無法產出分類規則"

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1, random_state=42)
    clf.fit(X, y)

    rules = export_text(clf, feature_names=feature_names,
                        class_names=["array", "session"])
    return clf, feature_names, rules


# ===================================================================
# 9. 資源掃描
# ===================================================================

def resource_sweep(runtimes, m_range):
    if not runtimes:
        return []
    baseline = sum(runtimes)
    results = []
    for m in m_range:
        mk, _ = schedule_heuristic(runtimes, m, "LPT")
        speedup = baseline / mk if mk > 0 else 0
        efficiency = speedup / m if m > 0 else 0
        results.append((m, mk, speedup, efficiency))
    return results


# ===================================================================
# 10. 推斷 slot 數
# ===================================================================

def infer_slots(jobs):
    events = []
    for j in jobs:
        events.append((j["start"], 1))
        events.append((j["end"], -1))
    events.sort()
    peak, cur = 0, 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return max(peak, 1)


# ===================================================================
# 11. 主流程
# ===================================================================

def analyze_step(step_name, jobs, m, ilp_time_limit=60, fail_rate=0.02):
    runtimes = [j["runtime"] for j in jobs]
    n = len(runtimes)
    if n == 0:
        return None

    dispatch_samples = estimate_dispatch_distribution(jobs)
    feat = extract_features(jobs, dispatch_samples)

    # --- Lower Bound (理想, 無 overhead) ---
    lb = max(max(runtimes), sum(runtimes) / m)

    # --- ILP ---
    ilp_mk, ilp_status, _ = solve_ilp(runtimes, m, ilp_time_limit)

    # --- 啟發式 ---
    lpt_mk, _ = schedule_heuristic(runtimes, m, "LPT")
    spt_mk, _ = schedule_heuristic(runtimes, m, "SPT")
    fifo_mk, _ = schedule_heuristic(runtimes, m, "FIFO")

    best_heuristic = min(
        [("LPT", lpt_mk), ("SPT", spt_mk), ("FIFO", fifo_mk)],
        key=lambda x: x[1]
    )

    # --- Array vs Session 多維度比較 ---
    comparison = compare_array_session(runtimes, m, dispatch_samples, fail_rate)
    dims = comparison["dimensions"]

    # 綜合判斷: 計算各維度的 winner
    scores = {"array": 0, "session": 0}
    for dim_name, dim_data in dims.items():
        w = dim_data.get("winner") or dim_data.get("favors") or dim_data.get("penalty_on")
        if w == "array":
            scores["array"] += 1
        elif w == "session":
            scores["session"] += 1

    better = "session" if scores["session"] > scores["array"] else "array"
    if scores["session"] == scores["array"]:
        # 平手時以 makespan 為主
        better = dims["makespan"]["winner"]

    # --- 長短 job 切分分析 ---
    ls_analysis = analyze_long_short_per_step(runtimes, m, dispatch_samples)

    result = {
        "step": step_name,
        "n_jobs": n,
        "m_slots": m,
        # 下界
        "lower_bound": round(lb, 2),
        # ILP
        "ilp_makespan": round(ilp_mk, 2) if ilp_mk else None,
        "ilp_status": ilp_status,
        # 啟發式
        "lpt_makespan": round(lpt_mk, 2),
        "spt_makespan": round(spt_mk, 2),
        "fifo_makespan": round(fifo_mk, 2),
        "best_heuristic": best_heuristic[0],
        "best_heuristic_mk": round(best_heuristic[1], 2),
        "lpt_vs_ilp": f"{(lpt_mk / ilp_mk - 1) * 100:.1f}%" if ilp_mk else "N/A",
        # Array vs Session 多維度
        "better_strategy": better,
        "array_score": scores["array"],
        "session_score": scores["session"],
        # 維度 1: Makespan
        "mk_array": dims["makespan"]["array"],
        "mk_session": dims["makespan"]["session"],
        "mk_winner": dims["makespan"]["winner"],
        # 維度 2: 資源佔用
        "slot_sec_array": dims["resource_occupation"]["array_slot_sec"],
        "slot_sec_session": dims["resource_occupation"]["session_slot_sec"],
        "slot_winner": dims["resource_occupation"]["winner"],
        # 維度 3: CoV 尾端
        "cov": round(feat["cov"], 4),
        "tail_idle_ratio": dims["cov_tail_penalty"]["tail_idle_ratio"],
        "cov_penalty_on": dims["cov_tail_penalty"]["penalty_on"],
        # 維度 4: 失敗重跑
        "retry_winner": dims["retry_cost"]["winner"],
        # 維度 5: Dispatch 風險
        "dispatch_risk": dims["dispatch_risk"]["risk_level"],
        # 維度 6: 顆粒度
        "granularity_favors": dims["granularity"]["favors"],
        # 特徵
        "mean_runtime": round(feat["mean_runtime"], 2),
        "dispatch_ratio": round(feat["dispatch_ratio"], 4),
        "range_ratio": round(feat["range_ratio"], 2),
    }
    return result, feat, better, ls_analysis, comparison


def main():
    ap = argparse.ArgumentParser(description="LSF 排程分析框架 v2")
    ap.add_argument("csv", help="輸入的 job CSV 檔")
    ap.add_argument("--slots", type=int, default=0,
                    help="每個 step 使用的 slot 數 (預設: 從資料推斷)")
    ap.add_argument("--ilp-timeout", type=int, default=60,
                    help="ILP solver 時間限制 (秒, 預設 60)")
    ap.add_argument("--sweep", action="store_true",
                    help="執行資源量掃描")
    ap.add_argument("-o", "--output", default="step_analysis.csv",
                    help="分析結果輸出 (預設 step_analysis.csv)")
    ap.add_argument("--rules", default="rules.txt",
                    help="Decision Tree 規則輸出 (預設 rules.txt)")
    ap.add_argument("--sweep-out", default="resource_sweep.csv",
                    help="資源掃描輸出 (預設 resource_sweep.csv)")
    ap.add_argument("--fail-rate", type=float, default=0.02,
                    help="Job 失敗率, 用於估算重跑代價 (預設 0.02 = 2%%)")
    ap.add_argument("--long-short-out", default="long_short_analysis.csv",
                    help="長短 job 切分分析輸出")
    args = ap.parse_args()

    # --- 載入 ---
    steps_data = load_csv(args.csv)
    total_jobs = sum(len(v) for v in steps_data.values())
    print(f"[i] 載入 {total_jobs} 個 job, {len(steps_data)} 個 step\n")

    m_global = args.slots

    # --- 逐 step 分析 ---
    all_results = []
    all_features = []
    all_labels = []
    all_ls = []

    step_keys = sorted(steps_data.keys(),
                       key=lambda s: (int(s) if s.isdigit() else float('inf'), s))

    hdr = (f"{'step':>10} {'jobs':>5} {'slots':>5} {'LB':>9} {'ILP':>9} "
           f"{'LPT':>9} {'best':>5} {'A vs S':>8} {'better':>8} {'score':>8}")
    print(hdr)
    print("-" * len(hdr))

    all_comparisons = []

    for step_name in step_keys:
        jobs = steps_data[step_name]
        m = m_global if m_global > 0 else infer_slots(jobs)
        m = max(m, 1)

        out = analyze_step(step_name, jobs, m, args.ilp_timeout, args.fail_rate)
        if out is None:
            continue
        result, feat, better, ls_analysis, comparison = out

        all_results.append(result)
        all_features.append(feat)
        all_labels.append(better)
        all_comparisons.append((step_name, comparison))
        if ls_analysis:
            ls_analysis["step"] = step_name
            all_ls.append(ls_analysis)

        ilp_str = f"{result['ilp_makespan']:.1f}" if result['ilp_makespan'] else "N/A"
        score_str = f"A{result['array_score']}:S{result['session_score']}"
        print(f"{step_name:>10} {result['n_jobs']:>5} {m:>5} "
              f"{result['lower_bound']:>9.1f} {ilp_str:>9} "
              f"{result['lpt_makespan']:>9.1f} "
              f"{result['best_heuristic']:>5} "
              f"{result['mk_array']:>9.1f}|{result['mk_session']:<.1f} "
              f"{result['better_strategy']:>8} "
              f"{score_str:>8}")

    # --- 輸出分析結果 ---
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
        print(f"\n[✓] {args.output}")

    # --- 多維度比較明細 ---
    if all_comparisons:
        print(f"\n{'='*70}")
        print("Array vs Session 多維度比較")
        print(f"{'='*70}")
        for step_name, comp in all_comparisons:
            dims = comp["dimensions"]
            print(f"\n  step {step_name}:")
            print(f"    {'維度':<20} {'Array':>12} {'Session':>12} {'較優':>10}")
            print(f"    {'─'*56}")
            # 1. Makespan
            print(f"    {'Makespan (秒)':<20} "
                  f"{dims['makespan']['array']:>12.1f} "
                  f"{dims['makespan']['session']:>12.1f} "
                  f"{dims['makespan']['winner']:>10}")
            # 2. 資源佔用
            print(f"    {'資源佔用 (slot-秒)':<20} "
                  f"{dims['resource_occupation']['array_slot_sec']:>12.1f} "
                  f"{dims['resource_occupation']['session_slot_sec']:>12.1f} "
                  f"{dims['resource_occupation']['winner']:>10}")
            # 3. CoV 尾端
            idle_r = dims['cov_tail_penalty']['tail_idle_ratio']
            print(f"    {'CoV尾端浪費':<20} "
                  f"{'N/A':>12} "
                  f"{idle_r:>11.1%} "
                  f"{'session' if dims['cov_tail_penalty']['penalty_on']=='session' else 'neutral':>10}")
            # 4. 失敗重跑
            print(f"    {'失敗重跑代價':<20} "
                  f"{dims['retry_cost']['array_retry_cost']:>12.1f} "
                  f"{dims['retry_cost']['session_retry_cost']:>12.1f} "
                  f"{dims['retry_cost']['winner']:>10}")
            # 5. Dispatch 風險
            print(f"    {'Dispatch 風險':<20} "
                  f"{dims['dispatch_risk']['risk_level']:>12} "
                  f"{'N/A':>12} "
                  f"{'session' if dims['dispatch_risk']['risk_level']=='high' else 'neutral':>10}")
            # 6. 顆粒度
            print(f"    {'顆粒度需求':<20} "
                  f"{'有價值' if dims['granularity']['matters'] else '影響小':>12} "
                  f"{'─':>12} "
                  f"{dims['granularity']['favors']:>10}")

    # --- 長短 job 切分分析 ---
    if all_ls:
        ls_fields = list(all_ls[0].keys())
        with open(args.long_short_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ls_fields)
            w.writeheader()
            w.writerows(all_ls)

        print(f"\n{'='*70}")
        print("長短 Job 切分分析: 驗證「長 job → array 沒差, 短 job → session 有利」")
        print(f"{'='*70}")
        print(f"\n  {'step':>10} │ {'短job群':^32} │ {'長job群':^32}")
        print(f"  {'':>10} │ {'n':>4} {'avg_rt':>7} {'d/rt':>6} "
              f"{'arr_mk':>7} {'ses_mk':>7} {'better':>7} │ "
              f"{'n':>4} {'avg_rt':>7} {'d/rt':>6} "
              f"{'arr_mk':>7} {'ses_mk':>7} {'better':>7}")
        print(f"  {'─'*10}─┼─{'─'*32}─┼─{'─'*32}")
        for r in all_ls:
            print(f"  {r['step']:>10} │ "
                  f"{r['n_short']:>4} {r['short_mean_rt']:>7.1f} "
                  f"{r['short_dispatch_ratio']:>6.3f} "
                  f"{r['short_array_mk']:>7.1f} {r['short_session_mk']:>7.1f} "
                  f"{r['short_better']:>7} │ "
                  f"{r['n_long']:>4} {r['long_mean_rt']:>7.1f} "
                  f"{r['long_dispatch_ratio']:>6.3f} "
                  f"{r['long_array_mk']:>7.1f} {r['long_session_mk']:>7.1f} "
                  f"{r['long_better']:>7}")

        # 摘要
        short_session_wins = sum(1 for r in all_ls if r["short_better"] == "session")
        long_array_wins = sum(1 for r in all_ls if r["long_better"] == "array")
        total = len(all_ls)
        print(f"\n  摘要: 短 job 群中 session 較優: {short_session_wins}/{total} 個 step "
              f"({short_session_wins/total*100:.0f}%)")
        print(f"        長 job 群中 array  較優: {long_array_wins}/{total} 個 step "
              f"({long_array_wins/total*100:.0f}%)")
        if short_session_wins > total * 0.6 and long_array_wins > total * 0.4:
            print(f"  → 資料支持「短 job → session, 長 job → array」的假說")
        elif short_session_wins > total * 0.6:
            print(f"  → 資料支持短 job 用 session, 但長 job 也傾向 session (dispatch 影響不大)")
        else:
            print(f"  → 資料未明確支持長短分流, 需進一步調查")

        print(f"\n[✓] {args.long_short_out}")

    # --- Decision Tree ---
    print(f"\n{'='*70}")
    print("Decision Tree 規則萃取")
    print(f"{'='*70}")
    if len(all_features) >= 2:
        clf, fnames, rules_text = train_decision_tree(all_features, all_labels)
        print(rules_text)
        with open(args.rules, "w", encoding="utf-8") as f:
            f.write("LSF Scheduler Analysis - Decision Tree Rules\n")
            f.write("=" * 60 + "\n\n")
            f.write("特徵說明:\n")
            f.write("  n_jobs         : job 數量\n")
            f.write("  mean_runtime   : 平均 runtime (秒)\n")
            f.write("  cov            : 變異係數 (std/mean)\n")
            f.write("  skewness       : 偏度\n")
            f.write("  range_ratio    : 最長/最短 job 比值\n")
            f.write("  dispatch_ratio : dispatch overhead / mean runtime\n")
            f.write("  pct_short_jobs : 短 job 佔比\n")
            f.write("  mean_dispatch  : 平均 dispatch 時間 (秒)\n\n")
            f.write("規則:\n")
            f.write(rules_text)
        print(f"[✓] {args.rules}")
    else:
        print("只有 1 個 step, 跳過 Decision Tree")

    # --- 資源掃描 ---
    if args.sweep:
        print(f"\n{'='*70}")
        print("資源量掃描")
        print(f"{'='*70}")
        sweep_rows = []
        for step_name in step_keys:
            jobs = steps_data[step_name]
            runtimes = [j["runtime"] for j in jobs]
            m_inferred = m_global if m_global > 0 else infer_slots(jobs)
            m_max = min(len(runtimes), max(m_inferred * 3, 20))

            results = resource_sweep(runtimes, range(1, m_max + 1))
            for m_val, mk, sp, eff in results:
                sweep_rows.append({
                    "step": step_name, "slots": m_val,
                    "makespan": round(mk, 2), "speedup": round(sp, 3),
                    "efficiency": round(eff, 4),
                })

            # 找邊際效益拐點
            if len(results) >= 2:
                prev_mk = results[0][1]
                for i, (m_val, mk, sp, eff) in enumerate(results[1:], 1):
                    improvement = (prev_mk - mk) / prev_mk if prev_mk > 0 else 0
                    if improvement < 0.03 and m_val >= m_inferred:
                        print(f"  step {step_name}: slot={m_val} 後邊際效益 <3% "
                              f"(makespan {mk:.0f}s, {eff:.0%} 效率)")
                        break
                    prev_mk = mk

        with open(args.sweep_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "slots", "makespan",
                                               "speedup", "efficiency"])
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"[✓] {args.sweep_out}")

    print("\n完成。")


if __name__ == "__main__":
    main()
