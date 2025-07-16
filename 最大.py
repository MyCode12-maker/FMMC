import re
from pathlib import Path

# 日志文件路径（请替换为你的路径）
log_path = Path("参数/32-6.txt")

# 正则：提取 k、Recall 和 NDCG（如 k=5、Recall@5: xxx, NDCG@5: xxx）
pattern = re.compile(
    r"Fused\s+Recall@(\d+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),?"
    r".*?NDCG@\1[:\s]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:,|\s|$)",
    re.IGNORECASE
)

# 我们只关注的 k 值
target_ks = {5, 10}

# 为每个 k 分别记录 Recall 最大和 NDCG 最大的行 (k → (recall, ndcg))
max_recall_dict = {}
max_ndcg_dict = {}

with log_path.open(encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            k = int(match.group(1))
            if k not in target_ks:
                continue  # 忽略不是目标的 k

            recall = float(match.group(2).rstrip(','))
            ndcg = float(match.group(3).rstrip(','))

            # 更新 Recall 最大值
            if k not in max_recall_dict or recall > max_recall_dict[k][0]:
                max_recall_dict[k] = (recall, ndcg)

            # 更新 NDCG 最大值
            if k not in max_ndcg_dict or ndcg > max_ndcg_dict[k][1]:
                max_ndcg_dict[k] = (recall, ndcg)

# 输出
print("===== 每个 k 的 Recall 最大和 NDCG 最大统计 =====")
for k in sorted(target_ks):
    print(f"\n🔹 k = {k}")

    if k in max_recall_dict:
        rec, ndcg = max_recall_dict[k]
        print(f"  📈 最大 Recall: {rec:.6f}  ")
    else:
        print("  最大 Recall: 无记录")

    if k in max_ndcg_dict:
        rec, ndcg = max_ndcg_dict[k]
        print(f"  📈 最大 NDCG  : {ndcg:.6f} ")
    else:
        print("  最大 NDCG: 无记录")
