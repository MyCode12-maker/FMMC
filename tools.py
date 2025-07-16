import torch
import pickle
from collections import defaultdict
from random import random
import settings
import pandas as pd
city = settings.city
if settings.enable_ssl and settings.enable_distance_sample:
    df_farthest_POIs = pd.read_csv(f"./raw_data/{city}_farthest_POIs.csv")
device = settings.gpuId if torch.cuda.is_available() else 'cpu'

def extract_poi_info_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        train_set = pickle.load(f)

    poi_info_dict = {}

    for user_seq in train_set:
        for visit_seq in user_seq:
            poi_list = visit_seq[0]
            cat_list = visit_seq[1]

            for poi, cat in zip(poi_list, cat_list):
                if poi not in poi_info_dict:
                    poi_info_dict[poi] = {"category": cat}
                elif poi_info_dict[poi]["category"] != cat:
                    print(f"警告：POI {poi} 出现多个类别：{poi_info_dict[poi]['category']} vs {cat}")

    return poi_info_dict


def average_dicts(dicts):
    avg_dict = defaultdict(float)
    count_dict = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            avg_dict[k] += v
            count_dict[k] += 1
    return {k: avg_dict[k] / count_dict[k] for k in avg_dict}

def hit_aware_vote(preds_list, item_hit_rate, top_k):
    """
    preds_list: List of [B, K] 推荐ID（多个模型输出）
    item_hit_rate: dict, 每个item的历史命中率（如Recall@k结果累积）
    """
    B = preds_list[0].shape[0]
    vote_scores = []

    for b in range(B):
        counter = defaultdict(float)
        for preds in preds_list:
            for rank, item in enumerate(preds[b]):
                weight = item_hit_rate.get(item.item(), 0.5)  # 默认0.5中性
                counter[item.item()] += (1.0 / (rank + 1)) * weight  # 排名越前贡献越大
        # 重新按 vote 分数排序
        top_items = sorted(counter.items(), key=lambda x: -x[1])[:top_k]
        vote_scores.append([item for item, _ in top_items])

    return torch.tensor(vote_scores)  # [B, top_k]
def compute_item_hit_rate(preds, labels, top_k):
    """
    preds: Tensor of shape [B, K]，每个用户的 top-K 推荐列表（已按 ID 排序）
    labels: Tensor of shape [B, 1]，每个用户的真实目标 item
    """
    item_hit_count = defaultdict(int)
    item_total_count = defaultdict(int)

    B = preds.size(0)

    for i in range(B):
        label = labels[i].item()
        for rank, item in enumerate(preds[i][:top_k]):
            item = item.item()
            item_total_count[item] += 1
            if item == label:
                item_hit_count[item] += 1

    # 构建命中率字典
    item_hit_rate = {}
    for item in item_total_count:
        hit = item_hit_count[item]
        total = item_total_count[item]
        item_hit_rate[item] = hit / (total + 1e-8)

    return item_hit_rate

def generate_positive_sample_by_category_all_v2(dataset, user_id, poi_info_dict, k=10, device='cuda'):
    strong_aug_samples = []
    weak_aug_samples = []

    user_sequences = []
    for user_days in dataset:
        for day_seq in user_days:
            if len(day_seq) >= 3 and len(day_seq[2]) > 0 and day_seq[2][0] == user_id:
                user_sequences.append(day_seq)

    if not user_sequences:
        print(f"[Warn] No sequences found for user {user_id}")
        return [], []

    visited_pois = set()
    for seq in user_sequences:
        visited_pois.update(seq[0])

    for original_seq in user_sequences:
        current_seq_pois = set(original_seq[0])
        category_to_indices = {}

        for idx, poi in enumerate(original_seq[0]):
            category = poi_info_dict.get(poi, {}).get("category", None)
            if category:
                category_to_indices.setdefault(category, []).append(idx)

        for category, indices in category_to_indices.items():
            candidate_pois = [
                poi for poi in visited_pois
                if poi_info_dict.get(poi, {}).get("category") == category and poi not in current_seq_pois
            ]

            if len(candidate_pois) < k:
                additional = [
                    poi for poi in visited_pois
                    if poi_info_dict.get(poi, {}).get("category") == category
                ]
                candidate_pois.extend(additional)

            candidate_pois = list(set(candidate_pois))

            if not candidate_pois:
                continue

            sampled_pois = random.sample(candidate_pois, min(k, len(candidate_pois)))

            for neg_poi in sampled_pois:
                fake_seq = copy.deepcopy(original_seq)
                for idx in indices:
                    fake_seq[0][idx] = neg_poi

                neg_tensor = generate_day_sample_to_device(fake_seq)
                if neg_tensor is not None:
                    # 判断替换的数量是否 >= 序列长度的一半
                    if len(indices) >= len(original_seq[0]) / 2:
                        weak_aug_samples.append(neg_tensor)
                    else:
                        strong_aug_samples.append(neg_tensor)

    return strong_aug_samples, weak_aug_samples


from tqdm import tqdm
import torch


def build_and_save_all_negatives_v2(dataset, poi_info_dict, k=5,
                                    strong_save_path='PHO_pos_samples_strong.pt',
                                    weak_save_path='PHO_pos_samples_weak.pt'):
    all_user_ids = set()

    for user_days in dataset:
        for seq in user_days:
            if len(seq) >= 3 and len(seq[2]) > 0:
                all_user_ids.add(seq[2][0])

    all_strong_neg_samples = {}
    all_weak_neg_samples = {}

    for uid in tqdm(all_user_ids, desc="Generating negatives (strong/weak)"):
        strong_neg, weak_neg = generate_positive_sample_by_category_all_v2(
            dataset=dataset,
            user_id=uid,
            poi_info_dict=poi_info_dict,
            k=k,
            device='cpu'  # 保存为CPU数据节省显存
        )
        if strong_neg:
            all_strong_neg_samples[uid] = strong_neg
        if weak_neg:
            all_weak_neg_samples[uid] = weak_neg

    torch.save(all_strong_neg_samples, strong_save_path)
    print(f"[✓] 强增强样本已保存至 {strong_save_path}")

    torch.save(all_weak_neg_samples, weak_save_path)
    print(f"[✓] 弱增强样本已保存至 {weak_save_path}")

def generate_sample_to_device(sample):
    sample_to_device = []
    if settings.enable_dynamic_day_length:
        last_day = sample[-1][5][0]
        for seq in sample:
            seq_day = seq[5][0]
            if last_day - seq_day < settings.sample_day_length:
                features = torch.tensor(seq[:5]).to(device)
                day_nums = torch.tensor(seq[5]).to(device)
                sample_to_device.append((features, day_nums))
    else:
        for seq in sample:
            features = torch.tensor(seq[:5]).to(device)
            day_nums = torch.tensor(seq[5]).to(device)
            sample_to_device.append((features, day_nums))

    return sample_to_device

def generate_day_sample_to_device(day_trajectory):
    features = torch.tensor(day_trajectory[:5]).to(device)
    day_nums = torch.tensor(day_trajectory[5]).to(device)
    day_to_device = (features, day_nums)
    return day_to_device

def generate_negative_sample_list(dataset, user_id, current_POI):
    k = settings.neg_sample_count
    neg_day_sample_to_device_list = []
    if settings.enable_distance_sample:
        # Random sample k negative samples from other users' trajectories
        # and the negative samples contain the farthest POIs from current POI
        farthest_POIs = set(df_farthest_POIs.iloc[current_POI].values.tolist())
        eligible_sequences = []
        for seq in dataset:
            if seq[0][2][0] != user_id:
                for i in range(len(seq)):
                    if set(seq[i][0]).intersection(farthest_POIs):
                        eligible_sequences.append(seq[i])

        if len(eligible_sequences) == 0:
            print(f'Can not find eligible_sequences, current POI {current_POI}')
        elif len(eligible_sequences) < k:
            neg_day_samples = eligible_sequences
        else:
            neg_day_samples = random.sample(eligible_sequences, k)
    else:
        # Random sample k negative samples from other users' trajectories
        neg_day_samples = random.sample([seq[-1] for seq in dataset if seq[0][2][0] != user_id], k)

    for neg_day_sample in neg_day_samples:
        neg_day_sample_to_device_list.append(generate_day_sample_to_device(neg_day_sample))
    return neg_day_sample_to_device_list

def recursive_to_tensor(data, device='cuda'):
    if isinstance(data, (list, tuple)):
        converted = [recursive_to_tensor(x, device) for x in data]
        return tuple(converted) if isinstance(data, tuple) else converted
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.tensor(data).to(device)

def generate_negative_sample_by_category_int(dataset, user_id, current_POI, poi_info_dict, k=5, device='cuda'):
    neg_day_sample_to_device_list = []

    # 找到用户访问过的序列
    user_sequences = []
    for user_days in dataset:
        for day_seq in user_days:
            if len(day_seq) >= 3 and len(day_seq[2]) > 0 and day_seq[2][0] == user_id:
                user_sequences.append(day_seq)

    # 找到包含 current_POI 的那条序列
    current_POI_sequence = None
    for seq in user_sequences:
        if current_POI in seq[0]:
            current_POI_sequence = seq
            break

    if current_POI_sequence is None:
        print(f"[Warn] Cannot find sequence with POI {current_POI} for user {user_id}, generating random negative sample")

        all_sequences = []
        for user_days in dataset:
            for day_seq in user_days:
                if len(day_seq) >= 3:
                    all_sequences.append(day_seq)

        if not all_sequences:
            print("[Warn] Dataset is empty, cannot generate negative sample")
            return []

        random_seq = random.choice(all_sequences)
        neg_tensor = generate_day_sample_to_device(random_seq)
        return [neg_tensor] if neg_tensor is not None else []

    # 当前 POI 类别
    current_category = poi_info_dict.get(current_POI, {}).get("category", None)
    if current_category is None:
        print(f"[Warn] Cannot find category for POI {current_POI}")
        return []

    # 找到所有同类但用户未访问过的 POI
    visited_pois = set()
    for seq in user_sequences:
        visited_pois.update(seq[0])  # 所有 POI

    candidate_pois = [poi for poi, info in poi_info_dict.items()
                      if info.get("category") == current_category and poi not in visited_pois]

    if not candidate_pois:
        print(f"[Warn] No unvisited POIs in category {current_category} for user {user_id}")
        return []

    sampled_pois = random.sample(candidate_pois, min(k, len(candidate_pois)))

    # 找出原序列中所有 current_category 类别的 POI 索引位置
    poi_indices_to_replace = [i for i, poi in enumerate(current_POI_sequence[0])
                              if poi_info_dict.get(poi, {}).get("category", None) == current_category]

    if not poi_indices_to_replace:
        print(f"[Warn] No POIs in sequence belong to category {current_category}")
        return []

    # 生成多个负样本序列
    for neg_poi in sampled_pois:
        fake_seq = copy.deepcopy(current_POI_sequence)
        for idx in poi_indices_to_replace:
            fake_seq[0][idx] = neg_poi
        neg_tensor = generate_day_sample_to_device(fake_seq)
        if neg_tensor is not None:
            neg_day_sample_to_device_list.append(neg_tensor)

    return neg_day_sample_to_device_list

import random
import copy

def generate_negative_sample_by_category_all(dataset, user_id, poi_info_dict, k=10, device='cuda'):
    neg_day_sample_to_device_list = []

    # 获取用户所有访问序列
    user_sequences = []
    for user_days in dataset:
        for day_seq in user_days:
            if len(day_seq) >= 3 and len(day_seq[2]) > 0 and day_seq[2][0] == user_id:
                user_sequences.append(day_seq)

    if not user_sequences:
        print(f"[Warn] No sequences found for user {user_id}")
        return []

    # 获取用户访问过的 POI
    visited_pois = set()
    for seq in user_sequences:
        visited_pois.update(seq[0])

    for original_seq in user_sequences:
        # 统计该序列中出现的 POI 类别 -> 所有该类别在序列中的索引
        category_to_indices = {}
        for idx, poi in enumerate(original_seq[0]):
            category = poi_info_dict.get(poi, {}).get("category", None)
            if category:
                category_to_indices.setdefault(category, []).append(idx)

        # 遍历每个类别，生成负样本
        for category, indices in category_to_indices.items():
            # 候选 POI：未访问过的同类 POI
            candidate_pois = [poi for poi, info in poi_info_dict.items()
                              if info.get("category") == category and poi not in visited_pois]

            if not candidate_pois:
                continue  # skip this category if no valid negative samples

            sampled_pois = random.sample(candidate_pois, min(k, len(candidate_pois)))

            for neg_poi in sampled_pois:
                fake_seq = copy.deepcopy(original_seq)
                for idx in indices:
                    fake_seq[0][idx] = neg_poi  # 替换同类 POI
                neg_tensor = generate_day_sample_to_device(fake_seq)
                if neg_tensor is not None:
                    neg_day_sample_to_device_list.append(neg_tensor)

    return neg_day_sample_to_device_list

def generate_postive_sample_by_category_all(dataset, user_id, poi_info_dict, k=10, device='cuda'):
    neg_day_sample_to_device_list = []

    # 获取用户所有访问序列
    user_sequences = []
    for user_days in dataset:
        for day_seq in user_days:
            if len(day_seq) >= 3 and len(day_seq[2]) > 0 and day_seq[2][0] == user_id:
                user_sequences.append(day_seq)

    if not user_sequences:
        print(f"[Warn] No sequences found for user {user_id}")
        return []

    # 获取用户访问过的 POI
    visited_pois = set()
    for seq in user_sequences:
        visited_pois.update(seq[0])

    for original_seq in user_sequences:
        # 当前序列中出现的 POI
        current_seq_pois = set(original_seq[0])

        # 统计该序列中出现的 POI 类别 -> 所有该类别在序列中的索引
        category_to_indices = {}
        for idx, poi in enumerate(original_seq[0]):
            category = poi_info_dict.get(poi, {}).get("category", None)
            if category:
                category_to_indices.setdefault(category, []).append(idx)

        for category, indices in category_to_indices.items():
            # 候选1：用户访问过 & 同类 & 不在当前序列中
            candidate_pois = [
                poi for poi in visited_pois
                if poi_info_dict.get(poi, {}).get("category") == category and poi not in current_seq_pois
            ]

            # 若不足 k 个，再补充：用户访问过 & 同类 & 不等于当前 POI（即便当前序列已出现）
            if len(candidate_pois) < k:
                additional = [
                    poi for poi in visited_pois
                    if poi_info_dict.get(poi, {}).get("category") == category and poi not in indices
                ]
                candidate_pois.extend(additional)

            # 去重
            candidate_pois = list(set(candidate_pois))

            if not candidate_pois:
                continue

            sampled_pois = random.sample(candidate_pois, min(k, len(candidate_pois)))

            for neg_poi in sampled_pois:
                fake_seq = copy.deepcopy(original_seq)
                for idx in indices:
                    fake_seq[0][idx] = neg_poi
                neg_tensor = generate_day_sample_to_device(fake_seq)
                if neg_tensor is not None:
                    neg_day_sample_to_device_list.append(neg_tensor)

    return neg_day_sample_to_device_list