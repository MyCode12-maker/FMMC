import datetime
import numpy as np
import time
import os
import pandas as pd
import settings
from FMMC import FMMC
from results.data_reader import print_output_to_file, calculate_average, clear_log_meta_model
from tools import *
device = settings.gpuId if torch.cuda.is_available() else 'cpu'
city = settings.city

if settings.enable_ssl and settings.enable_distance_sample:
    df_farthest_POIs = pd.read_csv(f"./raw_data/{city}_farthest_POIs.csv")

def train_model(train_set, test_set, h_params, vocab_size, device,run_name):
    torch.cuda.empty_cache()
    model_path = f"./results/{run_name}_model"
    log_path = f"./results/{run_name}_log"
    meta_path = f"./results/{run_name}_meta"

    print("parameters:", h_params)

    if os.path.isfile(f'./results/{run_name}_model'):
        try:
            os.remove(f"./results/{run_name}_meta")
            os.remove(f"./results/{run_name}_model")
            os.remove(f"./results/{run_name}_log")
        except OSError:
            pass
    file = open(log_path, 'wb')
    pickle.dump(h_params, file)
    file.close()
    # construct model
    rec_model = FMMC(
        vocab_size=vocab_size,
        f_embed_size=h_params['embed_size'],
        num_encoder_layers=h_params['tfp_layer_num'],
        num_lstm_layers=h_params['lstm_layer_num'],
        num_heads=h_params['head_num'],
        forward_expansion=h_params['expansion'],
        dropout_p=h_params['dropout']
    )

    rec_model = rec_model.to(device)

    # Continue with previous training
    start_epoch = 0
    if os.path.isfile(model_path):
        rec_model.load_state_dict(torch.load(model_path))
        rec_model.train()

        meta_file = open(meta_path, "rb")
        start_epoch = pickle.load(meta_file) + 1
        meta_file.close()

    params = list(rec_model.parameters())

    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}
    strong_pos_samples = torch.load('PHO_pos_samples_strong.pt')
    weak_pos_samples = torch.load('PHO_pos_samples_weak.pt')
    all_neg_samples = torch.load("PHO_pos_samples_strong.pt")

    for epoch in range(start_epoch, h_params['epoch']):
        begin_time = time.time()
        total_loss = 0.

        # 根据训练进度计算增强比例
        progress_ratio = epoch / h_params['epoch']
        if progress_ratio < 0.3:
            strong_ratio = 0.1
        elif progress_ratio < 0.7:
            strong_ratio = 0.5
        else:
            strong_ratio = 0.8

        weak_ratio = 1 - strong_ratio
        for sample in train_set:
            sample_to_device = generate_sample_to_device(sample)
            neg_sample_to_device_list = []
            if settings.enable_ssl:
                user_id = sample[0][2][0]
                current_POI = sample[-1][0][-2]
                neg_sample_to_device_list = generate_negative_sample_list(train_set, user_id, current_POI)

                strong_pos = strong_pos_samples.get(user_id, [])
                weak_pos = weak_pos_samples.get(user_id, [])

                strong_num = int(strong_ratio * 10)
                weak_num = 10 - strong_num

                selected_pos = []
                if strong_pos:
                    selected_pos += random.sample(strong_pos, min(strong_num, len(strong_pos)))
                if weak_pos:
                    selected_pos += random.sample(weak_pos, min(weak_num, len(weak_pos)))
                user_neg_samples = all_neg_samples.get(user_id, [])
                if len(user_neg_samples) == 0:
                    all_available_negatives = []
                    for neg_list in all_neg_samples.values():
                        all_available_negatives.extend(neg_list)

                    selected_neg_samples = random.sample(all_available_negatives, min(5, len(all_available_negatives)))
                else:
                    # 随机选 k 个
                    k = 10
                    selected_neg_samples = random.sample(user_neg_samples, min(k, len(user_neg_samples)))

            loss, _ ,_,_,final_att= rec_model(sample_to_device, neg_sample_to_device_list,selected_neg_samples)
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Test
        recall, ndcg, map = test_model(test_set, rec_model,epoch)
        recalls[epoch] = recall
        ndcgs[epoch] = ndcg
        maps[epoch] = map

        # Record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[epoch] = avg_loss
        print(f"epoch: {epoch}; average loss: {avg_loss}, time taken: {int(time.time() - begin_time)}s")
        # Save model
        torch.save(rec_model.state_dict(), model_path)
        # Save last epoch
        meta_file = open(meta_path, 'wb')
        pickle.dump(epoch, meta_file)
        meta_file.close()

        # Early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if len(past_10_loss) > 10 and abs(total_loss - np.mean(past_10_loss)) < h_params['loss_delta']:
            print(f"***Early stop at epoch {epoch}***")
            break

        file = open(log_path, 'wb')
        pickle.dump(loss_dict, file)
        pickle.dump(recalls, file)
        pickle.dump(ndcgs, file)
        pickle.dump(maps, file)
        file.close()

    print("============================")


def test_model(test_set, rec_model,epoch,ks=[1, 5, 10]):
    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, preds1,preds2,labels = [], [],[],[]

    all_neg_samples = torch.load("neg_samples_train.pt")

    for sample in test_set:
        sample_to_device = generate_sample_to_device(sample)

        neg_sample_to_device_list = []
        if settings.enable_ssl:
            user_id = sample[0][2][0]
            current_POI = sample[-1][0][-2]
            neg_sample_to_device_list = generate_negative_sample_list(test_set, user_id, current_POI)
            user_neg_samples = all_neg_samples.get(user_id, [])
            if len(user_neg_samples) == 0:
                all_available_negatives = []
                for neg_list in all_neg_samples.values():
                    all_available_negatives.extend(neg_list)

                selected_neg_samples = random.sample(all_available_negatives, min(10, len(all_available_negatives)))
            else:
                # 随机选 k 个
                k = 10
                selected_neg_samples = random.sample(user_neg_samples, min(k, len(user_neg_samples)))
        pred,pred1,pred2,label= rec_model.predict(sample_to_device, neg_sample_to_device_list,selected_neg_samples)
        preds.append(pred.detach())
        preds1.append(pred1.detach())
        preds2.append(pred2.detach())

        labels.append(label.detach())
    preds = torch.stack(preds, dim=0)
    preds1 = torch.stack(preds1, dim=0)
    preds2 = torch.stack(preds2, dim=0)
    pred_list = [preds, preds1, preds2]
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1)

    item_hit_rate = compute_item_hit_rate(preds,labels,20)
    item_hit_rate1 = compute_item_hit_rate(preds1,labels,20)
    item_hit_rate2 = compute_item_hit_rate(preds2,labels,20)
    item_hit_rate_mean = average_dicts([item_hit_rate, item_hit_rate1, item_hit_rate2])
    fused_preds = hit_aware_vote(pred_list,item_hit_rate_mean,20).to(labels.device)

    recalls, NDCGs, MAPs = {}, {}, {}
    recalls1, NDCGs1, MAPs1 = {}, {}, {}
    recalls_fused, NDCGs_fused, MAPs_fused = {}, {}, {}

    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")


        recalls1[k] = calc_recall(labels, preds1, k)
        NDCGs1[k] = calc_ndcg(labels, preds1, k)
        MAPs1[k] = calc_map(labels, preds1, k)
        print(f"KAN——Recall @{k} : {recalls1[k]},\tNDCG@{k} : {NDCGs1[k]},\tMAP@{k} : {MAPs1[k]}")


        recalls_fused[k] = calc_recall(labels, fused_preds, k)
        NDCGs_fused[k] = calc_ndcg(labels, fused_preds, k)
        MAPs_fused[k] = calc_map(labels, fused_preds, k)
        print(f"Fused Recall@{k}: {recalls_fused[k]},\tFused NDCG@{k}: {NDCGs_fused[k]},\tFused MAP@{k}: {MAPs_fused[k]}")


    return recalls_fused, NDCGs_fused, MAPs_fused

if __name__ == '__main__':
    # Get current time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Datetime of now：", now_str)

    # Get parameters
    h_params = {
        'expansion': 4,
        'lr': settings.lr,
        'epoch': settings.epoch,
        'loss_delta': 1e-3}

    processed_data_directory = './processed_data/'
    if settings.enable_dynamic_day_length:
        processed_data_directory += 'dynamic_day_length'
    else:
        processed_data_directory += 'original'

    # Read training data
    file = open(f"{processed_data_directory}/{city}_train", 'rb')
    train_set = pickle.load(file)

    poi_category_mapping = extract_poi_info_dict(f"{processed_data_directory}/{city}_train")
    #build_and_save_all_negatives_v2( dataset=train_set,poi_info_dict=poi_category_mapping,k=10,strong_save_path='SIN_pos_samples_strong.pt', weak_save_path='SIN_pos_samples_weak.pt')
    #build_and_save_all_negatives(train_set, poi_category_mapping, k=5, save_path='PHO_pos_samples_train.pt')

    file = open(f"{processed_data_directory}/{city}_valid", 'rb')
    valid_set = pickle.load(file)

    file = open(f"{processed_data_directory}/{city}_test", 'rb')
    test_set = pickle.load(file)
    #build_and_save_all_negatives(valid_set, poi_category_mapping, k=5, save_path='PHO_pos_samples_test.pt')


    # Read meta data
    file = open(f"{processed_data_directory}/{city}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {"POI": torch.tensor(len(meta["POI"])).to(device),
                  "cat": torch.tensor(len(meta["cat"])).to(device),
                  "user": torch.tensor(len(meta["user"])).to(device),
                  "hour": torch.tensor(len(meta["hour"])).to(device),
                  "day": torch.tensor(len(meta["day"])).to(device)}

    # Adjust specific parameters for each city
    if city == 'SIN':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 3
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
    elif city == 'NYC':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.1
        h_params['head_num'] = 1
    elif city == 'PHO':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 2
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1

    # Create output folder
    if not os.path.isdir('./results'):
        os.mkdir("./results")

    print(f'Current GPU {settings.gpuId}')
    for run_num in range(1, 1 + settings.run_times):
        run_name = f'{settings.output_file_name} {run_num}'
        print(run_name)

        train_model(train_set, valid_set, h_params, vocab_size, device,run_name=run_name)
        print_output_to_file(settings.output_file_name, run_num)

        t = random.randint(1, 9)
        print(f"sleep {t} seconds")
        time.sleep(t)

        clear_log_meta_model(settings.output_file_name, run_num)
    calculate_average(settings.output_file_name, settings.run_times)
