import math
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import settings
from ChebyKANLayer import ChebyKANLinear,decompose_to_time_domain_2d
import numpy as np
device = settings.gpuId if torch.cuda.is_available() else 'cpu'

class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, degree=0, stride=1, padding=0, dilation=1, groups=1, act=False,
                 bn=False, bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
        
def reconstructq(coeff_real, coeff_imag):
    recon_complex = coeff_real + 1j * coeff_imag  
    reconstructed_time = torch.fft.ifft(recon_complex, dim=-1).real
    return reconstructed_time
    
class ChebyKANLayer2(nn.Module):
    def __init__(self, in_features, out_features,order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
                            in_features,
                            out_features,
                            order)
    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,-1).contiguous()
        return x
class CheckInEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb), 1),torch.cat((user_emb, hour_emb, day_emb), 1)

class CheckInEmbedding1(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: [1, max_len, embed_size]
        self.register_buffer('pe', pe)

    def forward(self, x,direction='forward'):
        # x shape: [seq_len, embed_size] or [batch, seq_len, embed_size]
        if direction == 'forward':
            seq_len = x.size(0)
            return x + self.pe[0, :seq_len]
        else:
            seq_len = x.size(0)
            return x + self.pe[0, :seq_len].flip(0)

class EncoderBlockWithDirectionalPE(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, direction='forward',max_len=5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        self.encoder = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.direction = direction  # 可以设置 'forward' 或 'backward'

    def forward(self, x):
 
        x_pe = self.pos_encoder(x, direction=self.direction)
        out = self.encoder(x_pe, x_pe, x_pe)
        return out


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,
            embedding_layer1,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.embedding_layer1 = embedding_layer1
        self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                EncoderBlockWithDirectionalPE(
                    180,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    direction='forward',
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.layers2 = nn.ModuleList(
            [
                EncoderBlockWithDirectionalPE(
                    180,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    direction='forward',
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.patch_stride = 10
        self.patch_len = 20
        self.patch_num = int((embed_size - self.patch_len) / self.patch_stride + 2)
        self.kan_3d = ChebyKANLayer2(self.patch_len,self.patch_len, order=5)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.conv1 = BasicConv(self.patch_len, self.patch_len, groups=self.patch_len)
        self.norm1 = nn.LayerNorm(self.patch_len)
    def do_patching(self, x):
        # x: [B, L]
        x_end = x[:, -1:]  # [B, 1]
        x_padding = x_end.repeat(1, self.patch_stride)  # [B, patch_stride]
        x_new = torch.cat((x, x_padding), dim=-1)  # [B, L + patch_stride]

        x_new = x_new.unsqueeze(1)  # [B, 1, L+pad]
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len,
                               step=self.patch_stride)  # [B, 1, num_patches, patch_len]
        x_patch = x_patch.squeeze(1)  # [B, num_patches, patch_len]

        return x_patch

    def freq(self, feature_seq):

        out = self.embedding_layer1(feature_seq)
        out = torch.fft.fft(out, dim=-1)
        out_seq_real,out_seq_imag =out.real, out.imag
        out_seq_real = self.do_patching(out_seq_real)
        out_seq_imag = self.do_patching(out_seq_imag)
        out_seq_real = self.norm1 (out_seq_real)
        out_seq_imag = self.norm1 (out_seq_imag)
        out_seq_r = self.conv1(out_seq_real)
        out_seq_i = self.conv1(out_seq_imag)
        out_seq_real= self.kan_3d(out_seq_real)+out_seq_r
        out_seq_imag  = self.kan_3d(out_seq_imag )+out_seq_i
        out = reconstructq(out_seq_real,out_seq_imag)
        out = self.flatten(out)
        return out

    def forward(self, feature_seq,long=True):
        freq = self.freq(feature_seq)
        embedding,embedding1 = self.embedding_layer(feature_seq)  # [len, embedding]
        embedding_low,embedding_high = decompose_to_time_domain_2d(embedding, max_order=32, low=(0, 12), high=(8, 12))
        embedding1_low,embedding1_high = decompose_to_time_domain_2d(embedding1, max_order=32, low=(0, 12), high=(8, 12))


        out = embedding
        out1 = embedding1

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out)
        for layer in self.layers2:
            out1 = layer(out1)

        out = out + embedding_low
        out1 = out1 + embedding1_low

        out = torch.cat((out,out1),dim=-1)

        return out,freq


# Attention for query and key with different dimension
class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query)  # [embed_size]
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out

class FMMC(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=60,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5

        # Layers
        self.embedding = CheckInEmbedding(
            f_embed_size,
            vocab_size
        )
        self.embedding1= CheckInEmbedding1(
            f_embed_size,
            vocab_size
        )
        self.encoder = TransformerEncoder(
            self.embedding,
            self.embedding1,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        self.lstm = nn.LSTM(
            input_size=180,
            hidden_size=180,
            num_layers=num_lstm_layers,
            dropout=0.1
        )
        self.final_attention = Attention(
            qdim=f_embed_size,
            kdim=180*2
        )
        self.final_attention1 = Attention(
            qdim=f_embed_size,
            kdim=600
        )
        self.out_linear = nn.Sequential(nn.Linear(180*2, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))
        self.out_linear2 = nn.Sequential(nn.Linear(600, 300 * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(300 * forward_expansion, vocab_size["POI"]))

        self.dropout = nn.Dropout(p=0.1)
        self.loss_func = nn.CrossEntropyLoss()
        self.tryone_line2 = nn.Linear(180*2, f_embed_size)

        self.enhance_val = nn.Parameter(torch.tensor(0.5))
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
    def ssl(self, embedding_1, embedding_2, neg_embedding):
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))

        def single_infoNCE_loss_simple(embedding1, embedding2, neg_embedding):
            pos = score(embedding1, embedding2)
            neg1 = score(embedding1, neg_embedding)
            neg2 = score(embedding2, neg_embedding)
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss

        ssl_loss = single_infoNCE_loss_simple(embedding_1, embedding_2, neg_embedding)
        return ssl_loss
    def feature_mask(self, sequences, mask_prop):
        masked_sequences = []
        for seq in sequences:  # each long term sequences
            feature_seq, day_nums = seq[0], seq[1]
            seq_len = len(feature_seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            feature_seq[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_seq[4, masked_index] = self.vocab_size["day"]  # mask day

            masked_sequences.append((feature_seq, day_nums))
        return masked_sequences
    def forward(self, sample, neg_sample_list,negative_sample_list1):
        # Process input sample
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        short_term_features = short_term_sequence[0][:, :- 1]
        target = short_term_sequence[0][0, -1]
        user_id = short_term_sequence[0][2, 0]

        long_term_sequences = self.feature_mask(long_term_sequences, 0.1)
        # Long-term
        long_term_out = []
        long_freq = []
        for seq in long_term_sequences:

            output,freq = self.encoder(seq[0],long=True)
            long_term_out.append(output)
            long_freq.append(freq)
        long_term_catted = torch.cat(long_term_out, dim=0)
        long_freq_catted = torch.cat(long_freq, dim=0)

        # Short-term
        short_term_state,short_freq = self.encoder(short_term_features,long=False)

        # User enhancement
        user_embed = self.embedding.user_embed(user_id)
        embedding, embedding1 = self.embedding(short_term_features)
        embedding = embedding.unsqueeze(0)
        embedding1 = embedding1.unsqueeze(0)
        output, _ = self.lstm(embedding)
        output1, _ = self.lstm(embedding1)
        output = output.squeeze(0)
        output1 = output1.squeeze(0)
        out =torch.cat((output, output1), dim=1)
        short_term_enhance =out
        user_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.tryone_line2(
            torch.mean(short_term_enhance, dim=0))

        # SSL
        neg_sample1 = []
        neg_short_term_states = []
        for neg_day_sample in neg_sample_list:
            neg_trajectory_features = neg_day_sample[0]
            neg_short_term_state,_ = self.encoder(neg_trajectory_features)
            neg_short_term_state = torch.mean(neg_short_term_state, dim=0)
            neg_short_term_states.append(neg_short_term_state)

        for neg_sample_list in negative_sample_list1:
            neg_trajectory_features = neg_sample_list[0]
            neg_short_term_state,_ = self.encoder(neg_trajectory_features)
            neg_short_term_state = torch.mean(neg_short_term_state, dim=0)
            neg_sample1.append(neg_short_term_state)

        short_embed_mean = torch.mean(short_term_state, dim=0)
        long_embed_mean = torch.mean(long_term_catted, dim=0)


        neg_embed_mean = torch.mean(torch.stack(neg_short_term_states), dim=0)
        neg_sample_mean1 = torch.mean(torch.stack(neg_sample1), dim=0)


        ssl_loss = self.ssl(short_embed_mean, long_embed_mean, neg_embed_mean)
        ssl_loss2 = self.ssl(short_embed_mean, neg_sample_mean1, neg_embed_mean)
        ssl_loss3 = self.ssl(long_embed_mean, neg_sample_mean1, neg_embed_mean)
        ssl = ssl_loss2 + ssl_loss3 + ssl_loss

        # Final predict
        h_all = torch.cat((short_term_state, long_term_catted))
        h_freq = torch.cat((short_freq, long_freq_catted))

        final_freq = self.final_attention1(user_embed, h_freq, h_freq)
        output1 = self.out_linear2(final_freq)

        final_att = self.final_attention(user_embed, h_all, h_all)
        output = self.out_linear(final_att)

        output2 = self.a * output + (1-self.a) * output1

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)

        pred1 = torch.unsqueeze(output1, 0)
        pred_loss1 = self.loss_func(pred1, label)

        pred2 = torch.unsqueeze(output2, 0)
        pred_loss2 = self.loss_func(pred2, label)

        pred_loss = self.loss_func(pred, label)


        loss = pred_loss + pred_loss1 + pred_loss2 + ssl
        return loss, output,output1,output2,long_term_catted

    def predict(self, sample, neg_sample_list,negative_sample_list1):
        _, pred_raw ,pred_raw1,pred_raw2,_= self.forward(sample, neg_sample_list,negative_sample_list1)
        ranking = torch.sort(pred_raw, descending=True)[1]
        ranking1 = torch.sort(pred_raw1, descending=True)[1]
        ranking2 = torch.sort(pred_raw2, descending=True)[1]


        target = sample[-1][0][0, -1]

        return ranking,ranking1,ranking2,target
