import pandas as pd
import matplotlib.pyplot as plt

# 直接写入你给的实验数据，格式：param_type, param_value, k, max_recall, max_ndcg
data = [
    # r 系列
    ['r', '0', 5, 0.631579, 0.521311],
    ['r', '0', 10, 0.684211, 0.532787],
    ['r', '5', 5, 0.631579, 0.528873],
    ['r', '5', 10, 0.684211, 0.536480],
    ['r', '10', 5, 0.605263, 0.518225],
    ['r', '10', 10, 0.684211, 0.539111],
    ['r', '15', 5, 0.684211, 0.548095],
    ['r', '15', 10, 0.710526, 0.559974],

    # 32 系列（32 -6写作“32-6”方便）
    ['32', '6', 5, 0.631579, 0.539739],
    ['32', '6', 10, 0.631579, 0.552069],
    ['32', '12', 5, 0.657895, 0.521311],  # 从你的文本复制的Fused Recall@5数值
    ['32', '12', 10, 0.657895, 0.552069],
    ['32', '18', 5, 0.631579, 0.546833],
    ['32', '18', 10, 0.684211, 0.560812],
    ['32', '24', 5, 0.631579, 0.514094],
    ['32', '24', 10, 0.736842, 0.565958],

    # p 系列
    ['p', '10', 5, 0.605263, 0.520985],
    ['p', '10', 10, 0.684211, 0.531421],
    ['p', '20', 5, 0.631579, 0.521311],
    ['p', '20', 10, 0.684211, 0.532787],
    ['p', '30', 5, 0.605263, 0.505535],
    ['p', '30', 10, 0.657895, 0.517886],
    ['p', '40', 5, 0.578947, 0.489555],
    ['p', '40', 10, 0.631579, 0.518168],

    # 正 系列（正=写作“正”）
    ['正', '5', 5, 0.631579, 0.545008],
    ['正', '5', 10, 0.684211, 0.545008],
    ['正', '10', 5, 0.631579, 0.521311],
    ['正', '10', 10, 0.684211, 0.532787],
    ['正', '15', 5, 0.605263, 0.509292],
    ['正', '15', 10, 0.657895, 0.524187],
    ['正', '20', 5, 0.631579, 0.573148],
    ['正', '20', 10, 0.736842, 0.589752],
]

df = pd.DataFrame(data, columns=['param_type', 'param_value', 'k', 'max_recall', 'max_ndcg'])

import pandas as pd
import matplotlib.pyplot as plt

# 数据和 DataFrame 定义保持不变 ...
# （你已有的 data 与 df 初始化代码不用改）

plt.rcParams.update({
    'font.size': 20,       # 全局字体大小
    'axes.labelsize': 20,  # 坐标轴标签字体
    'axes.titlesize': 20,  # 子图标题字体
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 15
})

param_name_map = {
    'r': 'KAN Order $r$',
    '32': 'Chebyshev Term $n$',
    'p': 'Patch Size $p$',
    '正': 'Positive Samples'
}

def try_float(x):
    try:
        return float(x)
    except:
        return x

# 准备画布
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
axes = axes.flatten()

for idx, param in enumerate(df['param_type'].unique()):
    sub_df = df[df['param_type'] == param]
    x_labels = sorted(sub_df['param_value'].unique(), key=try_float)

    ax = axes[idx]
    #ax.set_title(param_name_map.get(param, param))

    for metric in ['max_recall', 'max_ndcg']:
        for k_val in [5, 10]:
            vals = []
            for val in x_labels:
                filtered = sub_df[(sub_df['param_value'] == val) & (sub_df['k'] == k_val)]
                if not filtered.empty:
                    vals.append(filtered[metric].values[0])
                else:
                    vals.append(None)
            label = f"{'Recall' if metric == 'max_recall' else 'NDCG'}@{k_val}"
            ax.plot(x_labels, vals, marker='o', linewidth=2, label=label)

    ax.set_xlabel(param_name_map.get(param, param))
    ax.set_ylabel("Score")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

# 去除多余空白并显示
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("figure.png", dpi=300, bbox_inches='tight')  # ✅ 应放在 show 之前
plt.show()
