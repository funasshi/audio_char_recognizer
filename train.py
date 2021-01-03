# %%
import numpy as np
import torch
import torchsummary

from torch import nn
from model import InceptionLSTM
import torch.optim as optim
from data_loader import audio_data_loader
from get_stpsd import get_stpsd
from create_graph import create_loss_graph, create_acc_graph
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
from word_processor3 import make_dataset, preprocess

# GPU,CPUの指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# batch数
batch = 8

# epoch数
epochs = 100

# モデル
model = InceptionLSTM(batch=batch, out_channels=64).to(device)

# 最適化アルゴリズム
optimizer = optim.Adam(params=model.parameters(), lr=0.0005)

# 損失関数
loss_fn = nn.CrossEntropyLoss()
# %%
# 生データセット
X, y, _ = make_dataset()

# log変換
# X = np.log(X)

# trainとtestに分ける(クラスごとの比率は一定)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y, shuffle=True)

# 前処理
# 0.clipping
# 1.標準化(データごと)
# 2.torch.Tensor化
# 3.型調整
X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)

# gpuがあればgpuに乗っける
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# %%

# 訓練関数


def train(epoch):
    model.train()  # modelを訓練可能に(accracyを出すところで勾配を切るため)
    train_data_loader = audio_data_loader(
        X_train, y_train, batch, shuffle=True, stpsd=False, aug_noise=False, aug_shift=False)  # data_loaderを定義

    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in train_data_loader:
        optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット
        output = model(data)  # 入力dataをinputし、出力を求める
        loss = loss_fn(output, target)  # 出力と訓練データの正解との誤差を求める
        loss.backward()  # 誤差のバックプロパゲーションを求める
        optimizer.step()  # バックプロパゲーションの値で重みを更新する
    print("epoch%s:終了   loss=%s" % (epoch, loss.item()))
    train_acc = train_accuracy()
    test_acc = test_accuracy()
    train_accuracy_topk(k=5)
    test_accuracy_topk(k=5)
    return loss.item(), train_acc, test_acc

# 訓練データのaccuracyを求める関数


def train_accuracy():
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0

    # データローダーから1ミニバッチずつ取り出して計算する
    output = model(X_train)  # 入力dataをinputし、出力を求める

    # 推論する
    pred = output.data.max(1, keepdim=True)[1]  # 出力ラベルを求める
    correct += pred.eq(y_train.data.view_as(pred)).sum()  # 正解と一緒だったらカウントアップ
    # 正解率を出力
    data_num = y_train.shape[0]  # データの総数
    accuracy = 100. * correct / data_num
    print('訓練データの正解率: {}/{} ({:.0f}%)\n'.format(correct,
                                                data_num, accuracy))
    return accuracy

# 参考: https://zenn.dev/nnabeyang/articles/8b643ca99ddab2a568e0
# 訓練データのtop-kに入っているかどうかのaccuracyを求める関数
def train_accuracy_topk(k=5):
    model.eval()  # ネットワークを推論モードに切り替える
    N = X_train.size(0)
    # データローダーから1ミニバッチずつ取り出して計算する
    output = model(X_train)  # 入力dataをinputし、出力を求める
    assert N == output.size(0)
    assert output.dim() == 2
    # 推論する
    # topkは上位k個のlabel
    _, topk_pred = output.topk(k, dim=1, largest=True, sorted=True)
    # expandするためにunsqueeze
    y_train_unsq = torch.unsqueeze(y_train, dim=1)
    # expand
    target = y_train_unsq.expand_as(topk_pred)
    # expandの確認
    assert (target[:, 0].eq(target[:, 1])).sum().item() == N
    topk_correct = topk_pred.eq(target).sum().item() # 正解と一緒だったらカウントアップ
    # 正解率を出力
    topk_accuracy = 100. * topk_correct / N
    print('訓練データの正解率(top-{}): {}/{} ({:.0f}%)\n'.format(k, topk_correct,
                                                N, topk_accuracy))
    return topk_accuracy



# テストデータのaccuracyを求める関数


def test_accuracy():
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0

    output = model(X_test)  # 入力dataをinputし、出力を求める

    # 推論する
    pred = output.data.max(1, keepdim=True)[1]  # 出力ラベルを求める
    correct += pred.eq(y_test.data.view_as(pred)).sum()  # 正解と一緒だったらカウントアップ

    # 正解率を出力
    data_num = y_test.shape[0]  # データの総数
    accuracy = 100. * correct / data_num
    print('テストデータの正解率: {}/{} ({:.0f}%)\n'.format(correct,
                                                 data_num, accuracy))
    return accuracy

# 上位k個
def test_accuracy_topk(k=5):
    model.eval()  # ネットワークを推論モードに切り替える
    N = X_test.size(0)

    output = model(X_test)  # 入力dataをinputし、出力を求める
    assert output.dim() == 2
    assert N == output.size(0)

    # topkは上位k個のlabel
    _, topk_pred = output.topk(k, dim=1, largest=True, sorted=True)
    # expandするためにunsqueeze
    y_test_unsq = torch.unsqueeze(y_test, dim=1)
    # expand
    target = y_test_unsq.expand_as(topk_pred)
    # expandの確認
    assert (target[:, 0].eq(target[:, 1])).sum().item() == N
    topk_correct = topk_pred.eq(target).sum().item() # 正解と一緒だったらカウントアップ
    # 正解率を出力
    topk_accuracy = 100. * topk_correct / N
    
    print('テストデータの正解率(top-{}): {}/{} ({:.0f}%)\n'.format(k, topk_correct,
                                                N, topk_accuracy))
    return topk_accuracy


# %%


# 訓練開始
loss_list = []
train_acc_list = []
test_acc_list = []
for epoch in range(epochs):
    loss, train_acc, test_acc = train(epoch)
    loss_list.append(loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


# loss関数のグラフ作成
create_loss_graph(loss_list, epochs)
create_acc_graph(train_acc_list, test_acc_list, epochs)

print(y_train.size())
