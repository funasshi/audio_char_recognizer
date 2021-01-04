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
from spectrogram import audio_to_spectrogram, make_batch_spectrogram

# GPU,CPUの指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# batch数
batch = 32

# epoch数
epochs = 300

# モデル
model = InceptionLSTM(batch=batch, out_channels=64).to(device)

# 最適化アルゴリズム
optimizer = optim.Adam(params=model.parameters(), lr=0.0005)

# 損失関数
loss_fn = nn.CrossEntropyLoss()
# %%
# 生データセット(テスト済み)
sub = False

if sub:
    fr = 4800
else:
    fr = 48000
X, y, _ = make_dataset(sub=sub)


print("")
# %%
# trainとtestに分ける(クラスごとの比率は一定)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y, shuffle=True)

# 前処理
# 0.clipping
# 1.標準化(データごと)
# 2.torch.Tensor化
# 3.型調整

# %%
X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)

# # accuracyを出す用データ
# X_train_spec = make_batch_spectrogram(X_train)
# X_test_spec = make_batch_spectrogram(X_test)

# %%
# gpuがあればgpuに乗っける
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
# X_train_spec = X_train_spec.to(device)
# X_test_spec = X_test_spec.to(device)
# %%
train_data_loader = audio_data_loader(
    X_train, y_train, batch, fr=fr, shuffle=False, feature="spectro", aug_noise=True, aug_shift=True)  # data_loaderを定義

for data, target in train_data_loader:
    print(data.shape)
    print(data[0])
    plt.imshow(data[0].cpu().numpy())
    break
# %%

# 訓練関数


def train(epoch):
    model.train()  # modelを訓練可能に(accracyを出すところで勾配を切るため)
    train_data_loader = audio_data_loader(
        X_train, y_train, batch, fr=fr, shuffle=True, feature="stpsd", aug_noise=False, aug_shift=False)  # data_loaderを定義
    acc_count = 0
    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in train_data_loader:
        optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット
        output = model(data)  # 入力dataをinputし、出力を求める
        loss = loss_fn(output, target)  # 出力と訓練データの正解との誤差を求める
        loss.backward()  # 誤差のバックプロパゲーションを求める
        optimizer.step()  # バックプロパゲーションの値で重みを更新する
        prediction = output.data.max(1)[1]
        acc_count += prediction.eq(target.data).sum()
    train_acc = (acc_count / X_train.shape[0]).item() * 100
    print("epoch{}:終了   loss={}   acc={}/{} ({:.0f})%".format(epoch, loss.item(), acc_count.item(), X_train.shape[0], (train_acc)))
    test_loss, test_acc, topk_accuracy = test_accuracy()
    return loss.item(), test_loss, train_acc, test_acc, topk_accuracy

# 訓練データのaccuracyを求める関数


def train_accuracy():
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0

    # データローダーから1ミニバッチずつ取り出して計算する
    acc_train_data_loader = audio_data_loader(
        X_train, y_train, 128, fr=fr, shuffle=False, feature="stpsd", aug_noise=False, aug_shift=False)  # data_loaderを定義

    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in acc_train_data_loader:
        output = model(data)  # 入力dataをinputし、出力を求める
        # 推論する
        pred = output.data.max(1, keepdim=True)[1]  # 出力ラベルを求める
        correct += pred.eq(target.data.view_as(pred)
                           ).sum()  # 正解と一緒だったらカウントアップ
    # 正解率を出力
    data_num = y_train.shape[0]  # データの総数
    accuracy = 100. * correct / data_num
    print('訓練データの正解率: {}/{} ({:.0f}%)\n'.format(correct,
                                                data_num, accuracy))
    return accuracy
# テストデータのaccuracyを求める関数


def test_accuracy(k=5):
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0
    loss = 0
    topk_correct = 0
    # データローダーから1ミニバッチずつ取り出して計算する
    acc_test_data_loader = audio_data_loader(
        X_test, y_test, 32, fr=fr, shuffle=False, feature="stpsd", aug_noise=False, aug_shift=False)  # data_loaderを定義

    # データローダーから1ミニバッチずつ取り出して計算する
    for i, (data, target) in enumerate(acc_test_data_loader):
        output = model(data)  # 入力dataをinputし、出力を求める
        # 推論する
        loss += loss_fn(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]  # 出力ラベルを求める
        correct += pred.eq(target.data.view_as(pred)
                           ).sum()  # 正解と一緒だったらカウントアップ

        # topkは上位k個のlabel
        _, topk_pred = output.topk(k, dim=1, largest=True, sorted=True)
        # expandするためにunsqueeze
        y_test_unsq = torch.unsqueeze(target, dim=1)
        # expand
        target = y_test_unsq.expand_as(topk_pred)
        # expandの確認
        topk_correct += topk_pred.eq(target).sum().item()  # 正解と一緒だったらカウントアップ
    # 正解率を出力
    data_num = y_test.shape[0]  # データの総数
    loss /= i
    accuracy = 100. * correct / data_num
    topk_accuracy = 100. * topk_correct / data_num
    print('test_loss:{}'.format(loss))
    print('テストデータの正解率: {}/{} ({:.0f}%)'.format(correct,
                                               data_num, accuracy))
    print('テストデータの正解率(top-{}): {}/{} ({:.0f}%)\n'.format(k, topk_correct,
                                                         data_num, topk_accuracy))
    return loss, accuracy, topk_accuracy

# 上位k個


# %%
# 訓練開始
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []
topk_acc_list = []


def from_the_middle(epoch, model):
    model_path = './params/model_para_{}epoch.pth'.format(epoch)
    model.load_state_dict(torch.load(model_path))
    return model


for epoch in range(epochs):
    train_loss, test_loss, train_acc, test_acc, topk_acc = train(epoch)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    topk_acc_list.append(topk_acc)

torch.save(model.state_dict(), './params/model_para_{}epoch.pth'.format(epochs))


# loss関数のグラフ作成
create_loss_graph(train_loss_list, test_loss_list, epochs)
create_acc_graph(train_acc_list, test_acc_list, topk_acc_list, epochs)
