from torch import nn
from model import InceptionLSTM
import torch.optim as optim
from data_loader import audio_data_loader

model = InceptionLSTM()
optimizer = optim.Adam()
loss_fn = nn.CrossEntropyLoss()

batch_size = 16
epochs = 100

X, y = make_data()
audio_data_loader = audio_data_loader(X, y, batch_size, shuffle=True)


def train(epoch):
    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in audio_data_loader:
        optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット
        output = model(data)  # 入力dataをinputし、出力を求める
        loss = loss_fn(output, target)  # 出力と訓練データの正解との誤差を求める
        loss.backward()  # 誤差のバックプロパゲーションを求める
        optimizer.step()  # バックプロパゲーションの値で重みを更新する
    print("epoch{}:終了\n".format(epoch))


for epoch in epochs:
    train(epoch)


def test():
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0

    # データローダーから1ミニバッチずつ取り出して計算する
    for data, target in loader_test:
        data, target = Variable(data), Variable(target)  # 微分可能に変換
        output = model(data)  # 入力dataをinputし、出力を求める

        # 推論する
        pred = output.data.max(1, keepdim=True)[1]  # 出力ラベルを求める
        correct += pred.eq(target.data.view_as(pred)).sum()  # 正解と一緒だったらカウントアップ

    # 正解率を出力
    data_num = len(loader_test.dataset)  # データの総数
    print('\nテストデータの正解率: {}/{} ({:.0f}%)\n'.format(correct,
                                                   data_num, 100. * correct / data_num))
