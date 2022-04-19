import csv
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as utils
from sklearn.metrics import classification_report, accuracy_score

# 固定随机数，使结果能复现
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# 超参
# EPOCH = 100
EPOCH = 10
BATCH_SIZE = 256
LEARN_RATE = 0.001
# LEARN_RATE = 0.01
DROPOUT = 0.5

MAX_LENGTH = 200
EMBEDDING_SIZE = 128
# HIDDEN_SIZE = 64
HIDDEN_SIZE = 32


# 处理数据
def parse_data(file_name):
    word2token = {"[PAD]": 0}
    parsed = []
    with open(file_name, mode="r", encoding="utf-8") as f:
        # 处理csv文件
        reader = csv.reader(f)
        for data in reader:
            # 跳过第一行
            if reader.line_num == 1:
                continue
            # 将句子中可能存在的逗号用空格分隔：…… election, 1796 …… → …… election , 1796 ……
            Q = ",".join([" "+x.strip()+" " for x in data[1].replace("\"", "").split(",")]).strip()
            A = ",".join([" "+x.strip()+" " for x in data[2].replace("\"", "").split(",")]).strip()
            # 拼接两个句子
            QA = Q + " ? " + A
            # 按空格分割成单词
            words = QA.split()
            # 构造转换词典
            for word in words:
                if word not in word2token:
                    ids = len(word2token)
                    word2token[word] = ids
            # 将单词转换成token，同时padding到最大长度
            QA_token = [word2token[x] for x in words] + [word2token["[PAD]"]]*(MAX_LENGTH-len(words))
            # 保存数据random.shuffle(parsed)
            #     train_set = parsed[:cut]
            #     test_set = parsed[cut:]
            parsed.append({"QA": QA, "length": len(words), "QA_token": QA_token, "label": int(data[3])})
    # 随机打乱数据后划分训练集和测试集
    # cut = int(0.4*len(parsed))
    # random.shuffle(parsed)
    # train_set = parsed[:cut]
    # test_set = parsed[cut:]
    # 返回结果
    random.shuffle(parsed)
    # train_set = parsed[0:1731]
    # test_set = parsed[1732:1998]
    train_set = parsed[0:2400]
    test_set = parsed[2401:3000]
    return word2token, train_set, test_set


# torch数据集
class QAData(Data.Dataset):
    def __init__(self, data):
        super(QAData, self).__init__()
        self.data = data

    def __getitem__(self, index):
        cur_data = self.data[index]
        return {
            "QA": cur_data["QA"],
            "length": torch.tensor(cur_data["length"],  dtype=torch.long),
            "QA_token": torch.tensor(cur_data["QA_token"],  dtype=torch.long),
            "label": torch.tensor(cur_data["label"],  dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


# RNN模型，这里使用的是GRU
class RNN(nn.Module):
    def __init__(self, inp_size, hid_size, num_layers=1, batch_first=False, dropout=0, bidirectional=False):
        super(RNN, self).__init__()

        self.inp_size = inp_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        if self.num_layers == 1:
            self.dropout = 0
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            self.inp_size,
            self.hid_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

    def forward(self, inp_len, inp):
        inp_len, indices = torch.sort(inp_len, descending=True)
        inp = inp[indices]
        inp = utils.pack_padded_sequence(inp, inp_len, batch_first=self.batch_first)

        out, _ = self.rnn(inp)
        out, _ = utils.pad_packed_sequence(out, batch_first=self.batch_first)
        _, indices = torch.sort(indices, descending=False)
        out = out[indices]

        return out


# 最终的分类模型：token → embedding → RNN → attention → linear
class QAClassify(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, dropout):
        super(QAClassify, self).__init__()

        # 单词数，用于嵌入层的第一维
        self.vocab_size = vocab_size
        # 嵌入层输出维度
        self.emb_size = emb_size
        # RNN输出维度
        self.hid_size = hid_size

        # 嵌入层
        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        # 双向RNN
        self.rnn = RNN(self.emb_size, self.hid_size//2, batch_first=True, bidirectional=True)
        # attention的权重矩阵
        self.w = nn.Linear(self.hid_size, 1, bias=0)
        # 线性层，用于压缩最后输出向量的维度至标签数量
        self.linear = nn.Linear(self.hid_size, 2)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, length, token, label=None):
        # 将token转换为embedding。token维度：batch_size*MAX_LENGTH，out维度：batch_size*MAX_LENGTH*emb_size
        out = self.emb(token)
        # 使用dropout防止过拟合
        out = self.dropout(out)
        # 使用rnn提取上下文特征。rnn输出维度：batch_size*m*hid_size，其中m是当前batch中最长句子长度
        out = self.rnn(length, out)
        # 使用attention将句子中每个词语的特征加权求和为一个向量：batch_size*m*hid_size → batch_size*hid_size
        out = self.tanh(self.softmax(self.w(self.tanh(out))).permute(0, 2, 1).matmul(out)[:, 0, :])
        # 将句子特征降维至标签数：batch_size*hid_size → batch_size*2 
        out = self.linear(out)
        if self.training:
            loss = self.loss(out, label)
            return loss
        else:
            pred = torch.max(out, dim=1).indices
            return pred


if __name__ == "__main__":
    # 有显卡就用显卡
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 读取数据
    word2token, train_set, test_set = parse_data("/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/phase1_relevant_fact_selection_trainset.csv")
    train_set = Data.DataLoader(dataset=QAData(train_set), batch_size=BATCH_SIZE, shuffle=True)
    test_set = Data.DataLoader(dataset=QAData(test_set), batch_size=BATCH_SIZE, shuffle=False)
    # 加载模型
    model = QAClassify(len(word2token), EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT)
    model.to(dev)
    # 加载优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARN_RATE)
    # 训练和测试
    for epoch in range(EPOCH):
        # 训练
        epoch_loss = 0
        model.train()
        for batch_data in train_set:
            length = batch_data["length"].to(dev)
            QA_token = batch_data["QA_token"].to(dev)
            label = batch_data["label"].to(dev)
            # 计算损失
            loss = model(length, QA_token, label=label)
            epoch_loss += loss.item()*length.shape[0]
            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_set)
        # 测试
        epoch_label = []
        epoch_pred = []
        model.eval()
        for batch_data in test_set:
            length = batch_data["length"].to(dev)
            QA_token = batch_data["QA_token"].to(dev)
            label = batch_data["label"].to(dev)
            # 预测
            pred = model(length, QA_token)
            # 保存
            epoch_label += label.cpu().numpy().tolist()
            epoch_pred += pred.cpu().numpy().tolist()
        # 计算结果
        # report = classification_report(epoch_label, epoch_pred, digits=4)
        report = accuracy_score(epoch_label, epoch_pred)
        print("EPOCH: " + str(epoch+1) + ", LOSS: " + str(round(epoch_loss, 3)))
        print(report)

