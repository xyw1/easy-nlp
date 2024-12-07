# 我是第1章的标题
循环神经网络（RNN）是一种神经网络类型，其神经元的输出在下一个时间步会反馈作为输入，使网络具有处理序列数据的能力。它能处理变长序列，挖掘数据中的时序信息，不过存在长期依赖问题，即难以处理长序列中相距较远的信息关联。
RNN与普通神经网络的主要区别在于其具有记忆功能，神经元的输出能作为下一步输入，可处理序列数据，且输入和输出长度不固定；普通神经网络一般处理独立同分布的数据，层与层之间是简单的前馈连接关系，输入输出的长度通常是固定的。

RNN的应用场景广泛，在自然语言处理方面，可用于语言模型来预测下一个单词的概率，还能完成机器翻译、文本生成任务；在语音识别领域，能够处理语音这种时间序列信号，提高识别准确率；在时间序列预测中，像股票价格预测、天气预测等，RNN通过学习历史数据模式预测未来值；在视频分析中，它可以处理视频帧序列，进行动作识别等操作。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba  # 用于中文分词
# 自定义 RNN 模型
class CustomRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.W_ih = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.W_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        print('x',x.size())
        batch_size, seq_length = x.size()
        embedded = self.embedding(x)
        print('embedded',embedded.size())
        hiddens = []
        for t in range(seq_length):
            x_t = embedded[:, t, :]
            print(x_t.size(),self.W_ih.size(),self.b_ih.size() ,h_prev.size(), self.W_hh.size(),self.b_hh.size())
            h_t = torch.tanh(torch.mm(x_t, self.W_ih) + self.b_ih + torch.mm(h_prev, self.W_hh) + self.b_hh)
            hiddens.append(h_t)
            h_prev = h_t
        h_final = hiddens[-1]
        output = self.fc(h_final)
        return output, h_prev

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# 读取周杰伦歌词文件，假设文件名为jay_chou_lyrics.txt，每行是一首歌词
def read_lyrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lyrics = f.readlines()
    return lyrics

# 构建数据集类
class LyricsDataset(Dataset):
    def __init__(self, lyrics, word2idx, seq_length):
        self.lyrics = lyrics
        self.word2idx = word2idx
        self.seq_length = seq_length
        self.vocab_size = len(word2idx)

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        words = jieba.lcut(lyric) # 分词。输出分词后的列表
        # 将歌词转换为索引序列
        indices = [self.word2idx.get(word, 0) for word in words]
        print('indices:',indices)
        # 生成输入和目标序列
        inputs = indices[:-1]
        targets = indices[1:]

        # 对输入序列进行填充或截断
        if len(inputs) < self.seq_length:
            inputs = [0] * (self.seq_length - len(inputs)) + inputs
        else:
            inputs = inputs[-self.seq_length:]
        inputs = torch.tensor(inputs, dtype=torch.long)

        # 对目标序列进行填充或截断
        if len(targets) < self.seq_length:
            targets = [0] * (self.seq_length - len(targets)) + targets
        else:
            targets = targets[-self.seq_length:]
        targets = torch.tensor(targets, dtype=torch.long)

        return inputs, targets


# 构建词表
def build_vocab(lyrics):
    word2idx = {"<PAD>": 0}
    idx = 1
    for lyric in lyrics:
        words = jieba.lcut(lyric)
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    return word2idx

# 训练函数
def train(model, dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            print('input:',inputs.size())
            optimizer.zero_grad()
            h_prev = model.init_hidden(inputs.size(0))
            print('init h_prev:',h_prev.size())
            output, h_prev = model(inputs, h_prev)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# 主函数
if __name__ == "__main__":
    file_path = "./test.txt"
    lyrics = read_lyrics(file_path)
    print(lyrics)
    word2idx = build_vocab(lyrics)
    print(word2idx)
    seq_length = 20
    dataset = LyricsDataset(lyrics, word2idx, seq_length)
    print(dataset[0])
    print(dataset[1])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    vocab_size = dataset.vocab_size
    hidden_size = 256
    num_layers = 2
    output_size = vocab_size
    model = CustomRNN(vocab_size, hidden_size, num_layers, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train(model, dataloader, optimizer, criterion, num_epochs=10)
```
