import torch
import torch.nn as nn

# 定义了两个神经网络模型：Model_T 和 Model_M，它们都继承自 torch.nn.Module，并且使用了不同的激活函数、网络结构和输入/输出形式



class Model_T(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义了模型的激活函数为 sigmoid
        self.activation = torch.sigmoid

        # self.sequential = nn.Sequential(...)：这是一个顺序容器，包含了多个层：
        # nn.Linear(3, 7)：一个输入大小为 3，输出大小为 7 的全连接层。
        # nn.PReLU(7)：一个带有 7 个参数的 PReLU 激活函数（PReLU 是一种自适应的 ReLU 激活函数，允许负值部分有可训练的斜率）。
        # 另两个 nn.Linear(7, 7) 和 nn.PReLU(7)：分别是另外一层全连接层和 PReLU 激活。
        # 最后一层 nn.Linear(7, 1)：输出层，大小为 1，表示最终输出一个单一的值。
        self.sequential = nn.Sequential(
            nn.Linear(3, 7), nn.PReLU(7),
            nn.Linear(7, 7), nn.PReLU(7),
            nn.Linear(7, 1)
        )

    # 前向传播函数 forward
    # 通过 self.sequential(x) 传递输入数据 x，经过一系列的线性变换和 PReLU 激活。
    # 最终通过 self.activation(...)，也就是 sigmoid 激活函数，对模型的输出进行压缩，得到一个 [0, 1] 范围内的结果。
    def forward(self, x):
        return self.activation(self.sequential(x))



class Model_M(nn.Module):
    def __init__(self):
        super().__init__()
        # 模型的激活函数是 exp（指数函数）
        self.activation = torch.exp


        # self.sequential = nn.Sequential(...)：这个顺序容器包含了更多的层：
        # nn.Linear(6, 21)：输入为 6 维，输出为 21 维的全连接层。
        # nn.PReLU(21)：为输出大小为 21 的前一层加入 PReLU 激活函数。
        # 后面是另外两层 nn.Linear(21, 21) 和 nn.PReLU(21)，都是 21 个输出单元。
        # 最后一层是 nn.Linear(21, 3)，输出 3 个值。
        self.sequential = nn.Sequential(
            nn.Linear( 6, 21), nn.PReLU(21),
            nn.Linear(21, 21), nn.PReLU(21),
            nn.Linear(21, 21), nn.PReLU(21),
            nn.Linear(21,  3)
        )

    # 前向传播函数 forward：
    # 输入 x 通过 self.sequential(x) 传递，通过层与层之间的线性变换和激活函数。
    # 最终输出通过 self.activation(...)，即 exp 函数，得到一个指数化的输出。
    def forward(self, x):
        return self.activation(self.sequential(x))