import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision.models as models
from utils import load_data, train, val, matplot_loss, matplot_acc
import matplotlib.pyplot as plt


# 解决中文显示问题
# 运行配置参数中的字体（font）为黑体（SimHei）
plt.rcParams['font.sans-serif'] = ['simHei']
# 运行配置参数总的轴（axes）正常显示正负号（minus）
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
train_dataloader, val_dataloader = load_data()

# 如果有NVIDA显卡，可以转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化，将模型转到device
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 11)
model.to(device)

# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义优化器(随机梯度下降法)
# params(iterable)：要训练的参数，一般传入的是model.parameters()
# lr(float)：learning_rate学习率，也就是步长
# momentum(float, 可选)：动量因子（默认：0），矫正优化率
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)

# 学习率每隔10轮变为原来的0.5
# StepLR：用于调整学习率，一般情况下会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果
# optimizer （Optimizer）：更改学习率的优化器
# step_size（int）：每训练step_size个epoch，更新一次参数
# gamma（float）：更新lr的乘法因子
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

# 训练次数
epoch = 100

# 用于判断最佳模型
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f'epoch{t + 1}\n----------')
    # 训练模型
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    # 验证模型
    val_loss, val_acc = val(val_dataloader, model, loss_fn)
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存最好的模型权重
    if val_acc > min_acc:
        folder = f'trained_model/{type(model).__name__}'
        # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
        if not os.path.exists(folder):
            # os.mkdir() 方法用于以数字权限模式创建目录
            os.mkdir(folder)
        min_acc = val_acc
        print(f'save best model，第{t + 1}轮')
        # torch.save(state, dir)：保存模型等相关参数，dir表示保存文件的路径+保存文件名
        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
        torch.save(model.state_dict(), folder + '/best_model.pth')

    # # 保存最后一轮权重
    # if t == epoch - 1:
    #     torch.save(model.state_dict(), 'trained_model/AlexNet/best_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)

print('done')
