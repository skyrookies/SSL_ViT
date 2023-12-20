import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# 读取日志文件为DataFrame
log_file = "notebooks/two_class_endo_attn_only_vit_09_fc_and_attn/log.txt"  # 替换为您的日志文件路径
df = pd.read_json(log_file, lines=True)

# 提取所需的列
epochs = df["epoch"]
train_lr = df["train_lr"]
train_loss = df["train_loss"]
test_loss = df["test_loss"]
test_acc1 = df["test_acc1"]

log_dir = "log/two_class_endo_attn_only_vit_09_fc_and_attn"  # 替换为您希望保存TensorBoard日志的路径
writer = SummaryWriter(log_dir)
highest_test_acc1 = 0.0
for epoch, lr, t_loss, v_loss, v_acc1 in zip(epochs, train_lr, train_loss, test_loss, test_acc1):
    writer.add_scalar("Train/learning_rate", lr, epoch)
    writer.add_scalar("Train/loss", t_loss, epoch)
    writer.add_scalar("Test/loss", v_loss, epoch)
    writer.add_scalar("Test/accuracy", v_acc1, epoch)
    # 更新最高测试准确率值
    if v_acc1 > highest_test_acc1:
        highest_test_acc1 = v_acc1
print("Max acc:{}".format(highest_test_acc1))
writer.add_scalar("Test/highest_accuracy", highest_test_acc1)
# 关闭SummaryWriter
writer.close()
# 绘制曲线图
plt.figure(figsize=(10, 10))

# 绘制学习率曲线
plt.subplot(2, 1, 1)
plt.plot(epochs, train_lr, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()

# 绘制训练损失、验证损失和测试准确率曲线
plt.subplot(2, 1, 2)
# plt.plot(epochs, train_loss, label="Train Loss")
# plt.plot(epochs, test_loss, label="Test Loss")
plt.plot(epochs, test_acc1, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metrics")
plt.legend()

plt.suptitle("two_class_endo_attn_only_vit_09_fc_and_attn", fontsize=16)

plt.tight_layout()
plt.show()
