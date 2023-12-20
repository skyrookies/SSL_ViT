import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process log files.')

    # 添加命令行参数
    parser.add_argument('--log_file', type=str, help='Path to log file')
    parser.add_argument('--log_dir', type=str, help='Directory for log output')

    return parser.parse_args()


def process_log(log_file, log_dir):
    # 读取日志文件为DataFrame
    df = pd.read_json(log_file, lines=True)

    # 提取所需的列
    epochs = df["epoch"]
    train_lr = df["train_lr"]
    train_loss = df["train_loss"]
    test_loss = df["test_loss"]
    test_acc1 = df["test_acc1"]

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


if __name__ == '__main__':
    args = parse_args()
    process_log(args.log_file, args.log_dir)
