import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import SSL_PVT as SPVTs
import sslutils as utils
from loss import TrainLoss
from dataset import Dataset2
import vit_model as VIT


def get_args_parser():
    parser = argparse.ArgumentParser('SSL_ViT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='pvt_tiny', type=str, choices=['pvt_tiny', 'pvt_small', 'vit', 'vit_21k'],
                        help="""Name of architecture to train.""")

    parser.add_argument('--img_size', default=224, type=int,
                        help="""Size in pixels of input square 2D patches.""")

    parser.add_argument('--out_dim', default=1000, type=int,
                        help="""Dimensionality of the  head output.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the UniMiSS head.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                        help="""Base EMA parameter for teacher update.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help="""Final value (after linear warmup) of the teacher temperature.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False,
                        help="""Whether or not to use half precision for training.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=0.3,
                        help="""Maximal parameter gradient norm if using gradient clipping.""")
    parser.add_argument('--batch_size_per_gpu', default=4, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs during which we keep the output layer fixed. 
                        Typically doing so during the first epoch helps training. 
                        Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0008, type=float,
                        help="""Learning rate at the end of linear warmup (highest LR used during training).""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="""Target LR at the end of optimization.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer.""")

    # Others
    parser.add_argument('--data_path', default='/home/ljtj/Documents/wsl/data_divided_2/train/cin2+/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--list_path2D', default='2D_images.txt', type=str)
    parser.add_argument('--output_dir', default="./weight", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--preweight_path', default="./snapshots", type=str, help='Path of pretrained weight.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--first_flag", default=True, type=bool,
                        help="whether to load the pretrained weight.Ture: on / true / 1 ")
    parser.add_argument("--first_epoch", default=0, type=int,
                        help="whether to load the pretrained checkpoint ")
    parser.add_argument('--freeze_student', default=False, action='store_true',
                        help="""Whether to freeze the student para""")
    parser.add_argument('--freeze_teacher', default=False, action='store_true',
                        help="""Whether to freeze the teacher para""")
    parser.add_argument('--attn-only', action='store_true')
    parser.add_argument('--mlp-only', action='store_true')
    return parser


def train_func(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    # print("git:\n  {}\n".format(utils.get_sha()))
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())), file=f)
    # print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_set2D = Dataset2(args.data_path, args.list_path2D, crop_size=args.img_size)

    data_loader = torch.utils.data.DataLoader(
        train_set2D,
        sampler=torch.utils.data.DistributedSampler(train_set2D, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Data loaded: there are {len(train_set2D)} images.")

    # ============ building student and teacher networks ... ============
    if args.arch == 'pvt_tiny':
        student = SPVTs.pvt_tiny(
            img_size=args.img_size,
            drop_path_rate=0.1,  # stochastic depth
        )
        teacher = SPVTs.pvt_tiny(
            img_size=args.img_size,
        )
    elif args.arch == 'vit_21k':
        student = VIT.VisionTransformer(img_size=args.img_size,
                                        patch_size=16,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12,
                                        representation_size=768,
                                        num_classes=21843,
                                        drop_path_ratio=0.1)
        teacher = VIT.VisionTransformer(img_size=224,
                                        patch_size=16,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12,
                                        representation_size=768,
                                        num_classes=21843)
    elif args.arch == 'vit':
        student = VIT.VisionTransformer(img_size=224,
                                        patch_size=16,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12,
                                        representation_size=None,
                                        num_classes=1000,
                                        drop_path_ratio=0.1)
        teacher = VIT.VisionTransformer(img_size=224,
                                        patch_size=16,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12,
                                        representation_size=None,
                                        num_classes=1000)
    else:
        print('Unknow arch')
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # for name, para in teacher.named_parameters():
    #     除head, fc外，其他权重全部冻结
    # if "attn" not in name or 'head' not in name:
    #     para.requires_grad_(False)
    # else:
    #     print("training student :{}".format(name))
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    # DDP wrapper...
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    trainloss = TrainLoss(
        args.out_dim,
        2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with UniMiSS
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        # linear scaling rule
        args.min_lr,
        args.epochs,
        niter_per_ep=len(data_loader.dataset),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader.dataset),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader.dataset))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": args.first_epoch}
    if args.first_flag:
        utils.start_from_pretrain(
            weights_path=args.preweight_path,
            model_one=student,
            model_two=teacher,
        )
        print("Pre_train weight has been loaded")
    else:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint0009.pth"),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            loss=trainloss,
        )

    start_epoch = to_restore["epoch"]

    for epoch in range(start_epoch, args.epochs):
        train_several_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader, epoch, epoch + 1,
                            optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, args)


def train_several_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader, start_epoch, end_epoch,
                        optimizer, lr_schedule, wd_schedule, momentum_schedule, fp16_scaler, args):
    start_time = time.time()
    print("Start " + " training !")
    if args.attn_only:
        for name_p, p in student.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
                print("training {}".format(name_p))
            else:
                p.requires_grad = False
        # try:
        #     student.head.weight.requires_grad = True
        #     student.head.bias.requires_grad = True
        # except:
        #     student.fc.weight.requires_grad = True
        #     student.fc.bias.requires_grad = True
        try:
            student.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in student.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
    # freeze weight except attention and head
    if args.mlp_only:
        for name, para in student.named_parameters():
            # 除head, fc外，其他权重全部冻结
            if "fc" in name or 'head' in name:
                # or 'block4.1.mlp' in name
                para.requires_grad_(True)
                print("training {}".format(name))
            else:
                para.requires_grad_(False)
    if args.freeze_student:
        for name, para in student.named_parameters():
            # 除attn,head外，其他权重全部冻结
            if "attn" not in name and 'head' not in name:
                para.requires_grad_(False)
            else:
                print("training student :{}".format(name))
    if args.freeze_teacher:
        # freeze weight except attention and head
        for name, para in teacher.named_parameters():
            # 除attn,head外，其他权重全部冻结
            if "attn" not in name and 'head' not in name:
                para.requires_grad_(False)
            else:
                print("training teacher :{}".format(name))
    for epoch in range(start_epoch, end_epoch):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch  ============
        student = nn.parallel.DistributedDataParallel(student.module, device_ids=[args.gpu])

        train_stats = train_one_epoch(student,
                                      teacher,
                                      teacher_without_ddp,
                                      trainloss,
                                      data_loader,
                                      optimizer,
                                      lr_schedule,
                                      wd_schedule,
                                      momentum_schedule,
                                      epoch,
                                      fp16_scaler,
                                      args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss': trainloss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f: f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training ' + ' time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, trainloss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader.dataset) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move  images to gpu
        images = [im.cuda(non_blocking=True) for im in subjects_batch]

        # teacher and student forward passes + compute  loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[0])
            student_output = student(images[1])

            loss = trainloss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSL_ViT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_func(args)
