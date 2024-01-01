# README

the code of my paper

### 1、Data Preparation

​	We use unlabeled endoscopy data for self-supervised training, the data can be download by this  

[link]: https://saras-mesad.grand-challenge.org/

 	After downloading the data, simply filter it and put it into the self-supervised data folder. Use the write_image.py script to write the image data into a directory. 

### 2、SSL-Training

cd ./ssl

```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM main.py \
--arch='vit' \
--data_path='./ssl-data' \
--list_path2D='./endoscope.txt' \
--preweight_path='./vit_base_patch16_224.pth' \
--batch_size_per_gpu=32 \
--epochs 10 \
--lr=5e-6 \
--min_lr=1e-6 \
--num_workers=12 \
--momentum_teacher=0.99 \
--clip_grad=0.3 \
--output_dir='./ssl_vit' \
--first_flag True \
--first_epoch 0 \
--saveckp_freq 1 \
--use_fp16 True \
--attn-only \
--warmup_epochs 0
```

### 3、Fine-tuning

cd ./fine-tune

```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM main.py \
--model 'ssl_vit' \
--batch-size  64 \
--epochs 200 \
--model-ema \
--sched cosine \
--lr 5e-4 \
--warmup-lr 1e-4 \
--min-lr 5e-5 \
--warmup-epochs 20 \
--smoothing 0.1 \
--mixup 0.8 \
--mlp-only \
--data-path './cervical_data' \
--output_dir './new_train' \
--start-point './ssl_vit/best-checkpoint.pth' \
--distributed \
--show-para

python draw_tensorboard.py \
--log_file "new_train//log.txt" \
--log_dir "new_log/"
```







the SSL  part modified based on DINO

```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

the fine-tune part modified based on ConvNext

```
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```

