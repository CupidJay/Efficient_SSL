CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py \
  -a cifar_resnet18 \
  --lr 1.0 --moco-t 0.2 --moco-m 0.5 \
  --batch-size 512 \
  --epochs 200 --cos \
  --lambda-kl 20.0 --pow 0.5 \
  --dim 2048 \
  --k 10000 \
  --cutmix --alpha 0.5 \
  --savedir /opt/caoyh/logs_singleview/cifar10/r18/LFP_old_100ep_lr_1.0_T_0.5_bs_512_m_0.5_lambda_20.0_dim_2048_cutmix_0.5 \
  --dataset cifar10 /opt/Dataset/cifar10