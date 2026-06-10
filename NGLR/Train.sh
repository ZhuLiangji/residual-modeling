python Train.py \
  --path /blue/ranka/zhu.liangji/ALatent_check/data/jhtdb.npz \
  --nrmse 1e-6 \
  --ckpt_path ./JHTDB/nglr_1e-6.pt \
  --block_t 60 \
  --block_h 128 \
  --block_w 128 \
  --train_epochs 60 \
  --device cuda
