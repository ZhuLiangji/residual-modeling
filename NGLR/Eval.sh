# python Eval.py \
#   --mode encode \
#   --engine gpu \
#   --path /blue/ranka/zhu.liangji/ALatent_check/data/jhtdb.npz \
#   --ckpt_path ./JHTDB/nglr_1e-6.pt \
#   --correction_path ./output/jhtdb_1e-6.nglr \
#   --device cuda \
#   --block_batch 96 
  
python Eval.py \
  --mode decode \
  --engine gpu \
  --path /blue/ranka/zhu.liangji/ALatent_check/data/jhtdb.npz \
  --ckpt_path ./JHTDB/nglr_1e-5.pt \
  --correction_path ./output/jhtdb_1e-5.nglr \
  --device cuda \
  --block_batch 96 \
  # --save_decode \
  # --decode_out ./output/jhtdb_1e-5_decoded.npz