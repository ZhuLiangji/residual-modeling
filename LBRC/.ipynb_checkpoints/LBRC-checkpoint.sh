# python LBRC.py \
  # --mode evl/encode/decode   # End-to-end evaluation mode: encode and immediately decode in memory
  # --path "${DATA_PATH}"      # Input npz file containing original_data, recons_data, and optionally latent_bit
  # --nrmse "${TARGET}"        # Target final NRMSE for residual correction
  # --block_t "${BLOCK_T}"     # Temporal block size
  # --block_h "${BLOCK_H}"     # Spatial block height
  # --block_w "${BLOCK_W}"     # Spatial block width
  # --level "${LEVEL}"         # Zstd compression level for bit-plane streams
  # --quant_iter "${QUANT_ITER}" # Number of binary-search iterations for block-wise quantization step
  
# Example command:  
python LBRC.py \
  --mode evl \
  --path ./data/e3sm.npz \
  --nrmse 1e-5 \
  --block_t 60 \
  --block_h 120 \
  --block_w 120 
  
# python LBRC.py \
#   --mode encode \
#   --path ./data/e3sm.npz \
#   --stream ./E3SM/1e-5.lbrc \
#   --nrmse 1e-5
  
# python LBRC.py \
#   --mode decode \
#   --path ./data/e3sm.npz \
#   --stream ./E3SM/1e-5.lbrc