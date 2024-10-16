HIP_FORCE_DEV_KERNARG=1 DISABLE_ADDMM_HIP_LT=0 TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_VISIBLE_DEVICES=7 numactl --cpunodebind=0 --membind=0 python ragged_hstu_attention_bench.py  --max-seq-len=1264 --actual-seq-len=745 --max-pos-ind=1024 --no-relative-bias
HIP_FORCE_DEV_KERNARG=1 DISABLE_ADDMM_HIP_LT=0 TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_VISIBLE_DEVICES=7 numactl --cpunodebind=0 --membind=0 python ragged_hstu_attention_bench.py  --max-seq-len=1264 --actual-seq-len=745 --max-pos-ind=1024
HIP_FORCE_DEV_KERNARG=1 DISABLE_ADDMM_HIP_LT=0 TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_VISIBLE_DEVICES=7 numactl --cpunodebind=0 --membind=0 python ragged_hstu_attention_bench.py
HIP_FORCE_DEV_KERNARG=1 DISABLE_ADDMM_HIP_LT=0 TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_VISIBLE_DEVICES=7 numactl --cpunodebind=0 --membind=0 python ragged_hstu_attention_bench.py --no-relative-bias

