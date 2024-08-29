# build xformers
git clone https://github.com/facebookresearch/xformers
cd xformers
export TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7"
pip install -e . -vvv


# build DeepSpeed
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed

modify `op_builder/builder.py` with `cxx_args`  '-fPIC'

mkdir -p deepspeed/ops/spatial/
export NVCC_PREPEND_FLAGS="--forward-unknown-opts"

export TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7"

DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_FP_QUANTIZER=0 DS_BUILD_GDS=0 \
DS_BUILD_CUTLASS_OPS=0 DS_BUILD_RAGGED_DEVICE_OPS=0 DS_BUILD_EVOFORMER_ATTN=0 \
DS_BUILD_SPARSE_ATTN=0 \
MAX_JOBS=16 pip install -e . --global-option="build_ext" --global-option="-j16"


CUDA_VISIBLE_DEVICES=0 \
python main.py --base configs/example_training/txt2img-clipl_debug.yaml








