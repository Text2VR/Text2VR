Directory of pre_checkpoint that mount on docker container

```bash
docker run --gpus all -it \
  -v ~/dreamscene360_docker/data:/workspace/DreamScene360/data \
  -v ~/dreamscene360_docker/output:/workspace/DreamScene360/output \
  -v ~/dreamscene360_docker/pre_checkpoints:/workspace/DreamScene360/pre_checkpoints \
```


# Checkpoints
1. From project home directory, create folder: **pre_checkpoints**
```
mkdir pre_checkpoints
```

2. Download required pretrained model `omnidata_dpt_depth_v2.ckpt` from this [dropbox link](https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=0) into **pre_checkpoints**. (Thanks to [PERF](https://github.com/perf-project/PeRF/tree/master/pre_checkpoints) for providing the models)


# In Container,
1. Submodules

```shell
pip install submodules/diff-gaussian-rasterization-depth # Rasterizer for RGB and depth
pip install submodules/simple-knn
```

2. Download required pretrained models for text2pano:
```
cd stitch_diffusion/pretrained_model
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors -O stable-diffusion-2-1-base.safetensors
cd ../vae
wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -O stablediffusion.vae.pt
cd ..
python download_lora.py
cd ..
```
