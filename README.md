# DSCache
This repository provides official implementation of:
> **Decouple and Cache: KV Cache Construction for Streaming Video Understanding (ICML 2026)**  
>Zhanzhong Pang, Dibyadip Chatterjee, Fadime Sener, Angela Yao

## Abstract
we propose Decoupled Streaming Cache (DSCache), a training-free cache construction mechanism that adapts pretrained offline models to streaming settings. DSCache maintains a cumulative past KV cache while constructing a separate instant cache on-demand, decoupled from past caches to preserve the informativeness of recent inputs. To enable position extrapolation beyond the training length, DSCache further incorporates a position-agnostic encoding strategy, ensuring KV caches to support unseen positions and preventing position overflow.

## Structure

```
.
├── datasets        processed benchmarks
├── Baseline        offline MLLMs evaluation
├── Uniform_Cache   uniform streaming cache evaluation
├── DSCache         our DSCache
└── llava           code for Llava-OV
```

## Preparation
- Setup: 1x Nvidia A100 (80GB)
- Prepare the conda environment based on `requirements.txt`
- Download benchmarks under `datasets/`
  - [StreamingBench](https://github.com/thunlp-mt/streamingbench)
  - [OVOBench](https://github.com/joeleelyf/ovo-bench)
  

## Evaluation

```bash
# more details can be found in run.sh.

cd DSCache

# Supported model: llava_ov_7b, qwen2.5-vl-7b
# Argument: context_len - cumulative cache size; frame_len - buffer size; stride - last frame resolution; continuous - use continuous position index or not; ltype - sampling stride for cumulative cache

python StreamingBench_last.py --model_name llava --context_len 16 --ltype ls_1 --frame_len 4 --stride 1.0 --continuous

```

## Citation

```latex
@article{pang2026decouple,
  title={Decouple and Cache: KV Cache Construction for Streaming Video Understanding},
  author={Pang, Zhanzhong and Chatterjee, Dibyadip and Sener, Fadime and Yao, Angela},
  journal={arXiv preprint arXiv:2605.01858},
  year={2026}
}
```
