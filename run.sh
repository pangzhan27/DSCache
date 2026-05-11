#!/bin/bash

cd Baseline
python StreamingBench.py --model_name llava
python OVOBench.py --model_name llava

python StreamingBench.py --model_name qwen
python OVOBench.py --model_name qwen


cd Uniform_Cache
python StreamingBench_stream.py --model_name llava --context_len 32
python OVOBench_stream.py --model_name llava  --context_len 32

# Since Qwen processes two frames per encoding, we set the context length to half of that used in LLaVA.
python StreamingBench_stream.py --model_name qwen --context_len 16
python OVOBench_stream.py --model_name qwen  --context_len 16


cd DSCache
python StreamingBench_last.py --model_name llava --context_len 16 --ltype ls_1 --frame_len 4 --stride 1.0 --continuous
python StreamingBench_last.py --model_name qwen --context_len 12 --ltype ls_2 --frame_len 4 --stride 2.0 --continuous

python OVOBench_last.py --model_name llava --context_len 24 --ltype ls_1 --frame_len 4 --stride 1.0 --continuous
python OVOBench_last.py --model_name qwen --context_len 12 --ltype ls_2 --frame_len 4 --stride 2.0 --continuous



