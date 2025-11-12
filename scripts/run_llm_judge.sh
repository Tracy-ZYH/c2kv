export PYTHONPATH=`pwd`/python/inference:$PYTHONPATH

python ./scripts/llm_judge.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --dataset amap --prediction results/amap/amap-qwen-32b_fr.json --api_url http://0.0.0.0:8000/v1 --prompt_file configs/llm_judge_prompts/coverage_zh.json --output results/amap/llm_judge/recompute.jsonl --max_concurrent 4
python ./scripts/llm_judge.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --dataset amap --prediction results/amap/amap-qwen-32b_reuse.json --api_url http://0.0.0.0:8000/v1 --prompt_file configs/llm_judge_prompts/coverage_zh.json --output results/amap/llm_judge/fullreuse.jsonl --max_concurrent 4
python ./scripts/llm_judge.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --dataset amap --prediction results/amap/amap-qwen-32b_epic16.json --api_url http://0.0.0.0:8000/v1 --prompt_file configs/llm_judge_prompts/coverage_zh.json --output results/amap/llm_judge/epic16.jsonl --max_concurrent 4
python ./scripts/llm_judge.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --dataset amap --prediction results/amap/amap-qwen-32b_cacheblend.json --api_url http://0.0.0.0:8000/v1 --prompt_file configs/llm_judge_prompts/coverage_zh.json --output results/amap/llm_judge/cacheblend.jsonl --max_concurrent 4

python ./scripts/extract_judge_score.py --score_dir results/amap/llm_judge --output_path results/amap/llm_judge/score.npz &
