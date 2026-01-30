# ============================================ Recompute ============================================ 

# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_fr.jsonl

# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_fr.jsonl

# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_fr.jsonl

# ============================================ Naive Reuse ============================================ 

# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_reuse.jsonl

# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_reuse.jsonl

# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_reuse.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_reuse.jsonl

# ============================================ CacheBlend (Top-k value difference selective recompute) ============================================ 

python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15

python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15

python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15

# ============================================ EPIC (Fix-pattern selective recompute) ============================================ 

# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_epic32.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_epic32.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_epic32.jsonl --recompute_type leading-32

# ============================================ Block Attention ============================================ 

# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset musique --output_file results/musique/llama3.1-8b_blockattn.jsonl 
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset samsum --output_file results/samsum/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset multinews --output_file results/multinews/llama3.1-8b_blockattn.jsonl






# ============================================ Recompute ============================================ 

# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_fr_snapkv4x.jsonl

# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_fr_snapkv4x.jsonl

# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_fr_snapkv4x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_fr_snapkv4x.jsonl

# ============================================ Naive Reuse ============================================ 

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_reuse_snapkv4x.jsonl

# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_reuse_snapkv4x.jsonl

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_reuse_snapkv4x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_reuse_snapkv4x.jsonl

# ============================================ CacheBlend (Top-k value difference selective recompute) ============================================ 

python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15

python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15

python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15
python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_vdiff0.15_snapkv4x.jsonl --recompute_type vdiff-0.15

# ============================================ EPIC (Fix-pattern selective recompute) ============================================ 

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_epic32_snapkv4x.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_epic32_snapkv4x.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset musique --output_file results/musique/qwen3-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset wikimqa --output_file results/wikimqa/qwen3-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset hotpotqa --output_file results/hotpotqa/qwen3-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset samsum --output_file results/samsum/qwen3-8b_epic32_snapkv4x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-8B --dataset multinews --output_file results/multinews/qwen3-8b_epic32_snapkv4x.jsonl --recompute_type leading-32

# ============================================ Block Attention ============================================ 

# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset musique --output_file results/musique/llama3.1-8b_blockattn_snapkv4x.jsonl 
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_blockattn_snapkv4x.jsonl
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_blockattn_snapkv4x.jsonl
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset samsum --output_file results/samsum/llama3.1-8b_blockattn_snapkv4x.jsonl
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset multinews --output_file results/multinews/llama3.1-8b_blockattn_snapkv4x.jsonl
