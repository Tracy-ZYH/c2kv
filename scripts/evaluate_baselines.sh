# ============================================ Recompute ============================================ 

# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset needle --output_file results/needle/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset gsm8k --output_file results/gsm8k/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset qasper --output_file results/qasper/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset gov_report --output_file results/gov_report/qwen3-4b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset qmsum --output_file results/qmsum/qwen3-4b_fr.jsonl

# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset needle --output_file results/needle/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gsm8k --output_file results/gsm8k/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qasper --output_file results/qasper/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gov_report --output_file results/gov_report/llama3.1-8b_fr.jsonl
# python python/inference/expr_fullcompute.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qmsum --output_file results/qmsum/llama3.1-8b_fr.jsonl

# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset needle --output_file results/needle/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --output_file results/gsm8k/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset qasper --output_file results/qasper/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset gov_report --output_file results/gov_report/qwen2.5-7b_fr.jsonl
# python python/inference/expr_fullcompute.py --model Qwen/Qwen2.5-7B-Instruct --dataset qmsum --output_file results/qmsum/qwen2.5-7b_fr.jsonl

# ============================================ Naive Reuse ============================================ 

# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset needle --output_file results/needle/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset gsm8k --output_file results/gsm8k/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset qasper --output_file results/qasper/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset gov_report --output_file results/gov_report/qwen3-4b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset qmsum --output_file results/qmsum/qwen3-4b_append-suffix16.jsonl

# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset needle --output_file results/needle/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gsm8k --output_file results/gsm8k/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qasper --output_file results/qasper/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gov_report --output_file results/gov_report/llama3.1-8b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qmsum --output_file results/qmsum/llama3.1-8b_append-suffix16.jsonl

# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset needle --output_file results/needle/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --output_file results/gsm8k/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset qasper --output_file results/qasper/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset gov_report --output_file results/gov_report/qwen2.5-7b_append-suffix16.jsonl
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset qmsum --output_file results/qmsum/qwen2.5-7b_append-suffix16.jsonl

# ============================================ CacheBlend (Top-k value difference selective recompute) ============================================ 

# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_vdiff0.15.jsonl --recompute_type vdiff-0.15

# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_vdiff0.15.jsonl --recompute_type vdiff-0.15

# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_vdiff0.15.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_vdiff0.15.jsonl --recompute_type vdiff-0.15

# ============================================ EPIC (Fix-pattern selective recompute) ============================================ 

# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset needle --output_file results/needle/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset gsm8k --output_file results/gsm8k/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset qasper --output_file results/qasper/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset gov_report --output_file results/gov_report/qwen3-4b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset qmsum --output_file results/qmsum/qwen3-4b_epic32.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset needle --output_file results/needle/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gsm8k --output_file results/gsm8k/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qasper --output_file results/qasper/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gov_report --output_file results/gov_report/llama3.1-8b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qmsum --output_file results/qmsum/llama3.1-8b_epic32.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset needle --output_file results/needle/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --output_file results/gsm8k/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset qasper --output_file results/qasper/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset gov_report --output_file results/gov_report/qwen2.5-7b_epic32.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --model Qwen/Qwen2.5-7B-Instruct --dataset qmsum --output_file results/qmsum/qwen2.5-7b_epic32.jsonl --recompute_type leading-32

# ============================================ Block Attention ============================================ 

# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset musique --output_file results/musique/llama3.1-8b_blockattn.jsonl 
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset samsum --output_file results/samsum/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset multinews --output_file results/multinews/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset needle --output_file results/needle/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset gsm8k --output_file results/gsm8k/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset qasper --output_file results/qasper/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset gov_report --output_file results/gov_report/llama3.1-8b_blockattn.jsonl
# python python/inference/expr_blockattention.py --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset qmsum --output_file results/qmsum/llama3.1-8b_blockattn.jsonl





# ============================================ Recompute ============================================ 

# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset needle --output_file results/needle/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset gsm8k --output_file results/gsm8k/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset qasper --output_file results/qasper/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset gov_report --output_file results/gov_report/qwen3-4b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset qmsum --output_file results/qmsum/qwen3-4b_fr_snapkv16x.jsonl

# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset needle --output_file results/needle/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gsm8k --output_file results/gsm8k/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qasper --output_file results/qasper/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset gov_report --output_file results/gov_report/llama3.1-8b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset qmsum --output_file results/qmsum/llama3.1-8b_fr_snapkv16x.jsonl

# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset needle --output_file results/needle/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --output_file results/gsm8k/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset qasper --output_file results/qasper/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset gov_report --output_file results/gov_report/qwen2.5-7b_fr_snapkv16x.jsonl
# python python/inference/expr_fullcompute.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset qmsum --output_file results/qmsum/qwen2.5-7b_fr_snapkv16x.jsonl

# ============================================ Naive Reuse ============================================ 

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_reuse_snapkv16x.jsonl

# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_reuse_snapkv16x.jsonl

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_reuse_snapkv16x.jsonl
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_reuse_snapkv16x.jsonl

# ============================================ CacheBlend (Top-k value difference selective recompute) ============================================ 

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15

# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_vdiff0.15_snapkv16x.jsonl --recompute_type vdiff-0.15

# ============================================ EPIC (Fix-pattern selective recompute) ============================================ 

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset musique --output_file results/musique/qwen3-4b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset wikimqa --output_file results/wikimqa/qwen3-4b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset hotpotqa --output_file results/hotpotqa/qwen3-4b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset samsum --output_file results/samsum/qwen3-4b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen3-4B-Instruct-2507 --dataset multinews --output_file results/multinews/qwen3-4b_epic32_snapkv16x.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset musique --output_file results/musique/llama3.1-8b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset samsum --output_file results/samsum/llama3.1-8b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset multinews --output_file results/multinews/llama3.1-8b_epic32_snapkv16x.jsonl --recompute_type leading-32

# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset musique --output_file results/musique/qwen2.5-7b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset wikimqa --output_file results/wikimqa/qwen2.5-7b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --output_file results/hotpotqa/qwen2.5-7b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset samsum --output_file results/samsum/qwen2.5-7b_epic32_snapkv16x.jsonl --recompute_type leading-32
# python python/inference/expr_reuse.py --compress snapkv --model Qwen/Qwen2.5-7B-Instruct --dataset multinews --output_file results/multinews/qwen2.5-7b_epic32_snapkv16x.jsonl --recompute_type leading-32

# ============================================ Block Attention ============================================ 

# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset musique --output_file results/musique/llama3.1-8b_blockattn_snapkv16x.jsonl 
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset wikimqa --output_file results/wikimqa/llama3.1-8b_blockattn_snapkv16x.jsonl
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset hotpotqa --output_file results/hotpotqa/llama3.1-8b_blockattn_snapkv16x.jsonl
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset samsum --output_file results/samsum/llama3.1-8b_blockattn_snapkv16x.jsonl
# python python/inference/expr_blockattention.py --compress snapkv --model /mnt/nas1/duchuheng/models/ldsjmdy--Tulu3-Block-FT --dataset multinews --output_file results/multinews/llama3.1-8b_blockattn_snapkv16x.jsonl
