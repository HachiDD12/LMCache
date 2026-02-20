#!/bin/bash
#SBATCH --job-name=simulate_cache
#SBATCH --output=/home/zhuofanc/zzz_django_logs/simulate_cache_run.out
#SBATCH --error=/home/zhuofanc/zzz_django_logs/simulate_cache_run.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00

# Your job commands go here
source /usr/share/Modules/init/bash
source ~/Cacheblend/.venv/bin/activate
cd ~

# Default log file if not provided as argument
LOG_FILE=${1:-/home/zhuofanc/chat_requests_v037_0102_20251111_055647_django_blend_chunk.json}
OUTPUT_FILE=${2:-/home/zhuofanc/zzz_django_logs/simulate_cache_results.txt}

# Run the cache simulation script
# Output is directed to stdout (captured by SLURM) and summary to stderr
uv run python3 ~/Cacheblend/LMCache/examples/blend_kv_v1/simulate_cache.py \
    --log-file "${LOG_FILE}" \
    --model "mistralai/Devstral-Small-2507" \
    --blend-special-str " # # " \
    > "${OUTPUT_FILE}" 2>&1

# print timestamp
echo "Run completed at: $(date)"

