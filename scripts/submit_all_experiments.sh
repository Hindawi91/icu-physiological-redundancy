#!/bin/bash
# =============================================================================
# submit_all_experiments.sh
#
# Submits ALL 15 experiments × 6 models = 90 Slurm jobs
#
# Group 1: Fully Non-Invasive → Blood Pressure (3 experiments)
#   1.  MAP  <- HR, O2Sat, Resp, Temp
#   2.  SBP  <- HR, O2Sat, Resp, Temp
#   3.  DBP  <- HR, O2Sat, Resp, Temp
#
# Group 2: Non-Invasive + Partial BP Sensor → Missing BP (9 experiments)
#   4.  MAP  <- HR, O2Sat, Resp, Temp, DBP
#   5.  MAP  <- HR, O2Sat, Resp, Temp, SBP
#   6.  MAP  <- HR, O2Sat, Resp, Temp, DBP, SBP
#   7.  SBP  <- HR, O2Sat, Resp, Temp, DBP
#   8.  SBP  <- HR, O2Sat, Resp, Temp, MAP
#   9.  SBP  <- HR, O2Sat, Resp, Temp, DBP, MAP
#   10. DBP  <- HR, O2Sat, Resp, Temp, SBP
#   11. DBP  <- HR, O2Sat, Resp, Temp, MAP
#   12. DBP  <- HR, O2Sat, Resp, Temp, SBP, MAP
#
# Group 3: Temperature Reconstruction (1 experiment)
#   13. Temp <- HR, O2Sat, Resp, MAP, SBP, DBP
#
# Group 4: Respiratory & Oxygenation Reconstruction (2 experiments)
#   14. Resp  <- HR, O2Sat, Temp, MAP, SBP, DBP
#   15. O2Sat <- HR, Resp, Temp, MAP, SBP, DBP
#
# Usage:
#   chmod +x submit_all_experiments.sh
#   ./submit_all_experiments.sh
# =============================================================================

PROJECT="/home/firas.hindawi/icu_lab_reconstruction"
PYTHON="/software/conda-envs/py-gpu-3.10.18-cu129/bin/python"
SLURM_DIR="${PROJECT}/slurm_scripts"

mkdir -p ${SLURM_DIR}
mkdir -p ${PROJECT}/logs
mkdir -p ${PROJECT}/checkpoints/models

MODELS="unet1d bilstm lstm_seq2seq tcn transformer conv_lstm"

# -- Experiment definitions: "TARGET|INPUTS" ----------------------------------
declare -a EXPERIMENTS=(
    # Group 1: Fully non-invasive → BP
    "MAP|HR,O2Sat,Resp,Temp"
    "SBP|HR,O2Sat,Resp,Temp"
    "DBP|HR,O2Sat,Resp,Temp"
    # Group 2: Non-invasive + partial BP → missing BP
    "MAP|HR,O2Sat,Resp,Temp,DBP"
    "MAP|HR,O2Sat,Resp,Temp,SBP"
    "MAP|HR,O2Sat,Resp,Temp,DBP,SBP"
    "SBP|HR,O2Sat,Resp,Temp,DBP"
    "SBP|HR,O2Sat,Resp,Temp,MAP"
    "SBP|HR,O2Sat,Resp,Temp,DBP,MAP"
    "DBP|HR,O2Sat,Resp,Temp,SBP"
    "DBP|HR,O2Sat,Resp,Temp,MAP"
    "DBP|HR,O2Sat,Resp,Temp,SBP,MAP"
    # Group 3: Temperature reconstruction
    "Temp|HR,O2Sat,Resp,MAP,SBP,DBP"
    # Group 4: Respiratory & oxygenation
    "Resp|HR,O2Sat,Temp,MAP,SBP,DBP"
    "O2Sat|HR,Resp,Temp,MAP,SBP,DBP"
)

echo "========================================================"
echo "  Submitting all experiments"
echo "  Experiments : ${#EXPERIMENTS[@]}"
echo "  Models      : $(echo $MODELS | wc -w)"
echo "  Total jobs  : $((${#EXPERIMENTS[@]} * $(echo $MODELS | wc -w)))"
echo "========================================================"

total_submitted=0

for experiment in "${EXPERIMENTS[@]}"; do

    TARGET=$(echo $experiment | cut -d'|' -f1)
    INPUTS=$(echo $experiment | cut -d'|' -f2)
    TARGET_TAG=$(echo $TARGET | tr ',' '-')
    INPUT_TAG=$(echo $INPUTS | tr ',' '-')
    RUN_TAG="${INPUT_TAG}_to_${TARGET_TAG}"

    echo ""
    echo "-- Target: ${TARGET} | Inputs: ${INPUTS} --------------------------"

    for model in $MODELS; do

        job_name="icu_${model:0:4}_${TARGET_TAG:0:5}"
        slurm_file="${SLURM_DIR}/${model}_${RUN_TAG}_job.slurm"
        out_file="${PROJECT}/logs/%j_${model}_${RUN_TAG}.out"
        err_file="${PROJECT}/logs/%j_${model}_${RUN_TAG}.err"

        cat > ${slurm_file} << SLURM
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=gpu_x450
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=${out_file}
#SBATCH --error=${err_file}

module load python/py-gpu/3.10.18
cd ${PROJECT}

echo "========================================================"
echo "  Model  : ${model}"
echo "  Target : ${TARGET}"
echo "  Inputs : ${INPUTS}"
echo "  Run tag: ${RUN_TAG}"
echo "========================================================"

${PYTHON} train.py \
    --model ${model} \
    --targets ${TARGET} \
    --inputs ${INPUTS} \
    --epochs 200 \
    --batch_size 128 \
    --seed 42 \
    --no_early_stopping

echo "Done: ${model} -> ${TARGET}"
SLURM

        sbatch ${slurm_file}
        echo "   Submitted: ${model} -> ${RUN_TAG}"
        total_submitted=$((total_submitted + 1))

    done
done

echo ""
echo "========================================================"
echo "  Total jobs submitted: ${total_submitted}"
echo "========================================================"
echo ""
echo "Monitor with:"
echo "   squeue -u \$USER"