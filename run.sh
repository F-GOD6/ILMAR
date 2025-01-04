#!/bin/bash

PYTHON_SCRIPT="train_imitation.py"

# 算法列表
ALGOS=("ilmar")
ENVS=("Humanoid-v2")
SEEDS=(2023)
DEVICES=(0 1 2 3)
MAX_CONCURRENT=12
# ALPHAS=(0.0 0.05 0.3 0.7 1.0)
# BETAS=(0.0 0.01 0.15 0.5 1.0)
ALPHAS=(1)
BETAS=(1)
declare -A device_task_count  
declare -A task_to_device     

for device in "${DEVICES[@]}"; do
    device_task_count[$device]=0
done

running_jobs=0

allocate_gpu() {
    local min_tasks=10000
    local selected_device=${DEVICES[0]}
    for device in "${DEVICES[@]}"; do
        if [ ${device_task_count[$device]} -lt $min_tasks ]; then
            min_tasks=${device_task_count[$device]}
            selected_device=$device
        fi
    done
    echo $selected_device
}

for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                for beta in "${BETAS[@]}"; do
                    while [ "$running_jobs" -ge "$MAX_CONCURRENT" ]; do
                        wait -n
                        running_jobs=$((running_jobs - 1))

                        for pid in "${!task_to_device[@]}"; do
                            if ! kill -0 "$pid" 2>/dev/null; then
                                completed_device=${task_to_device[$pid]}
                                device_task_count[$completed_device]=$((device_task_count[$completed_device] - 1))
                                unset task_to_device[$pid]
                            fi
                        done
                    done

                    selected_device=$(allocate_gpu)
                    echo "Running: algo=$algo, env=$env, seed=$seed, alpha=$alpha, beta=$beta, device=$selected_device"
                    CUDA_VISIBLE_DEVICES=$selected_device python $PYTHON_SCRIPT \
                        --algo "$algo" \
                        --env-id "$env" \
                        --seed "$seed" \
                        --alpha "$alpha" \
                        --beta "$beta" &


                    pid=$!
                    task_to_device[$pid]=$selected_device
                    device_task_count[$selected_device]=$((device_task_count[$selected_device] + 1))
                    running_jobs=$((running_jobs + 1))
                done
            done
        done
    done
done


wait
echo "All processes finished."
