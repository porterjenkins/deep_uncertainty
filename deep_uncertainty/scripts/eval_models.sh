if [ -z "$1" ] || [ -z "$2" ]; then
    echo "One ore more arguments are missing."
    echo "Usage: $0 <dataset-name> <results-dir>"
    exit 1
fi

dataset=$1
results_dir=$2
configs_dir="configs/${dataset}"

if [ -d "$configs_dir" ]; then
    :
else
    echo "Error: '$dataset' is not a supported dataset."
    exit 1
fi

head_list=()
for file in ${configs_dir}/*.yaml; do
  file_name=$(basename "$file" .yaml)
  head_list+=("$file_name")
done

for head in "${head_list[@]}"; do
    for v in 0 1 2 3 4; do
        echo "Evaluating ${head} model (version ${v}) on ${dataset}"
        python deep_uncertainty/evaluation/eval_model.py \
            --config-path ${configs_dir}/${head}.yaml \
            --log-dir ${results_dir}/${dataset}/${head}/version_${v} \
            --chkp-path chkp/${dataset}/${head}/version_${v}/best_loss.ckpt
    done
done
