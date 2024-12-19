if [ -z "$1" ]; then
    echo "Usage: $0 <dataset-name>"
    exit 1
fi

dataset=$1
configs_dir="configs/${dataset}/ensembles"

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
    echo "Evaluating ${head} ensemble on ${dataset}"
    python deep_uncertainty/evaluation/eval_ensemble.py \
        --config ${configs_dir}/${head}.yaml
    rm results/${dataset}/ensembles/${head}_ensemble/ensemble_config.yaml
done
