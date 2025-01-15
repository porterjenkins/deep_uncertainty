for head in ddpn beta_ddpn_0.5 beta_ddpn_1.0 poisson nbinom gaussian immer stirn seitzer_0.5 seitzer_1.0; do
    echo "Making COCO-people case studies for ${head}"
    for index in 10 12 50 72 84; do
        python deep_uncertainty/figures/generate_coco_people_case_study.py \
            --config-path configs/coco-people/${head}.yaml \
            --chkp-path chkp/coco-people/${head}/version_0/best_loss.ckpt \
            --save-path deep_uncertainty/figures/artifacts/case-studies/coco-people/${head}_${index}.pdf \
            --index ${index}
        python deep_uncertainty/figures/generate_coco_people_case_study.py \
            --config-path configs/coco-people/ensembles/${head}.yaml \
            --save-path deep_uncertainty/figures/artifacts/case-studies/coco-people/ensembles/${head}_${index}.pdf \
            --index ${index}
    done
done
