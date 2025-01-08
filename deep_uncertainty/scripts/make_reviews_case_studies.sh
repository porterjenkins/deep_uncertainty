for head in ddpn beta_ddpn_0.5 beta_ddpn_1.0 poisson nbinom gaussian immer stirn seitzer_0.5 seitzer_1.0; do
    echo "Making reviews case studies for ${head}"
    python deep_uncertainty/figures/generate_reviews_case_study.py \
        --config-path configs/reviews/${head}.yaml \
        --chkp-path chkp/reviews/${head}/version_0/best_loss.ckpt \
        --save-path deep_uncertainty/figures/artifacts/case-studies/reviews/${head}.pdf
    python deep_uncertainty/figures/generate_reviews_case_study.py \
        --config-path configs/reviews/ensembles/${head}.yaml \
        --save-path deep_uncertainty/figures/artifacts/case-studies/reviews/ensembles/${head}.pdf
done
