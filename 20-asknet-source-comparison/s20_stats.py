"""Fixed package comparison, not a new algorithm or component attribution."""
import numpy as np
import s20_core
from s19_metrics import score,cluster_comparison

def thresholds():
    return dict(clear_advantage=dict(mean_fold_pp=1.,positive_folds=3,min_worst_paired_rx_pp=-2.,unique_rx_ci_lower_gt=0.),
                practical_similarity=dict(margin_pp=1.,criterion='both RX-identity95%CI endpoints inside[-1,+1] and abs(fold mean)<1; descriptive, not formal population equivalence'),
                seed_scope='one fixed macroseed18101; four effective model/data seeds inherited from S19 folds')

def decision(rows):
    if len(rows)!=8 or {(r['fold'],r['arm']) for r in rows}!={(f,a) for f in range(4) for a in ['B0','ASK']}:
        raise ValueError('Exactly four paired B0/ASK source results required')
    comparison=cluster_comparison(rows,'ASK','B0',seed=20101)
    reverse=cluster_comparison(rows,'B0','ASK',seed=20101)
    def clear(c):return c['mean_fold_gain_pp']>=1 and c['positive_folds']>=3 and c['worst_paired_rx_pp']>=-2 and c['ci95_pp'][0]>0
    ask_clear=bool(clear(comparison));hst_clear=bool(clear(reverse))
    close=bool(abs(comparison['mean_fold_gain_pp'])<1 and comparison['ci95_pp'][0]>=-1 and comparison['ci95_pp'][1]<=1)
    if ask_clear:status='ASKNET_CLEAR_SOURCE_ADVANTAGE'
    elif hst_clear:status='HSTEM_CLEAR_SOURCE_ADVANTAGE'
    elif close:status='PRACTICALLY_CLOSE_ON_THIS_SOURCE_PROTOCOL'
    else:status='INCONCLUSIVE_SOURCE_COMPARISON'
    return dict(status=status,thresholds=thresholds(),ask_clear_advantage=ask_clear,hstem_clear_advantage=hst_clear,
                practical_similarity=close,comparison=comparison,reverse_comparison=reverse,
                next_step='Define separately registered component controls; no new module or fit in S20' if ask_clear else
                          'Audit source validation discrimination/coverage and seed uncertainty; do not choose a new module from this pilot',
                causal_component_identified=False,universal_equivalence_claim=False,real_target_access=False,
                official_recipe_confounded_with_architecture=True,additional_algorithm_training_allowed=False)
