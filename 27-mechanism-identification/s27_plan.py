"""Future B/C plans use architecture/receiver identities, no RF observations."""
import hashlib
from s27_common import HERE,P,write,now


def main():
    jobs=lambda arms:[dict(arm=a,fold=f,seed=s) for f in (0,1) for s in (201,202,203) for a in arms]
    common=dict(time=now(),execution_status='PLANNED_NOT_STARTED',automatic_fit_budget=0,
        seeds=[201,202,203],checkpoint='minimum clean permitted training-RX equal validation CE',
        training=dict(epochs=200,batch=64,optimizer='AdamW',lr=.001,weight_decay=.0005,warmup_epochs=5,
            augmentation='AWGN0..50dB',precision='original FP32 network; matmulTF32off/cuDNN TF32on'),
        forbidden=['Confirm','old blind','new target adaptation','replacement seeds','target-score-driven design changes'],
        history='All RXs have research history; none becomes a fresh independent acquisition.',
        launch_condition='A complete evidence report first; no automatic execution in this turn. Any later launch must retain this plan, validate checkpoints/streams and commit its execution authorization before source fitting.')
    b=dict(common,stage='S27_B_TOKENIZER_BACKEND_INTERACTION',planned_new_fits=12,jobs=jobs(['H0_P','LP_P']),
        factors=dict(tokenizer=['H0','LP'],backend=['original2xPlainMHA','2xTokenMLP then unchanged finalLN/mean/classifier']),
        new_readout=dict(block='LN(x+Linear128to64(GELU(Linear64to128(x))))',depth=2,
            mha_params_per_block=16768,mlp_params_per_block=16704,whole_model_parameter_delta=-128,
            width_rule='Hidden width128 is nearest integer matching 16768 to129*h+192; chosen without data.',
            token_shape=[32,64],input_projection_position_preLN_finalLN_classifier='unchanged',
            limits='No token interaction until pooling; MLP nonlinear depth differs; capacity and compute are not identical. Interaction is a backend-specific accuracy contrast, not identification of encoded information.'),
        initialization='Surviving frontend/tokenizer/input_proj/position/LNs/classifier match each corresponding original initialization. Identical new MLP heads acrossH0/LP use CPU stable_seed(S27B-Readout,data_seed), preserve other RNG.',
        source_packets='Original5RX/fold,8000train2000sourceval. Same frozen selected rows/labels and source order asS26.',
        reuse='SixH0 and sixS26LP MHA checkpoints are legal only after exact checkpoint/config/source-stream/precision recovery; no MHA refitting.',
        primary='(Acc(H0_M)-Acc(LP_M))-(Acc(H0_P)-Acc(LP_P)) inpp',
        statistics=dict(pairing='samefold/seed/RX in all4 cells; first average3seeds within each of10uniqueRX',
            bootstrap_replicates=10000,bootstrap_seed=27028,
            positive_signal='interaction mean>=0.5pp,conditional lowerCI>0,>=5/6positive fit interactions,both fold means positive',
            uncertainty='Conditional on historical models, chosen nonattention readout and current10RX; no fresh acquisition or architecture search uncertainty.'),
        outcomes=dict(positive='H0 relative benefit is larger for this MHA backend than this MLP backend; not universal attention necessity.',
            negative='No stable positive interaction; tokenizer gain may persist without attention or depend on another factor.',
            uncertain='Mixedfit/fold signs or small effects: report all4 cells and intervals; no alternativehead search.'))
    write(HERE/'PLAN_B.json',b)
    splits=[]
    for f in P['folds']:
        ranked=sorted(f['source'],key=lambda rx:hashlib.sha256(f"S27C-HeldRX-v1|{f['fold']}|{rx}".encode()).hexdigest())
        held=ranked[0];fit=[rx for rx in f['source'] if rx!=held]
        assert len(fit)==4 and held not in fit
        splits.append(dict(fold=f['fold'],training_receivers=fit,held_source_receiver=held,
            deterministic_ranking=ranked,held_selection='Minimum SHA256(S27C-HeldRX-v1|fold|rx); receiver names only, no performance or waveform selection.'))
    c=dict(common,stage='S27_C_SOURCE_LOSO_PILOT',planned_new_fits=12,jobs=jobs(['H0','LP']),splits=splits,
        source_packets='FourfitRX: first160of fixed200rows/TXtrain and last40val;6400train1600sourceval. HeldRX all200/TX=2000packets only after all12 models/configs sealed.',
        steps_per_epoch=100,total_steps_per_fit=20000,selection='HeldRX unavailable to optimizer, normalization updates, scheduler and checkpoint selection.',
        source_seed='stable_seed(S27C-H4x160,fold,seed); paired data/order/noise withinH0/LP; no reuse of5RX-trained weights.',
        checkpoint_initialization='Fresh H0/LP at paired data_seed, preserving original shared initialization; no pretrained5RX checkpoint.',
        primary='G=Acc(source_validation)-Acc(held_source); deltaG=G(H0)-G(LP). Report paired held accuracy difference and both source/held accuracies together.',
        secondary=['macroF1','perTXrecall','sourceRX-equalCE','heldCE','perfit and perheldRX accuracies'],
        fixed_features=dict(layers=['mean tokenizer output','mean preMHA','finalclassifier input'],
            packets='last40fixedrows/TX/RX for both source-val and held diagnostic400packets; heldaccuracy stillall2000',
            normalization='perpacketL2epsilon1e-12',metrics=['TXcentroidscatter','sameTXacrossRXscatter','withinTXRXscatter','Jc','Jw','trueclass logitmargin'],
            scope='All layers/allTX/allseeds; source and held distributions separately; ratios are geometry, not mutual information or invariance.'),
        statistics=dict(receiver_units=2,bootstrap_or_population_significance=False,
            report='All6pairedfit differences,3seedmeans perheldRX,fitSD; seeds are not6independent receivers.',
            consistent_pilot_signal='HeldH0-LP>=0.5pp anddeltaG<=-0.5pp in both receiver/fold means,>=5/6fit differences have both favorable signs.',
            ceiling_caveat='Source accuracy ceilings can distort gaps; a smaller gap alone does not show better held identification. Report fourcell accuracies and class-error patterns.'),
        outcomes=dict(positive='Limited evidence of smaller conditional known-to-held degradation for these2RX, with improved held accuracy; not general population proof.',
            negative='If source and held improve similarly without smallergap, describe stronger overall discrimination rather than RX-specific suppression.',
            uncertain='TwoRX or conflictingseeds/folds cannot support population claims; no automatic heldRX substitution or expansion.'),
        expansion='No automatic extraRX/fits. Any later expansion needs a new frozen execution budget and retains allPilot results; cannot depend on selecting favorable RX.')
    write(HERE/'PLAN_C.json',c)
    print('B/C frozen design files written; automaticfitbudget0. HeldC:',[(r['fold'],r['held_source_receiver']) for r in splits],flush=True)


if __name__=='__main__':main()
