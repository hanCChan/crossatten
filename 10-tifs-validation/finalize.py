"""Build concise decision reports from real evidence, with explicit exclusions."""
import json
import numpy as np
import pandas as pd
from stage import HERE, ROOT, ASK_SHA, save


def main():
    decision = json.loads((HERE / "decision_final.json").read_text())
    accepted = json.loads((HERE / "ask_acceptance.json").read_text())
    reg = json.loads((HERE / "registration.json").read_text())
    comparison = pd.read_csv(HERE / "final_comparisons.csv")
    costs = json.loads((HERE / "costs.json").read_text())
    failures = [str(p.relative_to(HERE)) for p in HERE.glob("jobs*.status.json")
                if json.loads(p.read_text()).get("failed")]
    source_count = len(list((HERE / "runs/source").glob("*/metrics.json")))
    ask_count = len(list((HERE / "runs/ask").glob("*/metrics.json")))
    final_count = len(list((HERE / "runs/final").glob("*/metrics.json")))
    ablation_count = len(list((HERE / "runs/ablation").glob("*/metrics.json")))
    totals = source_count + ask_count + final_count + ablation_count
    if totals > 116 or failures:
        raise ValueError("fit failure or budget violation")
    prefix = f"https://github.com/BeechburgPieStar/ASKNet-RFF/blob/{ASK_SHA}"
    protocol = [
        "# Protocol And Leakage Audit", "", "Evidence cutoff: 2026-10-26.",
        f"Candidate gate: **{decision['verdict']}**. Only source-held evidence selected candidate admission.",
        f"New formal fits: {totals} = source{source_count} + ASKNet{ask_count} + final{final_count} + ablation{ablation_count}. "
        "Four bounded smoke fits are separately disclosed. No silent retry or replacement.",
        "S00-S09 and shared libraries remained read-only. Pinned code and controls are hash-sealed.",
        "", "| Item | Frozen Rule |", "|---|---|",
        "| IQ | Non-equalized, 256 samples, first100/TX/RX/day, per-packet power normalization |",
        "| RX roles | Official fixed order, quarters; target never in fit/validation |",
        "| Dates | March1/8/15/23,2021; source days1-2, CR target1-2, CRD target3-4 |",
        "| Source screening | first/last held6 ManyRx or2 ManySig, seeds8101/8102, 40epochs |",
        "| Primary rule | Raw minimum source packet-validation CE, precommitted |",
        "| Candidate | Only continuous reliability tau=.05; no new architecture or loss |",
        "| Source gate | >=3/4 ManyRx folds positive, >=1pp, minRX>=-2pp, both groups/seeds positive; ManySig no systematic loss |",
        "| Formal candidate | Same packet-CE rule for both datasets; final9101/9102/9103 only if GO |",
        "| ASKNet | Official default structure/Adam/200max/plateau/ES10; independent seeds2023-2025 |",
        "| Target release | Training complete; committed checkpoint hash/metrics before separate scoring |",
        "", "The same physical RX appears as source in other folds under published cross-validation. "
        "This is not a new external blind deployment; historical target outcomes were already seen.",
        "Raw pickle reconstruction creates all cached roles inertly and verifies counts/normalization; "
        "it does not use target performance for fitting. Access logs classify subsequent fit/held/target reads.",
        "Historical target diagnostics are descriptive only and were run after algorithm/gate preregistration.",
        "Evidence: `registration.json`, `preparation.json`, `access.jsonl`, immutable jobs and run seals.",
        "Uncertainty: RX-cluster bootstrap conditions on fixed fold models; no population guarantee."]
    (HERE / "PROTOCOL_AND_LEAKAGE_AUDIT.md").write_text("\n".join(protocol) + "\n")
    ask = comparison[comparison.model == "S10_ask"]
    official_logs = accepted["upstream_logs"]
    lines = ["# ASKNet Fair Reproduction", "", f"Official commit: `{ASK_SHA}` (2026-08-12).",
        "24 official-path formal fits completed. No target score was consulted to change a default.",
        "Evidence levels: model/loader/training code and our execution verified; upstream checkpoints unavailable.",
        "Loader acceptance covered every cached cell with exactly zero float32 difference against "
        "official float64 normalization-then-cast. TX/RX/date counts and quarter partitions agree.",
        "Official training functions are called directly with the original CPU DataLoader/RNG, "
        "Adam1e-3/wd0/batch128, Plateau factor.1/patience5/min1e-6, early-stop10 and200 epoch limit.",
        "LSG Gumbel cutoff and complex unitary QR/SKO are unchanged. No our phase encoder is imported into ASKNet.",
        "", "| Dataset | Official Single-Run Log Mean (%) | Fair RX-Equal CRD (%) | Fair Macro-F1 (%) |",
        "|---|---:|---:|---:|"]
    for d in ("ManyRx", "ManySig"):
        released = np.mean([v["packet_acc_pct"] for k, v in official_logs.items() if k.startswith(d + ":")])
        row = ask[(ask.dataset == d) & (ask.task == "CRD")].iloc[0]
        lines.append(f"| {d} | {released:.4f} | {row.accuracy_pct:.4f} | {row.macro_f1_pct:.4f} |")
    lines += ["", "Upstream log metric is packet accuracy, averaged across four single runs; fair metric is RX equal "
        "across4folds/3seeds. Raw/fair packet and RX metrics are retained in target files.",
        "Environment differs: official1.11/CUDA11.3/3080Ti vs2.5/CUDA12.4/A100. Complex QR and random "
        "Gumbel selection can yield different runs; failure to exactly match a published number "
        "does not itself justify retuning.",
        "Baidu pretrained-share access was attempted and not downloaded. No verification of original "
        "checkpoints or numerical paper reproduction is claimed; code-path acceptance is narrower.",
        "Official LoRa uses a different dataset/spectrogram recipe, not Shen P-H5x160. Its number is excluded.",
        "No license file was found in the pinned repository. Third-party source remains external and is not redistributed.",
        "", "Primary sources:",
        f"- [Official model]({prefix}/WiSig/backbones/PatchNet.py)",
        f"- [Official training]({prefix}/WiSig/main.py)",
        f"- [Official loader]({prefix}/WiSig/utils/load_data.py)",
        f"- [Official logs]({prefix}/WiSig/logs/ASKNet/logs)",
        "- [Publisher DOI](https://doi.org/10.1109/LCOMM.2026.3718894). "
        "Crossref metadata was retrieved on2026-10-26 after proxy fallback."]
    (HERE / "ASKNET_FAIR_REPRODUCTION.md").write_text("\n".join(lines) + "\n")
    c = pd.DataFrame(costs["rows"])
    text = ["# Ablations And Costs", "", "Common device, batch128; raw IQ to logits in evaluation mode.",
        c.to_markdown(index=False, floatfmt=".4f"), "", costs["flop_limit"],
        "ASKNet has complex parameters. Parameter ELEMENT count and real-scalar degrees are reported separately; "
        "they must not be treated as identical across real/complex models.",
        "Soft and hard have exactly equal parameter counts and identical initialized states.",
        "No-HSTU cost is measured but performance benefit cannot be inferred from cost alone.",
        "S09 null-phase capacity control and Shen hierarchical-stem/HSTU evidence remain historical, "
        "not fresh WiSig same-config No-HSTU or No-reliability ablation evidence."]
    if not decision["admitted"]:
        text += ["", "Candidate STOP: formal soft/hard and new trained No-HSTU/No-reliability ablations "
                 "were deliberately NOT run. This honors the preregistered gate and saves36+24 fits.",
                 "Therefore neither a new reliability gain nor WiSig HSTU necessity is established by S10."]
    (HERE / "ABLATIONS_AND_COSTS.md").write_text("\n".join(text) + "\n")
    narrative = [
        "# TIFS Novelty Assessment", "", f"Final candidate verdict: **{decision['verdict']}**.",
        "Do not submit S10 as a proven new SOTA algorithm unless the independent source gate, final "
        "paired results and needed mechanism ablations all support it.",
        "", "| Claimed Contribution | Ownership/Status | Required Boundary |", "|---|---|---|",
        "| Physical phase encoding | S08/S09 existing adjacent complex phase; S10 fixed soft reliability is the only new candidate | "
        "Continuity and variance motivation are established; incremental classification utility is experimentally conditional |",
        "| Hierarchical RF tokens + HSTU | Existing framework, not newly invented in S10 | "
        "Shen ablations do not automatically prove same-config WiSig necessity |",
        "| Cross-RX/date/physics evaluation | S10 evidence contribution: audit, ASKNet, pairing, clustered intervals, controlled diagnostics | "
        "Benchmark protocol and analysis are not a new neural architecture |",
        "", "ASKNet differs by trainable spectral cutoff plus unitary low-frequency phase evolution. "
        "Our reliability is fixed local attenuation of adjacent complex phase; no learned Koopman compensation.",
        "Ma et al., *Towards Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification* "
        "(arXiv2512.12070v1,2025-12-12), uses spectrogram, contrastive pretraining and Siamese classification. "
        "Our current simple encoder does not subsume that method or establish priority over receiver-independent RFFI.",
        "The unnamed Liverpool2026 high-order spectral-difference paper remains unverified after targeted "
        "search and previous audit. No invented title/DOI or first-invention claim is made.",
        "", "The manuscript must distinguish a shared hierarchical-token framework from identical signal frontends: "
        "Shen LoRa and WiSig need different frontend adaptations.",
        "No unknown-TX rejection, spoofing or false-accept-rate experiments were run; hardware identity classification "
        "is not complete open-world authentication or attack resistance.",
        "", "Scientific goals of ManyRx>=90% and ManySig>=93% are not guarantees. RX confidence intervals "
        "and a finite comparator set prevent a universal SOTA claim."]
    if not decision["admitted"]:
        narrative += ["", "**STOP the soft reliability candidate.** Preserve S08/S09 results and use "
            "S10's negative paired evidence to constrain the paper's claims. No target-driven extension was started.",
            "Only the original framework and validation/evidence contributions survive; the new soft reliability "
            "must be described as a rejected candidate, not a core confirmed innovation."]
    narrative += ["", "Primary source index:",
        f"- [ASKNet pinned README]({prefix}/README.md)",
        "- [Ma et al. primary preprint](https://arxiv.org/abs/2512.12070v1)",
        "- Historical fair IB-RFF/WaveMLP: `../06-paper-validation/summary/paper/receiver_metrics.csv`.",
        "- Frozen original literature scope: `../09-phase-invariance/LITERATURE_CHECK.md`."]
    (HERE / "TIFS_NOVELTY_ASSESSMENT.md").write_text("\n".join(narrative) + "\n")
    save(HERE / "execution.json", dict(verdict=decision["verdict"], new_formal_fits=totals,
        source_fits=source_count, ask_fits=ask_count, final_fits=final_count,
        ablation_fits=ablation_count, smoke_fits=4, failures=failures,
        target_driven_extension=False, global_sota_claim=False,
        checkpoints_preserved=True, private_historical_weights_missing=True))
    title = ["# S10 Final Decision", "", f"**{decision['verdict']}** for the sole soft-reliability candidate.",
        f"{totals} formal fits and4 smoke fits; no failed fits or automatic retries.",
        "Results: `SOURCE_SCREENING.md`, `FINAL_CR_CRD_COMPARISONS.md`, "
        "`TIFS_NOVELTY_ASSESSMENT.md`. No S00-S09/shared-library changes.",
        "No global SOTA or guaranteed90% claim. Outcomes and exclusions are fully disclosed."]
    (HERE / "RESULTS.md").write_text("\n".join(title) + "\n")
    print("Final reports written:", decision["verdict"], "fits", totals)


if __name__ == "__main__":
    main()
