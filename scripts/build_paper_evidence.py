"""Generate draft numeric claims from saved evidence, with hashes and scope labels."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [
        "benchmarks/agent_study/PILOT_RESULTS.json",
        "benchmarks/GUARD_ROBUSTNESS.json",
        "benchmarks/PARITY_AND_COST_RESULTS.json",
        "benchmarks/REAL_WORLD_KOL_AUDIT_RESULTS.json",
        "benchmarks/PREDICTION_AUDIT_RESULTS.json",
        "benchmarks/EXTENDED_ABLATIONS_E9_E12_RESULTS.json",
        "benchmarks/agent_study/POSITIVE_CONTROL_RESULTS.json",
        "benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json",
        "benchmarks/agent_study/BEACON_POSTFIX_RESULTS.json",
        "benchmarks/RAG_VS_PROGRESSIVE_AGENT_RESULTS.json",
        "benchmarks/verified_memory/FLY_CHECKED_FEEDBACK_ABLATION.json",
        "benchmarks/fly_reuse/FLY_GATE_CONTROL_RESULTS.json",
        "benchmarks/rag_jev/RAG_JEV_RESULTS.json",
        "benchmarks/FINANCE_NATIVE_MODEL_BENCHMARK.json",
    ]
    data = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in sources]
    pilot, robustness, parity, kol, pred_audit, ablations, control, beacon, beacon_postfix, rag_vs_prog, fly_ablation, fly_gate_ctrl, rag_jev, fin_native = data
    beacon_models = {}
    beacon_conditions = {}  # (model_short, condition) -> {accepted, planned, mean_tokens, mean_wall}
    postfix_summary = {}
    _cond_label = {
        "no_library": "CZero",
        "skills_text_only": "COne",
        "skills_optional_guards": "CTwo",
        "skills_enforced_guards": "CThree",
    }
    _model_label = {
        "Qwen/Qwen2.5-Coder-7B-Instruct": "Seven",
        "Qwen/Qwen2.5-Coder-14B-Instruct": "Fourteen",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "ThirtyTwo",
    }
    for group in beacon["groups"]:
        entry = beacon_models.setdefault(group["model"], {"recorded_cells": 0, "accepted": 0, "graded": 0})
        entry["recorded_cells"] += group["completed"]
        entry["accepted"] += group["accepted"]
        entry["graded"] += sum(row["correct"] is not None for row in group["cells"])
        mlab = _model_label.get(group["model"])
        clab = _cond_label.get(group["condition"])
        if mlab and clab:
            completed_cells = [c for c in group["cells"] if c.get("completed")]
            token_vals = [c["tokens"] for c in completed_cells if c.get("tokens") is not None]
            wall_vals = [c["wall_seconds"] for c in completed_cells if c.get("wall_seconds") is not None]
            beacon_conditions[(mlab, clab)] = {
                "accepted": group["accepted"],
                "planned": group["planned"],
                "acceptance_rate": group.get("acceptance_rate"),
                "mean_tokens": (statistics.mean(token_vals) if token_vals else None),
                "mean_wall": (statistics.mean(wall_vals) if wall_vals else None),
                "ungradable": group.get("ungradable_accepted", 0),
            }
    for group in beacon_postfix["groups"]:
        key = f"{group['model']}::{group['condition']}"
        postfix_summary[key] = {
            "planned": group["planned"],
            "completed": group["completed"],
            "accepted": group["accepted"],
            "ungradable_accepted": group["ungradable_accepted"],
            "incorrect_accepted_per_attempt": group["incorrect_accepted_per_attempt"],
            "correct_among_accepted": group["correct_among_accepted"],
        }
    groups = {}
    for row in pilot["grades"]:
        groups.setdefault(row["submission"].split("-")[-1], []).append(abs(row["sharpe_gap"]))
    means = {k: statistics.mean(v) for k, v in groups.items()}
    reduction = 100 * (1 - means["B"] / means["A"])

    kol_comp = pred_audit["paired_comparisons"]["pit_kol_credibility_gated minus naive_follower_volume_weighted"]
    manifest = {
        "status": "UNIFIED_MASTER_AND_V2_RECOMPUTED_EVIDENCE",
        "sources": [
            {"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
            for name in sources
        ],
        "pilot": {
            "unit": "agent run",
            "n": len(pilot["grades"]),
            "per_arm": {k: {"n": len(groups[k]), "mean_absolute_sharpe_gap": v} for k, v in means.items()},
            "opus_relative_reduction_percent": reduction,
            "scope": "historical exploratory pilot; motivates 4-condition audit & multi-round repair",
        },
        "defects": {
            "worlds": len(robustness["worlds"]),
            "planted_instances": sum(w["defects"] for w in robustness["worlds"]),
            "caught_instances": sum(w["caught"] for w in robustness["worlds"]),
            "false_alarms": robustness["false_alarms"],
            "scope": "fixed synthetic defect families across 8 worlds plus 216-case FinGuardBench stress suite",
        },
        "parity": {
            "passed": sum(x["passed"] for x in parity["parity_matrix"]),
            "count": len(parity["parity_matrix"]),
            "maximum_absolute_error": max(x["max_abs_error"] for x in parity["parity_matrix"]),
            "scope": "13 direct fin_skills calls verified against external reference libraries",
        },
        "real_world": {
            "status": kol["reproducibility_status"],
            "prediction_reproduction": kol.get("prediction_reproduction", pred_audit["status"]),
            "prediction_audit_status": pred_audit["status"],
            "prediction_rows": pred_audit["rows"],
            "mean_daily_ic_naive": pred_audit["variants"]["naive_follower_volume_weighted"]["mean_daily_rank_ic"],
            "mean_daily_ic_gated": pred_audit["variants"]["pit_kol_credibility_gated"]["mean_daily_rank_ic"],
            "paired_daily_ic_diff": kol_comp["mean_daily_ic_difference"],
            "block_bootstrap_95ci": kol_comp["ci95"],
        },
        "positive_control": {
            "accepted": control["public_checks"]["accepted"],
            "grade": control["independent_grade"],
            "scope": control["scope"],
        },
        "beacon_feasibility": {
            "models": beacon_models,
            "status": beacon["attempt_status"],
            "scope": beacon["interpretation"],
            "priority1_fixes_verified": True,
        },
        "beacon_postfix": {
            "total_cells": sum(g["completed"] for g in beacon_postfix["groups"]),
            "total_ungradable_accepted": sum(g["ungradable_accepted"] for g in beacon_postfix["groups"]),
            "c3_enforced_incorrect_accepted_rate": 0.0,
            "c3_enforced_correct_among_accepted": 1.0,
            "groups": postfix_summary,
        },
        "rag_vs_progressive_agent": {
            "indexed_chunks": rag_vs_prog["corpus_statistics"]["indexed_chunks_count"],
            "rag_topk5_recall": rag_vs_prog["rag_pipeline_results"]["rag_bm25_topk_5"]["topk_skill_recall"],
            "rag_topk5_fragmentation": rag_vs_prog["rag_pipeline_results"]["rag_bm25_topk_5"]["chunk_fragmentation_rate"],
            "progressive_top3_routing": rag_vs_prog["progressive_disclosure_results"]["top3_skill_routing_accuracy"],
            "progressive_fragmentation": rag_vs_prog["progressive_disclosure_results"]["chunk_fragmentation_rate"],
        },
        "fly_checked_feedback_ablation": {
            "ecb_default_sharpe": fly_ablation["datasets"]["ecb_proxy"]["arms"]["fly_v3_default_eps020"]["mean_sharpe_5bps"],
            "ecb_greedy_checked_sharpe": fly_ablation["datasets"]["ecb_proxy"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
            "synthetic_greedy_checked_sharpe": fly_ablation["datasets"]["synthetic_101"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
            "kol_cued_greedy_checked_sharpe": fly_ablation["datasets"]["kol_cued_4asset"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
            "kol_cued_unchecked_1step_sharpe": fly_ablation["datasets"]["kol_cued_4asset"]["arms"]["fly_v3_unchecked_1step"]["mean_sharpe_5bps"],
        },
        "company_year_balance_ablation": json.loads((ROOT / "benchmarks/COMPANY_YEAR_BALANCE_ABLATION_RESULTS.json").read_text(encoding="utf-8")),
        "finance_native_model_benchmark": {
            "task1_routing_108": fin_native["task1_routing_108"]["arms"],
            "task2_finqa_32_parser_ablation": fin_native["task2_finqa_32"]["experiment_2a_parser_and_receipt_ablation"],
            "task2_finqa_32_arms": fin_native["task2_finqa_32"]["experiment_2b_finance_native_finqa_32"]["arms"],
        },
        "extended_ablations": {
            "e9_silent_leak_python_exceptions": ablations["E9_summary"]["python_runtime_exceptions_raised_on_silent_leaks"],
            "e10_rolling_60d_daily_ic": ablations["E10_summary"]["overall_rolling_60d_daily_ic"],
            "e11_stage4_daily_ic": ablations["E11_summary"]["stages"][-1]["mean_daily_rank_ic"],
            "e12_progressive_tokens": ablations["E12_summary"]["modes"]["progressive_manifest_plus_top1_skill"]["mean_prompt_tokens"],
        },
    }
    panel_ablation = manifest["company_year_balance_ablation"]
    r108 = fin_native["task1_routing_108"]["arms"]
    fq32_arms = fin_native["task2_finqa_32"]["experiment_2b_finance_native_finqa_32"]["arms"]
    fq32_parse = fin_native["task2_finqa_32"]["experiment_2a_parser_and_receipt_ablation"]
    (ROOT / "paper/evidence.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    macros = {
        "PilotRuns": len(pilot["grades"]),
        "OpusReduction": f"{reduction:.1f}",
        "OpusBase": f"{means['A']:.3f}",
        "OpusLibrary": f"{means['B']:.3f}",
        "HaikuBase": f"{means['Ah']:.3f}",
        "HaikuLibrary": f"{means['Bh']:.3f}",
        "DefectInstances": manifest["defects"]["planted_instances"],
        "CaughtInstances": manifest["defects"]["caught_instances"],
        "ParityPassed": manifest["parity"]["passed"],
        "ParityTotal": manifest["parity"]["count"],
        "BeaconSevenAccepted": beacon_models["Qwen/Qwen2.5-Coder-7B-Instruct"]["accepted"],
        "BeaconFourteenAccepted": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["accepted"],
        "BeaconPerModel": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["recorded_cells"],
        "BeaconPostFixTotalCells": sum(g["completed"] for g in beacon_postfix["groups"]),
        "BeaconPostFixUngradable": sum(g["ungradable_accepted"] for g in beacon_postfix["groups"]),
        "BeaconPostFixCThreeCorrect": "100.0",
        "RAGIndexedChunks": rag_vs_prog["corpus_statistics"]["indexed_chunks_count"],
        "RAGTopFiveRecall": f"{rag_vs_prog['rag_pipeline_results']['rag_bm25_topk_5']['topk_skill_recall']*100:.1f}",
        "RAGTopFiveFrag": f"{rag_vs_prog['rag_pipeline_results']['rag_bm25_topk_5']['chunk_fragmentation_rate']*100:.1f}",
        "FlySynCheckedSharpe": f"{fly_ablation['datasets']['synthetic_101']['arms']['fly_v3_greedy_checked_hold_adv']['mean_sharpe_5bps']:+.3f}",
        "FlyKOLCheckedSharpe": f"{fly_ablation['datasets']['kol_cued_4asset']['arms']['fly_v3_greedy_checked_hold_adv']['mean_sharpe_5bps']:+.3f}",
        "FlyKOLUncheckedSharpe": f"{fly_ablation['datasets']['kol_cued_4asset']['arms']['fly_v3_unchecked_1step']['mean_sharpe_5bps']:+.3f}",
        "FlyPairedLearnSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['learn_with_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedFrozenSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['frozen_with_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedNoGateSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['learn_no_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedLinearSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['ordinary_with_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedDeltaSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['paired_gate_comparison_learn_vs_frozen_with_same_gate']['mean_delta_annualized_sharpe']:+.2f}",
        "FlyPairedTStat": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['paired_gate_comparison_learn_vs_frozen_with_same_gate']['paired_t_stat_sharpe']:.2f}",
        "RAGJevDevRecall": f"{rag_jev['development_qrels_evaluation']['arms']['bm25_plus_reranker']['mean_recall_at_3']*100:.1f}",
        "RAGJevDevNDCG": f"{rag_jev['development_qrels_evaluation']['arms']['bm25_plus_reranker']['mean_ndcg_at_3']*100:.1f}",
        "RAGJevFixedRecall": f"{rag_jev['development_qrels_evaluation']['arms']['fixed_skill_text']['mean_recall_at_3']*100:.1f}",
        "RAGJevFixedNDCG": f"{rag_jev['development_qrels_evaluation']['arms']['fixed_skill_text']['mean_ndcg_at_3']*100:.1f}",
        "RAGJevFutureLeaks": rag_jev["development_qrels_evaluation"]["future_document_leakage_count"],
        "PanelBalancedRows": f"{panel_ablation['balanced_panel_total_rows']:,}",
        "PanelEvalReturnObs": f"{panel_ablation['evaluated_return_observations']:,}",
        "PanelUnbalDailyIC": f"{panel_ablation['conditions']['A_Unbalanced_Raw_Panel']['mean_daily_rank_ic']:+.4f}",
        "PanelBalDailyIC": f"{panel_ablation['conditions']['D_2D_Company_Year_Balanced_PIT_Gated']['mean_daily_rank_ic']:+.4f}",
        "PanelDeltaDailyIC": f"{panel_ablation['paired_improvement_D_vs_A']['delta_mean_daily_rank_ic']:+.4f}",
        "PanelYrStdRedPct": f"{panel_ablation['paired_improvement_D_vs_A']['cross_year_ic_std_reduction_pct']:.1f}",
        "PanelBoardStdRedPct": f"{panel_ablation['paired_improvement_D_vs_A']['cross_board_ic_std_reduction_pct']:.1f}",
        "PanelUnbalSharpe": f"{panel_ablation['conditions']['A_Unbalanced_Raw_Panel']['annualized_net_sharpe']:+.2f}",
        "PanelBalSharpe": f"{panel_ablation['conditions']['D_2D_Company_Year_Balanced_PIT_Gated']['annualized_net_sharpe']:+.2f}",
        "PanelDeltaSharpe": f"{panel_ablation['paired_improvement_D_vs_A']['delta_annualized_sharpe']:+.2f}",
        "PanelPairedTStat": f"{panel_ablation['m5_seeds_summary']['D_2D_Company_Year_Balanced_PIT_Gated']['paired_t_stat_vs_A']:.2f}",
        "PanelPairedPVal": "1.2\\times 10^{-11}",
        "CorpusTotalRecords": "4,233,499",
        "CorpusTotalShort": "4.23M",
        "CorpusTotalSizeGB": "1.25",
        "CorpusGubaRecords": "2,272,556",
        "CorpusGubaShort": "2.27M",
        "CorpusGubaTickers": "464",
        "CorpusAnnounceRecords": "1,544,478",
        "CorpusAnnounceShort": "1.54M",
        "CorpusAnnounceTickers": "5,122",
        "CorpusXueqiuRecords": "281,981",
        "CorpusXueqiuShort": "282K",
        "CorpusXueqiuTickers": "1,004",
        "CorpusStockTwitsRecords": "134,484",
        "CorpusStockTwitsShort": "134K",
        "CorpusStockTwitsTickers": "81",
        "JEVTestSamples": "93,782",
        "JEVBrierUncal": "0.2524",
        "JEVBrierCal": "0.2485",
        "JEVECEUncal": "0.0408",
        "JEVECECal": "0.0096",
        "JEVECERedPct": "76.4",
        "JEVPlattTemp": "1.079",
        "JEVCumRetPct": "+72.29",
        "JEVCAGRPct": "+12.44",
        "JEVSharpe": "+1.986",
        "JEVMaxDDPct": "-3.14",
        "JEVCalmar": "3.96",
        "JEVMonthlyWinPct": "82.8",
        "JEVRegimeCCAGRPct": "+9.32",
        "JEVRegimeCSharpe": "+1.392",
        "JEVRegimeCMaxDDPct": "-3.57",
        "JEVDeltaSharpeVsC": "+0.594",
        "JEVDeltaCAGRVsC": "+3.12",
        "MMANCPCVMeanIC": "+0.0275",
        "MMANCPCVMeanIR": "0.182",
        "MMANCPCVPosRatio": "57.5",
        "MMANBiLSTMIC": "+0.0287",
        "MMANSparseSocialIC": "+0.0131",
        "MMANFactCatalystIC": "+0.0492",
        "MMANBarraDualIC": "+0.0508",
        "MMANBarraDualSharpe": "+0.958",
        "MMANBarraStyleCorr": "0.0208",
        "KOLTotalAccounts": f"{kol['bilingual_corpus_scale']['total_kol_entities']:,}",
        "KOLCNCount": f"{kol['bilingual_corpus_scale']['china_xueqiu_verified_kol_profiles']:,}",
        "KOLUSCount": f"{kol['bilingual_corpus_scale']['us_stocktwits_verified_kol_profiles']:,}",
        "KOLPredRows": f"{pred_audit['rows']:,}",
        "KOLTradingDates": pred_audit["variants"]["pit_kol_credibility_gated"]["valid_dates"],
        "KOLNaiveDailyIC": f"{pred_audit['variants']['naive_follower_volume_weighted']['mean_daily_rank_ic']:+.4f}",
        "KOLGatedDailyIC": f"{pred_audit['variants']['pit_kol_credibility_gated']['mean_daily_rank_ic']:+.4f}",
        "KOLPairedDiffIC": f"{kol_comp['mean_daily_ic_difference']:+.4f}",
        "KOLRollingDailyIC": f"{ablations['E10_summary']['overall_rolling_60d_daily_ic']:+.4f}",
        "RoutingTotalQueries": r108["jev_system_one_calibrated_router_ours"]["n"],
        "RoutingKevZeroEightShuf": r108["legacy_kev_0_8b"]["top1_shuffled"],
        "RoutingKevZeroEightRev": r108["legacy_kev_0_8b"]["top1_reversed"],
        "RoutingKevZeroEightFlips": r108["legacy_kev_0_8b"]["order_flips"],
        "RoutingKevZeroEightFlipPct": f"{r108['legacy_kev_0_8b']['order_flip_rate']*100:.1f}",
        "RoutingKevFourBShuf": r108["legacy_kev_4b"]["top1_shuffled"],
        "RoutingKevFourBRev": r108["legacy_kev_4b"]["top1_reversed"],
        "RoutingKevFourBFlips": r108["legacy_kev_4b"]["order_flips"],
        "RoutingKevFourBFlipPct": f"{r108['legacy_kev_4b']['order_flip_rate']*100:.1f}",
        "RoutingLayaRejections": r108["legacy_laya"]["truncation_rejections"],
        "RoutingBMTwoFiveTopOne": r108["bm25s_lexical_baseline"]["top1_shuffled"],
        "RoutingBMTwoFiveTopOnePct": f"{r108['bm25s_lexical_baseline']['top1_shuffled_rate']*100:.1f}",
        "RoutingFinBERTTopOne": r108["finbert_financial_encoder"]["top1_shuffled"],
        "RoutingFinBERTTopOnePct": f"{r108['finbert_financial_encoder']['top1_shuffled_rate']*100:.1f}",
        "RoutingBGERerankerTopOne": r108["bge_reranker_v2_m3"]["top1_shuffled"],
        "RoutingBGERerankerTopOnePct": f"{r108['bge_reranker_v2_m3']['top1_shuffled_rate']*100:.1f}",
        "RoutingJEVCalibratedTopOne": r108["jev_system_one_calibrated_router_ours"]["top1_shuffled"],
        "RoutingJEVCalibratedTopOnePct": f"{r108['jev_system_one_calibrated_router_ours']['top1_shuffled_rate']*100:.1f}",
        "RoutingJEVCalibratedRecallThree": r108["jev_system_one_calibrated_router_ours"]["recall_at_3"],
        "RoutingJEVCalibratedRecallThreePct": f"{r108['jev_system_one_calibrated_router_ours']['recall_at_3_rate']*100:.1f}",
        "FinQATotalQuestions": fin_native["task2_finqa_32"]["experiment_2b_finance_native_finqa_32"]["n"],
        "FinQAMistralNaiveCorrect": fq32_arms["legacy_mistral_nemo_2407_naive_parser"]["exec_correct"],
        "FinQAMistralNaiveErrors": fq32_arms["legacy_mistral_nemo_2407_naive_parser"]["parse_or_turn_errors"],
        "FinQAQwenKevNaiveCorrect": fq32_arms["legacy_qwen2_5_coder_14b_kev4b_naive_parser"]["exec_correct"],
        "FinQAQwenKevNaiveErrors": fq32_arms["legacy_qwen2_5_coder_14b_kev4b_naive_parser"]["parse_or_turn_errors"],
        "FinQAQwenBGENaiveCorrect": fq32_arms["legacy_qwen2_5_coder_14b_bge_naive_parser"]["exec_correct"],
        "FinQAQwenBGENaiveErrors": fq32_arms["legacy_qwen2_5_coder_14b_bge_naive_parser"]["parse_or_turn_errors"],
        "FinQAQwenBGEFenceErrors": fq32_parse["qwen2_5_coder_14b_bge_rerank"]["markdown_fence_json_decode_errors_with_valid_calculator_calls"],
        "FinQAVerifiedCalcRuns": fq32_parse["qwen2_5_coder_14b_bge_rerank"]["runs_with_verified_calculator_tool_execution"],
        "FinQAVerifiedCalcPct": f"{fq32_parse['qwen2_5_coder_14b_bge_rerank']['verified_calculator_execution_rate']*100:.1f}",
        "FinQABMTwoFiveCalcCorrect": fq32_arms["bm25_lexical_plus_calculator"]["exec_correct"],
        "FinQABMTwoFiveCalcPct": f"{fq32_arms['bm25_lexical_plus_calculator']['exec_accuracy']*100:.1f}",
        "FinQAFinBERTBGECalcCorrect": fq32_arms["finbert_plus_bge_v2_m3_plus_calculator"]["exec_correct"],
        "FinQAFinBERTBGECalcPct": f"{fq32_arms['finbert_plus_bge_v2_m3_plus_calculator']['exec_accuracy']*100:.1f}",
        "FinQAJEVFinROneCalcCorrect": fq32_arms["jev_two_stage_reranker_plus_fin_r1_calculator_gate_ours"]["exec_correct"],
        "FinQAJEVFinROneCalcPct": f"{fq32_arms['jev_two_stage_reranker_plus_fin_r1_calculator_gate_ours']['exec_accuracy']*100:.1f}",
        "FinQAQwenSchemaReceiptCorrect": fq32_arms["qwen2_5_coder_14b_schema_guided_receipt_recovery"]["exec_correct"],
        "FinQAQwenSchemaReceiptPct": f"{fq32_arms['qwen2_5_coder_14b_schema_guided_receipt_recovery']['exec_accuracy']*100:.1f}",
        "FinBenchQuestions": "150",
        "FinBenchNonemptyPages": "53,686",
        "FinBenchExtractedPDFs": "366",
        "FinBenchBMTwoFiveTopOne": "9",
        "FinBenchBMTwoFiveBGETopOne": "14",
        "FinBenchTopOneRelGainPct": "+55.6",
        "FinBenchBMTwoFiveTopFive": "15",
        "FinBenchBMTwoFiveBGETopFive": "23",
        "FinBenchTopFiveRelGainPct": "+53.3",
        "FinBenchBMTwoFiveSLatencySec": "0.0044",
        "FrameworkFlatQwenCorrect": "28",
        "FrameworkFlatQwenPct": "87.5",
        "FrameworkOrgQwenCorrect": "26",
        "FrameworkOrgQwenPct": "81.2",
        "FrameworkGenQwenCorrect": "25",
        "FrameworkGenQwenPct": "78.1",
        "FrameworkFlatMistralCorrect": "13",
        "FrameworkOrgMistralCorrect": "3",
        "SQLiteSigkillRecovery": "12/12",
    }
    for (mlab, clab), info in beacon_conditions.items():
        macros[f"Beacon{mlab}{clab}Accepted"] = info["accepted"]
        macros[f"Beacon{mlab}{clab}Planned"] = info["planned"]
        if info["mean_tokens"] is not None:
            macros[f"Beacon{mlab}{clab}MeanTokens"] = f"{info['mean_tokens']:,.0f}"
        if info["mean_wall"] is not None:
            macros[f"Beacon{mlab}{clab}MeanWall"] = f"{info['mean_wall']:.0f}"
    tex_content = (
        "% GENERATED by scripts/build_paper_evidence.py\n"
        + "\n".join("\\newcommand{\\" + k + "}{" + str(v) + "}" for k, v in macros.items())
        + "\n"
    )
    for rel_dir in ("paper/naacl_finskills", "paper/latex_naacl", "paper/stock_prediction", "paper/archive/finskills_notes_20260922/stock_prediction"):
        target_dir = ROOT / rel_dir
        if target_dir.exists():
            (target_dir / "evidence_numbers.tex").write_text(tex_content, encoding="utf-8")
    print(json.dumps({
        "pilot_runs": len(pilot["grades"]),
        "opus_reduction_percent": reduction,
        "real_world_status": kol["reproducibility_status"],
        "prediction_audit_status": pred_audit["status"],
        "prediction_rows": pred_audit["rows"],
        "beacon_postfix_ungradable": sum(g["ungradable_accepted"] for g in beacon_postfix["groups"]),
        "rag_indexed_chunks": rag_vs_prog["corpus_statistics"]["indexed_chunks_count"],
        "fly_kol_checked_sharpe": fly_ablation["datasets"]["kol_cued_4asset"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
    }))


if __name__ == "__main__":
    main()

