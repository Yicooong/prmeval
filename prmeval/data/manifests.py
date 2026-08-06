RBM_1M_OOD = [
    "ykorkmaz_usc_trossen_rfm_usc_trossen",
    "jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm",
    "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top",
    "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking",
    "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking",
    "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all",
]

DATASET_MANIFESTS = {"rbm-1m-ood": RBM_1M_OOD}


def resolve_manifest(name: str, explicit_paths: list[str] | None = None) -> list[str]:
    return list(explicit_paths or DATASET_MANIFESTS.get(name, [name]))
