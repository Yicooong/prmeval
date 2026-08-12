import unittest

from dataset_unify.converters import get_dataset_converter
from dataset_unify.registry import DatasetConverterRegistry, MatchMode


class DatasetConverterRegistryTest(unittest.TestCase):
    def test_matches_case_insensitively(self):
        registry = DatasetConverterRegistry[str]()
        registry.register("demo", patterns=("demo",))("converter")

        self.assertEqual(registry.get("My_DEMO_Dataset"), "converter")

    def test_supports_prefix_matching(self):
        registry = DatasetConverterRegistry[str]()
        registry.register("oxe", patterns=("oxe_",), match_mode=MatchMode.PREFIX)("converter")

        self.assertEqual(registry.get("OXE_language_table"), "converter")
        with self.assertRaises(KeyError):
            registry.get("custom_oxe_language_table")

    def test_rejects_duplicate_registration(self):
        registry = DatasetConverterRegistry[str]()
        registry.register("demo")("first")

        with self.assertRaisesRegex(ValueError, "Duplicate dataset converter"):
            registry.register("DEMO")("second")

    def test_unknown_dataset_lists_registered_converters(self):
        registry = DatasetConverterRegistry[str]()
        registry.register("demo")("converter")

        with self.assertRaisesRegex(KeyError, "Available converters: demo"):
            registry.get("unknown")

    def test_builtin_catalog_resolves_representative_dataset_names(self):
        dataset_names = (
            "libero_90",
            "AgiBotWorld",
            "oxe_language_table",
            "fino_net",
            "mit_franka_p-rank_rfm",
            "utd_so101_clean_policy_ranking_wrist",
            "usc_koch_human_robot_paired_robot",
            "rbm-1m-ood",
        )

        for dataset_name in dataset_names:
            with self.subTest(dataset_name=dataset_name):
                self.assertTrue(callable(get_dataset_converter(dataset_name)))


if __name__ == "__main__":
    unittest.main()
