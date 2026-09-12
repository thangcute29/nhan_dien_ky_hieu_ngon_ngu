import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
from Cloud_server.Api.context_agent import ContextAgent, VI_DICTIONARY
from Cloud_server.Database.edge_cases.learning_manager import LearningManager
from Data_preparation.Prepare_sequences import frame_indices, load_wlasl_records
from Shared_lib.sequence_utils import normalize_sequence, prepare_sequence


class SequencePipelineTests(unittest.TestCase):
    def test_prepare_sequence_contract_and_missing_hand_mask(self):
        raw = np.zeros((5, 126), dtype=np.float32)
        raw[:, 63:] = np.arange(63, dtype=np.float32) + 10
        raw[:, 63] += np.arange(5, dtype=np.float32)
        prepared = prepare_sequence(raw)
        self.assertEqual(prepared.shape, (30, 126))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertTrue(np.all(prepared[:, :63] == 0))

    def test_normalization_preserves_trajectory(self):
        raw = np.zeros((2, 126), dtype=np.float32)
        raw[:, 63:] = 1
        raw[1, 63::3] += 0.25
        normalized = normalize_sequence(raw)
        self.assertAlmostEqual(float(normalized[1, 63]), 0.25)

    def test_frame_indices_are_bounded_and_fixed_length(self):
        indices = frame_indices(12, 1, 99, 30)
        self.assertEqual(len(indices), 30)
        self.assertGreaterEqual(min(indices), 0)
        self.assertLess(max(indices), 12)

    def test_wlasl_numeric_id_maps_to_gloss(self):
        sequence_dir = Path(config.SEQUENCES_DIR)
        records, invalid = load_wlasl_records(
            Path(config.WLASL_METADATA_PATH),
            Path(config.WLASL_CLASS_LIST_PATH),
            Path(config.WLASL_VIDEOS_DIR),
        )
        self.assertFalse(invalid)
        self.assertEqual(len(records), 2038)
        self.assertTrue(any(r["available"] and r["gloss"] for r in records))

    def test_english_translation_is_identity(self):
        self.assertEqual(ContextAgent().process("apple", "en"), "apple")

    def test_all_wlasl_100_glosses_have_vietnamese_text(self):
        class_names = {
            int(class_id): gloss
            for class_id, gloss in (
                line.split('\t', 1)
                for line in Path(config.WLASL_CLASS_LIST_PATH).read_text(encoding='utf-8').splitlines()
            )
        }
        metadata = json.loads(Path(config.WLASL_METADATA_PATH).read_text(encoding='utf-8'))
        glosses = {class_names[item['action'][0]] for item in metadata.values()}
        self.assertFalse(glosses - set(VI_DICTIONARY))

    def test_feedback_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as root:
            manager = LearningManager()
            manager.unverified_dir = str(Path(root) / "pending")
            manager.labeled_dir = str(Path(root) / "labeled")
            Path(manager.unverified_dir).mkdir()
            Path(manager.labeled_dir).mkdir()
            self.assertFalse(manager.promote_to_labeled("../secret.json", "apple"))


if __name__ == "__main__":
    unittest.main()
