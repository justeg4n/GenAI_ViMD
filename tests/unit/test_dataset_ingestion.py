"""
Unit Tests — Dataset Ingestion
"""

import io
import pytest
import numpy as np


class TestViMDLoader:
    """Unit tests for ViMDLoader."""

    def test_province_to_region_mapping(self):
        from dataset.ingestion.vimd_loader import PROVINCE_TO_REGION
        assert "hanoi" in PROVINCE_TO_REGION
        assert PROVINCE_TO_REGION["hanoi"] == "North"
        assert PROVINCE_TO_REGION.get("hochiminh") == "South"

    def test_decode_audio_hf_format(self):
        from dataset.ingestion.vimd_loader import ViMDLoader
        loader = ViMDLoader()
        # Simulate HF audio dict
        audio_array = np.random.randn(16000).astype(np.float32) * 0.1
        audio_field = {"array": audio_array, "sampling_rate": 16000, "path": "test.wav"}
        decoded, sr = loader._decode_audio(audio_field)
        assert sr == 16000
        assert len(decoded) == 16000
        assert decoded.dtype == np.float32

    def test_decode_audio_resamples(self):
        from dataset.ingestion.vimd_loader import ViMDLoader
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not available")
        loader = ViMDLoader(target_sr=16000)
        audio_array = np.random.randn(8000).astype(np.float32) * 0.1
        audio_field = {"array": audio_array, "sampling_rate": 8000, "path": "test.wav"}
        decoded, sr = loader._decode_audio(audio_field)
        assert sr == 16000
        assert len(decoded) == 16000

    def test_build_metadata(self):
        from dataset.ingestion.vimd_loader import ViMDLoader
        loader = ViMDLoader()
        row = {
            "filename": "HN_001", "text": "Xin chào",
            "speakerID": "spk_HN_001", "gender": 0,
            "province": "HN", "region": "North",
            "length": 3.5, "set": "train"
        }
        meta = loader._build_metadata(row)
        assert meta["transcript"] == "Xin chào"
        assert meta["province"] == "HN"
        assert meta["gender"] == 0
        assert meta["language"] == "vi"
        assert meta["duration"] == 3.5


class TestEthnicCorpusLoader:
    """Unit tests for EthnicCorpusLoader."""

    def test_seed_vocabulary_loads(self):
        from dataset.ingestion.ethnic_corpus_loader import EthnicCorpusLoader, EDE_VI_SEED, BAHNAR_VI_SEED
        loader = EthnicCorpusLoader(include_seed=True)
        dd = loader.load_all()
        assert "ede_vi" in dd
        assert "bah_vi" in dd
        assert len(dd["ede_vi"]) >= len(EDE_VI_SEED) * 2
        assert len(dd["bah_vi"]) >= len(BAHNAR_VI_SEED) * 2

    def test_no_seed_empty_without_dir(self):
        from dataset.ingestion.ethnic_corpus_loader import EthnicCorpusLoader
        loader = EthnicCorpusLoader(include_seed=False)
        dd = loader.load_all()
        assert len(dd["ede_vi"]) == 0
        assert len(dd["bah_vi"]) == 0

    def test_pairs_have_required_fields(self):
        from dataset.ingestion.ethnic_corpus_loader import EthnicCorpusLoader
        loader = EthnicCorpusLoader(include_seed=True)
        dd = loader.load_all()
        row = dd["ede_vi"][0]
        assert "source_lang" in row
        assert "target_lang" in row
        assert "source_text" in row
        assert "target_text" in row
