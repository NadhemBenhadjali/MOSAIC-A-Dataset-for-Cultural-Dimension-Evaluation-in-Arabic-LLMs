"""Project constants for MOSAIC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class Dimension:
    """Metadata for one Hofstede dimension in the MOSAIC dataset."""

    code: str
    name: str
    source_file: str
    option_1_label: str
    option_2_label: str
    reported_label: str
    reported_option: int


DIMENSIONS: dict[str, Dimension] = {
    "idv": Dimension(
        code="idv",
        name="Individualism vs. Collectivism",
        source_file="individualistic_vs_collectivist_prompts.csv",
        option_1_label="individualism",
        option_2_label="collectivism",
        reported_label="collectivism",
        reported_option=2,
    ),
    "lto": Dimension(
        code="lto",
        name="Long-Term vs. Short-Term Orientation",
        source_file="long_term_vs_short_term_orientation_prompts.csv",
        option_1_label="long_term_orientation",
        option_2_label="short_term_orientation",
        reported_label="short_term_orientation",
        reported_option=2,
    ),
    "mas": Dimension(
        code="mas",
        name="Masculinity vs. Femininity",
        source_file="mas_prompts.csv",
        option_1_label="masculinity",
        option_2_label="femininity",
        reported_label="masculinity",
        reported_option=1,
    ),
    "pdi": Dimension(
        code="pdi",
        name="Power Distance Index",
        source_file="power_distance_index_prompts.csv",
        option_1_label="high_power_distance",
        option_2_label="low_power_distance",
        reported_label="hierarchy",
        reported_option=1,
    ),
    "uai": Dimension(
        code="uai",
        name="Uncertainty Avoidance Index",
        source_file="uncertainty_avoidance_prompts.csv",
        option_1_label="high_uncertainty_avoidance",
        option_2_label="low_uncertainty_avoidance",
        reported_label="uncertainty_avoidance",
        reported_option=1,
    ),
}


MODEL_REPOS: dict[str, str] = {
    "AceGPT": "FreedomIntelligence/AceGPT-v1.5-7B-Chat",
    "Allam": "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "Aya": "CohereLabs/aya-expanse-8b",
    "command-r7b": "CohereLabs/c4ai-command-r7b-arabic-02-2025",
    "Falcon": "tiiuae/Falcon3-7B-Instruct",
    "Fanar": "QCRI/Fanar-1-9B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
}


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "AceGPT": "AceGPT v1.5 7B",
    "Allam": "ALLaM 7B",
    "Aya": "Aya 8B",
    "command-r7b": "Command-R7B",
    "Falcon": "Falcon 3 7B",
    "Fanar": "Fanar 9B",
    "llama": "Llama 3.1 8B",
    "Qwen": "Qwen 2.5 7B",
}


ARABIC_COUNTRIES: list[str] = [
    "الجزائر",
    "مصر",
    "العراق",
    "الأردن",
    "الكويت",
    "لبنان",
    "ليبيا",
    "المغرب",
    "قطر",
    "السعودية",
    "سوريا",
    "تونس",
    "الإمارات",
]


SEEDS: list[int] = [42, 22, 36, 94, 10]
