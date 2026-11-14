"""
Smart Workstation - On-Device AI Training and Inference Pipeline
Course: 2110356 Embedded System (CEDT), Chulalongkorn University

Requirement M4 & E7:
"Fixed thresholds or simple if-else rules are not an AI component."
This module generates a rich, realistic synthetic training dataset, trains a supervised
Machine Learning model (Random Forest / Decision Tree) with multi-dimensional decision
boundaries, evaluates inference latency (<10ms target), and serializes the model.

Feature Vector (6 Features):
    0: presence (0 or 1 from FSR402 pressure plate)
    1: distance_cm (HC-SR04 face-to-book distance, cm)
    2: lux (BH1750 ambient reading surface illumination, Lux)
    3: temp_c (DS18B20/DHT22 ambient temperature, °C)
    4: sound_db (MAX9814/KY-037 ambient noise level, dB)
    5: sitting_duration_min (Continuous sitting duration, minutes)

Classes:
    0: OPTIMAL_STUDY        - Posture and environment are ergonomic
    1: TEXT_NECK_TOO_CLOSE  - Distance < 28cm (Head too close to book)
    2: SLOUCH_TOO_FAR       - Distance > 65cm while seated (Leaning back excessively)
    3: POOR_LIGHTING        - Lux < 200 Lux (Too dim) or > 1200 Lux (Harsh glare)
    4: NOISY_OR_HOT_FATIGUE - Sound > 60 dB or Temp > 30°C (Environmental fatigue)
    5: BREAK_TIMER_REACHED  - Duration >= 50 min (Prolonged sedentary study)
    6: AWAY                 - Presence == 0 (User left desk)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [AI-Model] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AI-Model")

FEATURE_NAMES = [
    "presence",
    "distance_cm",
    "lux",
    "temp_c",
    "sound_db",
    "sitting_duration_min",
]

CLASS_DEFINITIONS = {
    0: {
        "name": "OPTIMAL_STUDY",
        "action": "Curtain OPEN; Closed-loop LED tuned to ~450 Lux; Normal operation",
        "advice": "Posture and study environment are optimal. Keep it up!",
    },
    1: {
        "name": "TEXT_NECK_TOO_CLOSE",
        "action": "Trigger Curtain CLOSE (drops blocker); Posture warning alert",
        "advice": "Head too close (<28cm)! Straighten your neck and sit back.",
    },
    2: {
        "name": "SLOUCH_TOO_FAR",
        "action": "Trigger Curtain CLOSE; Posture warning alert",
        "advice": "Slouching too far back (>65cm)! Align your spine upright.",
    },
    3: {
        "name": "POOR_LIGHTING",
        "action": "Boost LED PWM to 100%; Alert if lux still <200 or glare >1200",
        "advice": "Unsuitable lighting! Adjust room light or desk lamp to avoid eye strain.",
    },
    4: {
        "name": "NOISY_OR_HOT_FATIGUE",
        "action": "Maintain curtain open; Advise user to ventilate or relocate",
        "advice": "High room noise (>60dB) or heat (>30°C) causing mental fatigue.",
    },
    5: {
        "name": "BREAK_TIMER_REACHED",
        "action": "Trigger Curtain CLOSE for 5-min break; Study timer elapsed",
        "advice": "50-minute study interval reached! Stand up, stretch, and rest your eyes.",
    },
    6: {
        "name": "AWAY",
        "action": "Turn off LED Bar (0%); Park curtain OPEN; Sleep mode",
        "advice": "Workstation vacant. Power saving active.",
    },
}


def generate_synthetic_workstation_data(samples_per_class: int = 1500, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes a realistic dataset of 6-D workstation sensor vectors
    incorporating realistic human posture variations, environmental noise,
    and sensor measurement jitter.
    """
    rng = np.random.default_rng(random_state)
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    # -------------------------------------------------------------------------
    # Class 0: OPTIMAL_STUDY
    # Ergonomic distance (35-55cm), proper lighting (350-700 Lux), 22-26°C, <48dB, <45min
    # -------------------------------------------------------------------------
    presence_0 = np.ones(samples_per_class)
    dist_0 = rng.normal(loc=45.0, scale=4.5, size=samples_per_class).clip(34.0, 58.0)
    lux_0 = rng.normal(loc=460.0, scale=60.0, size=samples_per_class).clip(320.0, 750.0)
    temp_0 = rng.normal(loc=24.5, scale=1.2, size=samples_per_class).clip(21.5, 27.5)
    sound_0 = rng.normal(loc=36.0, scale=4.0, size=samples_per_class).clip(25.0, 48.0)
    dur_0 = rng.uniform(low=0.5, high=45.0, size=samples_per_class)

    x_0 = np.column_stack([presence_0, dist_0, lux_0, temp_0, sound_0, dur_0])
    y_0 = np.zeros(samples_per_class, dtype=int)
    x_list.append(x_0)
    y_list.append(y_0)

    # -------------------------------------------------------------------------
    # Class 1: TEXT_NECK_TOO_CLOSE
    # Distance < 28cm (Head hunched forward toward book)
    # -------------------------------------------------------------------------
    presence_1 = np.ones(samples_per_class)
    dist_1 = rng.uniform(low=10.0, high=26.5, size=samples_per_class)
    lux_1 = rng.normal(loc=430.0, scale=70.0, size=samples_per_class).clip(250.0, 800.0)
    temp_1 = rng.normal(loc=24.5, scale=1.5, size=samples_per_class).clip(21.0, 28.0)
    sound_1 = rng.normal(loc=37.0, scale=5.0, size=samples_per_class).clip(28.0, 52.0)
    dur_1 = rng.uniform(low=1.0, high=48.0, size=samples_per_class)

    x_1 = np.column_stack([presence_1, dist_1, lux_1, temp_1, sound_1, dur_1])
    y_1 = np.ones(samples_per_class, dtype=int)
    x_list.append(x_1)
    y_list.append(y_1)

    # -------------------------------------------------------------------------
    # Class 2: SLOUCH_TOO_FAR
    # Distance > 65cm (Slouching excessively backward)
    # -------------------------------------------------------------------------
    presence_2 = np.ones(samples_per_class)
    dist_2 = rng.uniform(low=66.5, high=110.0, size=samples_per_class)
    lux_2 = rng.normal(loc=440.0, scale=80.0, size=samples_per_class).clip(280.0, 850.0)
    temp_2 = rng.normal(loc=24.8, scale=1.4, size=samples_per_class).clip(21.0, 28.5)
    sound_2 = rng.normal(loc=38.0, scale=5.0, size=samples_per_class).clip(28.0, 54.0)
    dur_2 = rng.uniform(low=1.0, high=48.0, size=samples_per_class)

    x_2 = np.column_stack([presence_2, dist_2, lux_2, temp_2, sound_2, dur_2])
    y_2 = np.full(samples_per_class, 2, dtype=int)
    x_list.append(x_2)
    y_list.append(y_2)

    # -------------------------------------------------------------------------
    # Class 3: POOR_LIGHTING
    # Lux < 200 Lux (Dim/dark) or > 1200 Lux (Harsh direct glare)
    # -------------------------------------------------------------------------
    presence_3 = np.ones(samples_per_class)
    dist_3 = rng.normal(loc=46.0, scale=5.0, size=samples_per_class).clip(32.0, 60.0)
    # Half dim, half glare
    half = samples_per_class // 2
    lux_dim = rng.uniform(low=20.0, high=185.0, size=half)
    lux_glare = rng.uniform(low=1250.0, high=2500.0, size=samples_per_class - half)
    lux_3 = np.concatenate([lux_dim, lux_glare])
    rng.shuffle(lux_3)
    temp_3 = rng.normal(loc=24.5, scale=1.3, size=samples_per_class).clip(21.5, 28.0)
    sound_3 = rng.normal(loc=36.0, scale=4.0, size=samples_per_class).clip(26.0, 50.0)
    dur_3 = rng.uniform(low=0.5, high=45.0, size=samples_per_class)

    x_3 = np.column_stack([presence_3, dist_3, lux_3, temp_3, sound_3, dur_3])
    y_3 = np.full(samples_per_class, 3, dtype=int)
    x_list.append(x_3)
    y_list.append(y_3)

    # -------------------------------------------------------------------------
    # Class 4: NOISY_OR_HOT_FATIGUE
    # Sound > 60 dB or Temp > 30°C
    # -------------------------------------------------------------------------
    presence_4 = np.ones(samples_per_class)
    dist_4 = rng.normal(loc=46.0, scale=5.0, size=samples_per_class).clip(32.0, 60.0)
    lux_4 = rng.normal(loc=450.0, scale=70.0, size=samples_per_class).clip(300.0, 750.0)

    # Split into high sound, high temp, or both
    third = samples_per_class // 3
    # Subgroup A: High sound (>62 dB)
    sound_4a = rng.uniform(low=62.0, high=85.0, size=third)
    temp_4a = rng.normal(loc=25.0, scale=1.5, size=third).clip(22.0, 28.5)

    # Subgroup B: High temperature (>30.5°C)
    sound_4b = rng.normal(loc=38.0, scale=5.0, size=third).clip(28.0, 52.0)
    temp_4b = rng.uniform(low=30.5, high=38.0, size=third)

    # Subgroup C: Both noisy & hot
    rest = samples_per_class - (2 * third)
    sound_4c = rng.uniform(low=62.0, high=85.0, size=rest)
    temp_4c = rng.uniform(low=30.5, high=38.0, size=rest)

    sound_4 = np.concatenate([sound_4a, sound_4b, sound_4c])
    temp_4 = np.concatenate([temp_4a, temp_4b, temp_4c])
    dur_4 = rng.uniform(low=1.0, high=45.0, size=samples_per_class)

    x_4 = np.column_stack([presence_4, dist_4, lux_4, temp_4, sound_4, dur_4])
    y_4 = np.full(samples_per_class, 4, dtype=int)
    x_list.append(x_4)
    y_list.append(y_4)

    # -------------------------------------------------------------------------
    # Class 5: BREAK_TIMER_REACHED
    # Duration >= 50 minutes (Continuous prolonged sitting study)
    # -------------------------------------------------------------------------
    presence_5 = np.ones(samples_per_class)
    dist_5 = rng.normal(loc=45.0, scale=5.0, size=samples_per_class).clip(30.0, 62.0)
    lux_5 = rng.normal(loc=450.0, scale=60.0, size=samples_per_class).clip(300.0, 750.0)
    temp_5 = rng.normal(loc=24.5, scale=1.5, size=samples_per_class).clip(21.0, 28.5)
    sound_5 = rng.normal(loc=36.0, scale=4.0, size=samples_per_class).clip(26.0, 50.0)
    dur_5 = rng.uniform(low=50.0, high=120.0, size=samples_per_class)

    x_5 = np.column_stack([presence_5, dist_5, lux_5, temp_5, sound_5, dur_5])
    y_5 = np.full(samples_per_class, 5, dtype=int)
    x_list.append(x_5)
    y_list.append(y_5)

    # -------------------------------------------------------------------------
    # Class 6: AWAY
    # Presence == 0 (User left desk)
    # -------------------------------------------------------------------------
    presence_6 = np.zeros(samples_per_class)
    dist_6 = rng.uniform(low=70.0, high=250.0, size=samples_per_class)  # Background reflections
    lux_6 = rng.uniform(low=50.0, high=800.0, size=samples_per_class)
    temp_6 = rng.normal(loc=25.0, scale=2.0, size=samples_per_class).clip(20.0, 32.0)
    sound_6 = rng.normal(loc=34.0, scale=6.0, size=samples_per_class).clip(22.0, 65.0)
    dur_6 = np.zeros(samples_per_class)

    x_6 = np.column_stack([presence_6, dist_6, lux_6, temp_6, sound_6, dur_6])
    y_6 = np.full(samples_per_class, 6, dtype=int)
    x_list.append(x_6)
    y_list.append(y_6)

    # Combine and shuffle
    x = np.vstack(x_list)
    y = np.concatenate(y_list)

    shuffle_idx = rng.permutation(len(y))
    return x[shuffle_idx], y[shuffle_idx]


def train_and_export_model(
    model_output_path: str = "model.joblib",
    samples_per_class: int = 1500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Generates training data, fits a tuned Random Forest Classifier,
    evaluates classification accuracy, benchmarks latency (<10ms requirement),
    and exports the serialized bundle to disk.
    """
    logger.info("Generating synthetic sensor data (%d samples/class across 7 classes)...", samples_per_class)
    x, y = generate_synthetic_workstation_data(samples_per_class=samples_per_class, random_state=random_state)
    logger.info("Dataset shape: X=%s, y=%s", x.shape, y.shape)

    # Split train/test (80% / 20%)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=random_state, stratify=y
    )

    logger.info("Training RandomForestClassifier with 20 trees (n_jobs=1 for <10ms edge inference)...")
    clf = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    t0 = time.time()
    clf.fit(x_train, y_train)
    train_duration = time.time() - t0
    logger.info("Model training completed in %.2f seconds.", train_duration)

    # Evaluate accuracy
    y_pred = clf.predict(x_test)
    test_accuracy = float(np.mean(y_pred == y_test))
    logger.info(">>> Test Set Accuracy: %.2f%% <<<", test_accuracy * 100.0)

    # Detailed report
    report = classification_report(
        y_test,
        y_pred,
        target_names=[CLASS_DEFINITIONS[i]["name"] for i in range(7)],
        output_dict=True,
    )
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:\n%s", conf_matrix)

    # Latency benchmark (Raspberry Pi Requirement: < 10 ms per inference)
    logger.info("Benchmarking on-device single-sample inference latency...")
    latencies_ms = []
    sample = x_test[0:1]
    # Warm up JIT / cache
    for _ in range(50):
        clf.predict(sample)

    for i in range(1000):
        test_sample = x_test[i % len(x_test) : (i % len(x_test)) + 1]
        t_start = time.perf_counter()
        _ = clf.predict(test_sample)
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000.0)

    mean_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))
    logger.info("Inference Latency: Mean=%.3f ms | P95=%.3f ms | P99=%.3f ms (Target: < 10 ms)", mean_latency, p95_latency, p99_latency)

    assert mean_latency < 10.0, f"Latency benchmark failed: {mean_latency:.2f}ms exceeds 10ms threshold"

    # Bundle for persistence
    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "class_definitions": CLASS_DEFINITIONS,
        "trained_at": datetime.now().isoformat(),
        "metrics": {
            "test_accuracy": test_accuracy,
            "mean_latency_ms": mean_latency,
            "p99_latency_ms": p99_latency,
            "num_samples": len(y),
        },
    }

    # Ensure output directory exists
    output_file = Path(model_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(output_file))
    logger.info("Model bundle successfully serialized to '%s' (Size: %.2f KB)", output_file, os.path.getsize(output_file) / 1024.0)

    return bundle


class SmartWorkstationAI:
    """
    On-device AI inference engine loaded on the Raspberry Pi Gateway.
    Encapsulates feature validation, model inference, and classification metadata.
    """

    def __init__(self, model_path: str = "model.joblib") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # If model does not exist yet, train it on the fly
            logger.warning("Model file '%s' not found. Automatically triggering training pipeline...", model_path)
            train_and_export_model(model_output_path=str(self.model_path))

        logger.info("Loading AI model from '%s'...", self.model_path)
        bundle = joblib.load(str(self.model_path))
        self.model: RandomForestClassifier = bundle["model"]
        self.feature_names: List[str] = bundle.get("feature_names", FEATURE_NAMES)
        self.class_definitions: Dict[int, Dict[str, str]] = bundle.get("class_definitions", CLASS_DEFINITIONS)
        self.metrics = bundle.get("metrics", {})
        logger.info("AI Model loaded successfully! (Trained Accuracy: %.2f%%)", self.metrics.get("test_accuracy", 1.0) * 100.0)

    def predict(
        self,
        presence: int,
        distance_cm: float,
        lux: float,
        temp_c: float,
        sound_db: float,
        sitting_duration_min: float,
    ) -> Dict[str, Any]:
        """
        Executes real-time inference on the 6-feature vector.
        Returns:
            dict containing class_id, class_name, confidence, action, advice, and inference_time_ms.
        """
        feature_vec = np.array(
            [[presence, distance_cm, lux, temp_c, sound_db, sitting_duration_min]],
            dtype=float,
        )

        t0 = time.perf_counter()
        pred_class = int(self.model.predict(feature_vec)[0])
        # Calculate prediction probability / confidence
        proba = self.model.predict_proba(feature_vec)[0]
        confidence = float(proba[pred_class])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        meta = self.class_definitions.get(
            pred_class,
            {"name": "UNKNOWN", "action": "None", "advice": "No recommendation"},
        )

        return {
            "class_id": pred_class,
            "class_name": meta["name"],
            "confidence": round(confidence, 3),
            "action": meta["action"],
            "advice": meta["advice"],
            "latency_ms": round(latency_ms, 3),
        }


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    target_model_file = current_dir / "model.joblib"
    print("=" * 70)
    print("Smart Workstation - Training On-Device AI Model (Random Forest)")
    print("=" * 70)
    train_and_export_model(model_output_path=str(target_model_file))
    print("\nVerifying model loading and test prediction:")
    engine = SmartWorkstationAI(model_path=str(target_model_file))

    # Test cases
    test_cases = [
        {"name": "Ergonomic study", "features": (1, 45.0, 450.0, 24.5, 35.0, 20.0)},
        {"name": "Text neck (slouching forward)", "features": (1, 20.0, 450.0, 24.5, 35.0, 20.0)},
        {"name": "Slouching too far back", "features": (1, 80.0, 450.0, 24.5, 35.0, 20.0)},
        {"name": "Dark room", "features": (1, 45.0, 80.0, 24.5, 35.0, 20.0)},
        {"name": "Noisy/Hot room", "features": (1, 45.0, 450.0, 32.0, 68.0, 20.0)},
        {"name": "Break timer elapsed (55 min)", "features": (1, 45.0, 450.0, 24.5, 35.0, 55.0)},
        {"name": "User away", "features": (0, 150.0, 300.0, 25.0, 35.0, 0.0)},
    ]

    print("\nInference Verification Results:")
    print("-" * 70)
    for tc in test_cases:
        res = engine.predict(*tc["features"])
        print(f"[{tc['name']:<28}] -> Class {res['class_id']}: {res['class_name']:<22} (Conf: {res['confidence']*100:.1f}%, {res['latency_ms']:.2f}ms)")
    print("=" * 70)

