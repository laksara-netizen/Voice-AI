"""CPU speaker verification using SpeechBrain ECAPA-TDNN."""
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from threading import Lock

import numpy as np


class EcapaVoice:
    def __init__(self):
        self.root = Path("voiceprints/ecapa-v1")
        self.threshold = float(os.environ.get("VOICE_ECAPA_THRESHOLD", "0.75"))
        if not -1 <= self.threshold <= 1:
            raise ValueError("VOICE_ECAPA_THRESHOLD must be between -1 and 1")
        self._model = None
        self._lock = Lock()

    def extract_features(self, path):
        import torch

        decoded = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
             "-map", "0:a:0", "-ac", "1", "-ar", "16000",
             "-t", "31", "-f", "f32le", "pipe:1"],
            capture_output=True, check=True, timeout=20,
        )
        waveform = np.frombuffer(decoded.stdout, dtype="<f4").copy()
        if not 16000 <= waveform.size <= 480000:
            raise ValueError("Record between 1 and 30 seconds of audio")
        if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) < 0.001:
            raise ValueError("Recording contains no usable audio")

        with self._lock, torch.inference_mode():
            if self._model is None:
                from speechbrain.inference.classifiers import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy

                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                    local_strategy=LocalStrategy.COPY,
                )
            embedding = self._model.encode_batch(torch.from_numpy(waveform)[None, :])
        return self._unit_vector(embedding.cpu().numpy().reshape(-1))

    @staticmethod
    def _unit_vector(vector):
        norm = np.linalg.norm(vector)
        if vector.ndim != 1 or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid speaker embedding")
        return vector / norm

    def _sample_paths(self, user_id):
        directory = self.root / str(user_id)
        return [directory / f"sample_{number}.npy" for number in (1, 2, 3)]

    @staticmethod
    def _write_vector(path, vector):
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                np.save(temp, vector, allow_pickle=False)
                temp.close()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)

    def template_path(self, user_id):
        return self.root / str(user_id) / "centroid.npy"

    def save_voice_sample(self, user_id, sample_number, embedding):
        if sample_number not in (1, 2, 3):
            raise ValueError("Invalid enrollment sample number")
        self._write_vector(self._sample_paths(user_id)[sample_number - 1], embedding)

    def get_sample_count(self, user_id):
        return sum(path.is_file() for path in self._sample_paths(user_id))

    def create_voiceprint(self, user_id):
        samples = [np.load(path, allow_pickle=False) for path in self._sample_paths(user_id)]
        centroid = self._unit_vector(np.mean(samples, axis=0))
        self._write_vector(self.template_path(user_id), centroid)

    def authenticate_voice(self, user_id, embedding):
        centroid = np.load(self.template_path(user_id), allow_pickle=False)
        score = float(np.clip(np.dot(self._unit_vector(centroid), embedding), -1, 1))
        return score >= self.threshold, score

    def clear_user_voiceprint(self, user_id):
        for path in [self.template_path(user_id), *self._sample_paths(user_id)]:
            path.unlink(missing_ok=True)
