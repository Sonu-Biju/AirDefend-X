"""
logic.py — AirDefend-X Core Security Engine
============================================
Layer 1 — Identity (RSA-2048)
Layer 2 — Kinematic Hard Filter (pre-filter for truly impossible absolutes)
Layer 3 — Behavioral AI (Isolation Forest) — primary physics verdict
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Layer 1 — RSA-2048
# ---------------------------------------------------------------------------

def generate_keypair():
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    return private_key, private_key.public_key()


def build_payload(icao24, velocity, vertrate, geoaltitude, lat, lon) -> bytes:
    fields = {
        "icao24":      str(icao24),
        "velocity":    round(float(velocity),    4),
        "vertrate":    round(float(vertrate),    4),
        "geoaltitude": round(float(geoaltitude), 4),
        "lat":         round(float(lat),         6),
        "lon":         round(float(lon),         6),
    }
    return json.dumps(fields, sort_keys=True).encode("utf-8")


def sign_message(private_key, icao24, velocity, vertrate, geoaltitude, lat, lon) -> bytes:
    payload = build_payload(icao24, velocity, vertrate, geoaltitude, lat, lon)
    return private_key.sign(payload, padding.PKCS1v15(), hashes.SHA256())


def verify_signature(public_key, icao24, velocity, vertrate,
                     geoaltitude, lat, lon, signature: bytes) -> bool:
    if not isinstance(signature, (bytes, bytearray)) or len(signature) == 0:
        return False
    payload = build_payload(icao24, velocity, vertrate, geoaltitude, lat, lon)
    try:
        public_key.verify(signature, payload, padding.PKCS1v15(), hashes.SHA256())
        return True
    except (InvalidSignature, Exception):
        return False


class AircraftKeyRegistry:
    def __init__(self):
        self._registry     = {}
        self._private_keys = {}

    def register(self, icao24: str):
        priv, pub = generate_keypair()
        self._registry[icao24]     = pub
        self._private_keys[icao24] = priv

    def get_public_key(self, icao24):
        return self._registry.get(icao24)

    def get_private_key(self, icao24):
        return self._private_keys.get(icao24)

    def is_registered(self, icao24: str) -> bool:
        return icao24 in self._registry

    def sign_for(self, row) -> bytes:
        priv = self.get_private_key(row['icao24'])
        if priv is None:
            return b""
        return sign_message(priv, row['icao24'], row['velocity'],
                             row['vertrate'], row['geoaltitude'],
                             row['lat'], row['lon'])


# ---------------------------------------------------------------------------
# Layer 2 — Kinematic Hard Filter
# ---------------------------------------------------------------------------

HARD_MAX_VELOCITY_MS = 500.0
HARD_MAX_VERTRATE_MS = 50.0
HARD_MIN_ALT_M       = -500.0
HARD_MAX_ALT_M       = 20000.0
MAX_VELOCITY_MS      = HARD_MAX_VELOCITY_MS


class KinematicFilter:
    def __init__(self):
        self._state = defaultdict(lambda: None)

    def check(self, row) -> tuple[bool, str]:
        vel = float(row['velocity'])
        vr  = float(row['vertrate'])
        alt = float(row['geoaltitude'])
        if vel < 0 or vel > HARD_MAX_VELOCITY_MS:
            return False, f"velocity {vel:.1f} m/s exceeds hard limit"
        if abs(vr) > HARD_MAX_VERTRATE_MS:
            return False, f"vertical rate {vr:.1f} m/s exceeds hard limit"
        if alt < HARD_MIN_ALT_M or alt > HARD_MAX_ALT_M:
            return False, f"altitude {alt:.0f} m outside valid range"
        self._state[row['icao24']] = {
            'velocity': vel, 'alt': alt,
            'lat': float(row['lat']), 'lon': float(row['lon']),
            'time': float(row.get('time', 0))
        }
        return True, "ok"


# ---------------------------------------------------------------------------
# Layer 3 — Behavioral AI (Isolation Forest)
# ---------------------------------------------------------------------------

BEHAVIORAL_FEATURES = [
    'velocity', 'vertrate', 'geoaltitude',
    'vel_x', 'vel_z_ratio', 'alt_norm', 'energy_proxy'
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    track_rad = np.radians(out['track'] if 'track' in out.columns
                           else pd.Series(0.0, index=out.index))
    out['vel_x']        = out['velocity'] * np.cos(track_rad)
    safe_vel            = out['velocity'].replace(0, np.nan)
    out['vel_z_ratio']  = (out['vertrate'] / safe_vel).fillna(0).clip(-1, 1)
    out['alt_norm']     = out['geoaltitude'].clip(0, 12000) / 12000.0
    out['energy_proxy'] = (0.5 * out['velocity'] ** 2 +
                           9.81 * out['geoaltitude'].clip(0, None)) / 1e6
    return out


class BehavioralDetector:
    def __init__(self, contamination: float = 0.005):
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            contamination=contamination,
            n_estimators=300,
            max_samples='auto',
            random_state=42
        )
        self._trained = False

    def train(self, legitimate_df: pd.DataFrame):
        feats = compute_features(legitimate_df)
        X     = feats[BEHAVIORAL_FEATURES].fillna(0).values
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X))
        self._trained = True

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self._trained:
            n = len(df)
            return np.ones(n, dtype=int), np.zeros(n)
        feats    = compute_features(df)
        X        = feats[BEHAVIORAL_FEATURES].fillna(0).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled), self.model.decision_function(X_scaled)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_registry   = AircraftKeyRegistry()
_kinematic  = KinematicFilter()
_behavioral = BehavioralDetector(contamination=0.005)
_trained    = False

# Threshold is set dynamically at training time from the actual score distribution
AI_CONFIDENCE_THRESHOLD = -0.01   # updated by train_behavioral_model()


def train_behavioral_model(legitimate_df: pd.DataFrame):
    """
    Call this ONCE at app bootstrap with clean legitimate data only.
    Never call inside run_middleware — that causes the attack row to
    pollute the training set.
    Dynamically sets AI_CONFIDENCE_THRESHOLD from the actual score
    distribution so it adapts to any training dataset.
    """
    global _trained, AI_CONFIDENCE_THRESHOLD
    if not _trained and len(legitimate_df) >= 50:
        _behavioral.train(legitimate_df)
        _trained = True
        # Set threshold just below the lowest legitimate score.
        # This means ANY frame more anomalous than the worst legitimate
        # flight in training will be flagged — no hardcoded magic number.
        from sklearn.ensemble import IsolationForest
        feats   = compute_features(prepare_dataframe(legitimate_df))
        X       = feats[BEHAVIORAL_FEATURES].fillna(0).values
        X_scaled = _behavioral.scaler.transform(X)
        legit_scores = _behavioral.model.decision_function(X_scaled)
        # Sit just below the minimum legitimate score with a small margin
        AI_CONFIDENCE_THRESHOLD = float(legit_scores.min()) - 0.002
        print(f"[AI] Threshold set to {AI_CONFIDENCE_THRESHOLD:.6f} (legit min was {legit_scores.min():.6f})")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ['icao24', 'velocity', 'vertrate', 'geoaltitude', 'lat', 'lon']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    out = df.copy()
    for col in ['velocity', 'vertrate', 'geoaltitude', 'lat', 'lon']:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    if 'time' not in out.columns:
        out['time'] = out.index.astype(float)
    return out


def register_and_sign_legitimate(df: pd.DataFrame) -> pd.DataFrame:
    """Register every aircraft and sign all rows. Call once at bootstrap."""
    df = df.copy()
    for idx, row in df.iterrows():
        icao = row['icao24']
        if not _registry.is_registered(icao):
            _registry.register(icao)
        if 'signature' not in df.columns or not isinstance(df.at[idx, 'signature'], bytes):
            df.at[idx, 'signature'] = _registry.sign_for(row)
    return df


def make_physics_attack_row(base_row: pd.Series, attack_icao: str) -> pd.Series:
    """
    Build a physics-attack frame with its own unique registered ICAO.
    Signed correctly so RSA passes — only AI catches it.
    """
    # Register the attack ICAO if not already done
    if not _registry.is_registered(attack_icao):
        _registry.register(attack_icao)

    atk = base_row.copy()
    atk['icao24']      = attack_icao
    # Anomalous flight envelope: stall-speed slow at high cruise altitude
    # with an extreme climb rate — impossible combination in practice
    # Deliberately extreme combination: near-stall speed at high cruise altitude
    # with a violent climb rate. Each value alone could be borderline plausible,
    # but together they form an impossible flight-envelope profile that sits
    # far outside the learned legitimate distribution.
    atk['velocity']    = 60.0     # dangerously slow at cruise altitude
    atk['geoaltitude'] = 12000.0  # near ceiling of civil aviation envelope
    atk['vertrate']    = 45.0     # extreme climb rate

    # Sign with the attack ICAO's key — RSA will pass, AI must catch it
    priv = _registry.get_private_key(attack_icao)
    atk['signature'] = sign_message(
        priv, attack_icao,
        atk['velocity'], atk['vertrate'],
        atk['geoaltitude'], atk['lat'], atk['lon']
    )
    return atk


def run_middleware(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline. run_middleware does NOT train the model.
    Train must be called separately at bootstrap via train_behavioral_model().
    """
    df = prepare_dataframe(df)

    result = df.copy()
    result['auth_passed']   = False
    result['kinematic_ok']  = True
    result['iso_score']     = 1
    result['ai_confidence'] = 0.0

    # ── Layer 1: RSA ─────────────────────────────────────────────────────────
    for idx, row in result.iterrows():
        icao = row['icao24']
        sig  = row.get('signature', None)
        if not _registry.is_registered(icao):
            result.at[idx, 'auth_passed'] = False
            continue
        pub = _registry.get_public_key(icao)
        if isinstance(sig, bytes) and len(sig) > 0:
            result.at[idx, 'auth_passed'] = verify_signature(
                pub, icao, row['velocity'], row['vertrate'],
                row['geoaltitude'], row['lat'], row['lon'], sig
            )
        else:
            result.at[idx, 'auth_passed'] = False

    # ── Layer 2: Kinematic hard pre-filter ───────────────────────────────────
    for idx, row in result.iterrows():
        ok, _ = _kinematic.check(row)
        result.at[idx, 'kinematic_ok'] = ok

    # ── Layer 3: AI ──────────────────────────────────────────────────────────
    if _trained:
        labels, scores = _behavioral.score(result)
        result['iso_score']     = labels
        result['ai_confidence'] = scores

    # ── Final Classification ─────────────────────────────────────────────────
    result['status'] = 'LEGITIMATE'

    result.loc[~result['auth_passed'], 'status'] = 'FAKE IDENTITY'

    physics_fail = (
        (~result['kinematic_ok']) |
        (
            (result['iso_score']     == -1) &
            (result['ai_confidence'] <  AI_CONFIDENCE_THRESHOLD)
        )
    )
    result.loc[(result['status'] == 'LEGITIMATE') & physics_fail,
               'status'] = 'IMPROPER PHYSICS'

    # ── Colours ──────────────────────────────────────────────────────────────
    def assign_color(status):
        if status == 'LEGITIMATE':       return 0,   255, 0
        if status == 'IMPROPER PHYSICS': return 255, 0,   0
        return 255, 165, 0

    result[['r', 'g', 'b']] = pd.DataFrame(
        result['status'].apply(assign_color).tolist(),
        index=result.index
    )

    return result
