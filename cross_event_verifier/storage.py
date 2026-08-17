"""身份、证据、原型和审计历史的 SQLite 适配器。

该存储器在边界处有意保持简单：NumPy 向量打包为归一化 float32 blob，结构化值
使用 JSON，每次公开写入都受连接锁保护。因此验证器既可以在测试中使用
``:memory:``，也可以在 GUI 中使用持久化文件。
"""

from __future__ import annotations

from contextlib import contextmanager
import base64
from dataclasses import asdict
import json
import sqlite3
import threading
import time
from typing import Any, Iterator

import numpy as np

from .types import (
    AppearanceAbsorptionRequest,
    CandidateRecord,
    FeatureBundle,
    GaitEnrollmentEvent,
    Observation,
    Prototype,
    TrackQuality,
    VerificationState,
    normalize_vector,
)


def _pack_vector(value: np.ndarray | None) -> tuple[bytes | None, int | None]:
    """归一化向量，并返回紧凑的 SQLite blob 及其维度。"""
    vector = normalize_vector(value)
    if vector is None:
        return None, None
    return vector.astype(np.float32).tobytes(), int(vector.size)


def _unpack_vector(blob: bytes | None, dimension: int | None) -> np.ndarray | None:
    """解码一个已存储的 float32 blob，并恢复其归一化表示。"""
    if blob is None or dimension is None or int(dimension) <= 0:
        return None
    if len(blob) != int(dimension) * np.dtype(np.float32).itemsize:
        return None
    vector = np.frombuffer(blob, dtype=np.float32, count=int(dimension)).copy()
    return normalize_vector(vector)


def _json_default(value: Any) -> Any:
    """序列化元数据和审计载荷中使用的 NumPy 标量/数组。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"cannot encode {type(value)!r}")


class SqliteStore:
    """带有小型事务方法的线程安全 SQLite 适配器。

    Schema 创建具有幂等性，写入使用 SQLite 上下文管理器提交；``transaction``
    为需要同时更新多张表的调用方提供明确的全有或全无接口。
    """

    def __init__(self, path: str = ":memory:") -> None:
        """打开连接，并创建缺失的表/索引。"""
        self.path = path
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        if path != ":memory:":
            # WAL lets the GUI/readers continue while the worker commits a
            # frame transaction.  In-memory databases do not support WAL.
            self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        """如有需要，创建持久化身份/证据/审计 schema。"""
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    identity_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    proposed_identity TEXT,
                    confirmed_identity TEXT,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    event_json TEXT NOT NULL DEFAULT '[]',
                    independent_event_count INTEGER NOT NULL DEFAULT 0,
                    high_quality_evidence_count INTEGER NOT NULL DEFAULT 0,
                    conflict_count INTEGER NOT NULL DEFAULT 0,
                    challenge_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prototypes (
                    prototype_id TEXT PRIMARY KEY,
                    identity_id TEXT NOT NULL,
                    zone TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    quality REAL NOT NULL,
                    camera_id TEXT,
                    view_angle TEXT,
                    clothing_tag TEXT,
                    source_event_id TEXT,
                    created_at REAL NOT NULL,
                    model_version TEXT NOT NULL DEFAULT 'unconfigured',
                    feature_schema TEXT NOT NULL DEFAULT 'unconfigured-v1',
                    artifact_sha256 TEXT NOT NULL DEFAULT 'unverified',
                    preprocess_version TEXT NOT NULL DEFAULT 'unversioned-v1',
                    joint_format TEXT NOT NULL DEFAULT 'unknown',
                    sequence_length INTEGER,
                    tta_mode TEXT NOT NULL DEFAULT 'unknown',
                    coordinate_contract TEXT NOT NULL DEFAULT 'unknown',
                    embedding_dimensions_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_prototypes_identity_zone
                    ON prototypes(identity_id, zone, modality);
                CREATE TABLE IF NOT EXISTS observations (
                    event_id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    camera_id TEXT NOT NULL,
                    capture_session_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    end_timestamp REAL,
                    appearance BLOB,
                    appearance_dimension INTEGER,
                    gait BLOB,
                    gait_dimension INTEGER,
                    quality_json TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    feature_schema TEXT NOT NULL DEFAULT 'unconfigured-v1',
                    calibration_version TEXT NOT NULL DEFAULT 'heuristic-default-v1',
                    threshold_version TEXT NOT NULL,
                    source_event_json TEXT NOT NULL DEFAULT '[]',
                    challenge_id TEXT,
                    challenge_response_json TEXT NOT NULL DEFAULT '{}',
                    appearance_request_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    artifact_sha256 TEXT NOT NULL DEFAULT 'unverified',
                    preprocess_version TEXT NOT NULL DEFAULT 'unversioned-v1',
                    joint_format TEXT NOT NULL DEFAULT 'unknown',
                    sequence_length INTEGER,
                    tta_mode TEXT NOT NULL DEFAULT 'unknown',
                    coordinate_contract TEXT NOT NULL DEFAULT 'unknown',
                    embedding_dimensions_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_observations_candidate_timestamp
                    ON observations(candidate_id, timestamp);
                CREATE TABLE IF NOT EXISTS appearance_requests (
                    request_id TEXT PRIMARY KEY,
                    identity_id TEXT NOT NULL,
                    issued_by_event_id TEXT NOT NULL,
                    candidate_id TEXT,
                    gait_probability REAL NOT NULL,
                    gait_quality REAL NOT NULL,
                    issued_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    response_event_id TEXT,
                    camera_id TEXT,
                    track_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    event_id TEXT NOT NULL,
                    identity_id TEXT,
                    kind TEXT NOT NULL,
                    score REAL,
                    margin REAL,
                    independent INTEGER NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_candidate
                    ON evidence(candidate_id, created_at);
                CREATE TABLE IF NOT EXISTS automation_gait_events (
                    candidate_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    stability REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    feature_schema TEXT NOT NULL,
                    calibration_version TEXT NOT NULL DEFAULT 'heuristic-default-v1',
                    artifact_sha256 TEXT NOT NULL DEFAULT 'unverified',
                    preprocess_version TEXT NOT NULL DEFAULT 'unversioned-v1',
                    joint_format TEXT NOT NULL DEFAULT 'unknown',
                    sequence_length INTEGER,
                    tta_mode TEXT NOT NULL DEFAULT 'unknown',
                    coordinate_contract TEXT NOT NULL DEFAULT 'unknown',
                    embedding_dimensions_json TEXT NOT NULL DEFAULT '{}',
                    camera_id TEXT NOT NULL DEFAULT 'unknown-camera',
                    capture_session_id TEXT NOT NULL DEFAULT 'unknown-session',
                    start_timestamp REAL,
                    end_timestamp REAL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(candidate_id, event_key)
                );
                CREATE INDEX IF NOT EXISTS idx_automation_gait_events_candidate
                    ON automation_gait_events(candidate_id, created_at);
                CREATE TABLE IF NOT EXISTS gait_enrollment_events (
                    identity_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    capture_session_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    stability REAL NOT NULL,
                    quality REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    view_angle TEXT,
                    created_at REAL NOT NULL,
                    model_version TEXT NOT NULL DEFAULT 'unconfigured',
                    feature_schema TEXT NOT NULL DEFAULT 'unconfigured-v1',
                    calibration_version TEXT NOT NULL DEFAULT 'heuristic-default-v1',
                    artifact_sha256 TEXT NOT NULL DEFAULT 'unverified',
                    preprocess_version TEXT NOT NULL DEFAULT 'unversioned-v1',
                    joint_format TEXT NOT NULL DEFAULT 'unknown',
                    sequence_length INTEGER,
                    tta_mode TEXT NOT NULL DEFAULT 'unknown',
                    coordinate_contract TEXT NOT NULL DEFAULT 'unknown',
                    embedding_dimensions_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(identity_id, event_key)
                );
                CREATE INDEX IF NOT EXISTS idx_gait_enrollment_events_identity
                    ON gait_enrollment_events(identity_id, created_at);
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    entity_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                """
            )
            self._ensure_column(
                "observations", "appearance_request_id", "TEXT"
            )
            self._ensure_column(
                "prototypes", "model_version", "TEXT NOT NULL DEFAULT 'unconfigured'"
            )
            self._ensure_column(
                "prototypes", "feature_schema", "TEXT NOT NULL DEFAULT 'unconfigured-v1'"
            )
            for column, definition in (
                ("artifact_sha256", "TEXT NOT NULL DEFAULT 'unverified'"),
                ("preprocess_version", "TEXT NOT NULL DEFAULT 'unversioned-v1'"),
                ("joint_format", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("sequence_length", "INTEGER"),
                ("tta_mode", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("coordinate_contract", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("embedding_dimensions_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                self._ensure_column("prototypes", column, definition)
            self._ensure_column(
                "observations", "feature_schema", "TEXT NOT NULL DEFAULT 'unconfigured-v1'"
            )
            self._ensure_column(
                "observations", "calibration_version", "TEXT NOT NULL DEFAULT 'heuristic-default-v1'"
            )
            for column, definition in (
                ("artifact_sha256", "TEXT NOT NULL DEFAULT 'unverified'"),
                ("preprocess_version", "TEXT NOT NULL DEFAULT 'unversioned-v1'"),
                ("joint_format", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("sequence_length", "INTEGER"),
                ("tta_mode", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("coordinate_contract", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("embedding_dimensions_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                self._ensure_column("observations", column, definition)
            self._ensure_column("appearance_requests", "camera_id", "TEXT")
            self._ensure_column("appearance_requests", "track_id", "TEXT")
            for column, definition in (
                ("calibration_version", "TEXT NOT NULL DEFAULT 'heuristic-default-v1'"),
                ("artifact_sha256", "TEXT NOT NULL DEFAULT 'unverified'"),
                ("preprocess_version", "TEXT NOT NULL DEFAULT 'unversioned-v1'"),
                ("joint_format", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("sequence_length", "INTEGER"),
                ("tta_mode", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("coordinate_contract", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("embedding_dimensions_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                self._ensure_column("automation_gait_events", column, definition)
            for column, definition in (
                ("camera_id", "TEXT NOT NULL DEFAULT 'unknown-camera'"),
                ("capture_session_id", "TEXT NOT NULL DEFAULT 'unknown-session'"),
                ("start_timestamp", "REAL"),
                ("end_timestamp", "REAL"),
            ):
                self._ensure_column("automation_gait_events", column, definition)
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(2, ?)",
                (time.time(),),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(3, ?)",
                (time.time(),),
            )
            self.connection.execute("PRAGMA user_version = 3")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        """为旧版本数据库执行小型增量迁移。"""
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """在出错回滚的事务中提供一个连接。

        写入方法可以嵌套在这个上下文中。只有最外层事务负责提交或回滚，
        避免 ``verify_batch`` 内部的每次 ``save_*`` 调用都触发一次磁盘提交。
        """
        with self._lock:
            outermost = self._transaction_depth == 0
            if outermost:
                self.connection.execute("BEGIN")
            self._transaction_depth += 1
            try:
                yield self.connection
                if outermost:
                    self.connection.commit()
            except Exception:
                if outermost:
                    self.connection.rollback()
                raise
            finally:
                self._transaction_depth -= 1

    @contextmanager
    def _write_scope(self) -> Iterator[sqlite3.Connection]:
        """为单次写入或外层批量事务提供统一的提交边界。"""
        with self._lock:
            if self._transaction_depth > 0:
                yield self.connection
            else:
                with self.connection:
                    yield self.connection

    def close(self) -> None:
        """所有工作线程停止后关闭 SQLite 连接。"""
        with self._lock:
            self.connection.close()

    def backup_to(self, path: str) -> None:
        """将当前数据库一致地复制到另一个 SQLite 文件。"""

        with self._lock, sqlite3.connect(path) as target:
            self.connection.backup(target)

    def clear_all_data(self) -> None:
        """清除身份图库及其运行证据，但保留数据库 schema。"""

        tables = (
            "identities",
            "candidates",
            "prototypes",
            "observations",
            "appearance_requests",
            "evidence",
            "automation_gait_events",
            "gait_enrollment_events",
            "snapshots",
            "audit_log",
        )
        with self._lock, self.connection:
            for table in tables:
                self.connection.execute(f"DELETE FROM {table}")

    def upsert_identity(
        self,
        identity_id: str,
        state: VerificationState = VerificationState.CONFIRMED_IDENTITY,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """创建或更新身份行，但不修改其原型。"""
        now = time.time()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=_json_default)
        with self._write_scope():
            self.connection.execute(
                """
                INSERT INTO identities(identity_id,state,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(identity_id) DO UPDATE SET
                    state=excluded.state,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (identity_id, state.value, metadata_json, now, now),
            )

    def set_identity_state(self, identity_id: str, state: VerificationState) -> None:
        """只更新已有身份的生命周期状态。"""
        with self._write_scope():
            self.connection.execute(
                "UPDATE identities SET state=?, updated_at=? WHERE identity_id=?",
                (state.value, time.time(), identity_id),
            )

    def identity_states(self) -> dict[str, VerificationState]:
        """返回以公开 ID 为键的持久化身份生命周期状态。"""
        with self._lock:
            rows = self.connection.execute("SELECT identity_id,state FROM identities").fetchall()
        return {row["identity_id"]: VerificationState(row["state"]) for row in rows}

    def save_candidate(self, candidate: CandidateRecord) -> None:
        """插入或更新隔离/正式候选及其序列化证据链接。"""
        with self._write_scope():
            self.connection.execute(
                """
                INSERT INTO candidates(
                    candidate_id,state,proposed_identity,confirmed_identity,evidence_json,
                    event_json,independent_event_count,high_quality_evidence_count,
                    conflict_count,challenge_json,metadata_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    state=excluded.state,
                    proposed_identity=excluded.proposed_identity,
                    confirmed_identity=excluded.confirmed_identity,
                    evidence_json=excluded.evidence_json,
                    event_json=excluded.event_json,
                    independent_event_count=excluded.independent_event_count,
                    high_quality_evidence_count=excluded.high_quality_evidence_count,
                    conflict_count=excluded.conflict_count,
                    challenge_json=excluded.challenge_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    candidate.candidate_id,
                    candidate.state.value,
                    candidate.proposed_identity,
                    candidate.confirmed_identity,
                    json.dumps(candidate.evidence_ids, ensure_ascii=False),
                    json.dumps(candidate.event_ids, ensure_ascii=False),
                    candidate.independent_event_count,
                    candidate.high_quality_evidence_count,
                    candidate.conflict_count,
                    json.dumps(candidate.challenge_ids, ensure_ascii=False),
                    json.dumps(candidate.metadata, ensure_ascii=False, default=_json_default),
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> CandidateRecord:
        """从一行 SQLite 记录恢复领域候选。"""
        return CandidateRecord(
            candidate_id=row["candidate_id"],
            state=VerificationState(row["state"]),
            proposed_identity=row["proposed_identity"],
            confirmed_identity=row["confirmed_identity"],
            evidence_ids=list(json.loads(row["evidence_json"])),
            event_ids=list(json.loads(row["event_json"])),
            independent_event_count=int(row["independent_event_count"]),
            high_quality_evidence_count=int(row["high_quality_evidence_count"]),
            conflict_count=int(row["conflict_count"]),
            challenge_ids=list(json.loads(row["challenge_json"])),
            metadata=dict(json.loads(row["metadata_json"])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def get_candidate(self, candidate_id: str) -> CandidateRecord | None:
        """加载一个候选；未知时返回 ``None``。"""
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        return self._candidate_from_row(row) if row else None

    def list_candidates(self) -> list[CandidateRecord]:
        """按更新时间从新到旧列出候选，供诊断和 GUI 查看。"""
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM candidates ORDER BY updated_at DESC"
            ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def save_prototypes(self, prototypes: list[Prototype], *, replace_identity: str | None = None, zone: str | None = None) -> None:
        """持久化归一化原型，可选替换某个身份区域。"""
        with self._write_scope():
            if replace_identity is not None:
                if zone is None:
                    self.connection.execute("DELETE FROM prototypes WHERE identity_id=?", (replace_identity,))
                else:
                    self.connection.execute(
                        "DELETE FROM prototypes WHERE identity_id=? AND zone=?",
                        (replace_identity, zone),
                    )
            for prototype in prototypes:
                blob, dimension = _pack_vector(prototype.vector)
                if blob is None or dimension is None:
                    continue
                expected_dimension = dict(prototype.embedding_dimensions).get(
                    prototype.modality
                )
                if expected_dimension is not None and int(expected_dimension) != int(dimension):
                    raise ValueError(
                        "prototype dimension does not satisfy its embedding contract: "
                        f"{prototype.modality}={dimension}, expected={expected_dimension}"
                    )
                self.connection.execute(
                    """
                    INSERT INTO prototypes(
                        prototype_id,identity_id,zone,modality,vector,dimension,quality,
                        camera_id,view_angle,clothing_tag,source_event_id,created_at,
                        model_version,feature_schema,artifact_sha256,preprocess_version,
                        joint_format,sequence_length,tta_mode,coordinate_contract,
                        embedding_dimensions_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(prototype_id) DO UPDATE SET
                        identity_id=excluded.identity_id,
                        zone=excluded.zone,
                        modality=excluded.modality,
                        vector=excluded.vector,
                        dimension=excluded.dimension,
                        quality=excluded.quality,
                        camera_id=excluded.camera_id,
                        view_angle=excluded.view_angle,
                        clothing_tag=excluded.clothing_tag,
                        source_event_id=excluded.source_event_id,
                        created_at=excluded.created_at,
                        model_version=excluded.model_version,
                        feature_schema=excluded.feature_schema,
                        artifact_sha256=excluded.artifact_sha256,
                        preprocess_version=excluded.preprocess_version,
                        joint_format=excluded.joint_format,
                        sequence_length=excluded.sequence_length,
                        tta_mode=excluded.tta_mode,
                        coordinate_contract=excluded.coordinate_contract,
                        embedding_dimensions_json=excluded.embedding_dimensions_json
                    """,
                    (
                        prototype.prototype_id,
                        prototype.identity_id,
                        prototype.zone,
                        prototype.modality,
                        sqlite3.Binary(blob),
                        dimension,
                        prototype.quality,
                        prototype.camera_id,
                        prototype.view_angle,
                        prototype.clothing_tag,
                        prototype.source_event_id,
                        prototype.created_at,
                        prototype.model_version,
                        prototype.feature_schema,
                        prototype.artifact_sha256,
                        prototype.preprocess_version,
                        prototype.joint_format,
                        prototype.sequence_length,
                        prototype.tta_mode,
                        prototype.coordinate_contract,
                        json.dumps(dict(prototype.embedding_dimensions), ensure_ascii=False),
                    ),
                )

    def load_prototypes(self) -> list[Prototype]:
        """加载所有可解码原型，并跳过损坏或空向量。"""
        with self._lock:
            rows = self.connection.execute("SELECT * FROM prototypes").fetchall()
        result: list[Prototype] = []
        for row in rows:
            vector = _unpack_vector(row["vector"], row["dimension"])
            if vector is None:
                continue
            result.append(
                Prototype(
                    identity_id=row["identity_id"],
                    modality=row["modality"],
                    vector=vector,
                    zone=row["zone"],
                    quality=float(row["quality"]),
                    camera_id=row["camera_id"],
                    view_angle=row["view_angle"],
                    clothing_tag=row["clothing_tag"],
                    source_event_id=row["source_event_id"],
                    prototype_id=row["prototype_id"],
                    created_at=float(row["created_at"]),
                    model_version=row["model_version"],
                    feature_schema=row["feature_schema"],
                    artifact_sha256=row["artifact_sha256"],
                    preprocess_version=row["preprocess_version"],
                    joint_format=row["joint_format"],
                    sequence_length=(
                        int(row["sequence_length"])
                        if row["sequence_length"] is not None
                        else None
                    ),
                    tta_mode=row["tta_mode"],
                    coordinate_contract=row["coordinate_contract"],
                    embedding_dimensions=dict(
                        json.loads(row["embedding_dimensions_json"] or "{}")
                    ),
                )
            )
        return result

    def save_observation(self, observation: Observation, candidate_id: str | None = None) -> None:
        """持久化一个归一化观察值及其质量/审计元数据。"""
        normalized = observation.normalized()
        appearance, appearance_dimension = _pack_vector(normalized.features.appearance)
        gait, gait_dimension = _pack_vector(normalized.features.gait)
        declared_dimensions = dict(normalized.embedding_dimensions)
        for modality, actual_dimension in (
            ("appearance", appearance_dimension),
            ("gait", gait_dimension),
        ):
            expected_dimension = declared_dimensions.get(modality)
            if (
                actual_dimension is not None
                and expected_dimension is not None
                and int(actual_dimension) != int(expected_dimension)
            ):
                raise ValueError(
                    "observation dimension does not satisfy embedding contract: "
                    f"{modality}={actual_dimension}, expected={expected_dimension}"
                )
        quality_json = json.dumps(asdict(normalized.quality), ensure_ascii=False, default=_json_default)
        with self._write_scope():
            self.connection.execute(
                """
                INSERT OR REPLACE INTO observations(
                    event_id,candidate_id,camera_id,capture_session_id,track_id,timestamp,
                    end_timestamp,appearance,appearance_dimension,gait,gait_dimension,
                    quality_json,model_version,threshold_version,source_event_json,
                    feature_schema,calibration_version,challenge_id,challenge_response_json,
                    appearance_request_id,metadata_json,artifact_sha256,preprocess_version,
                    joint_format,sequence_length,tta_mode,coordinate_contract,
                    embedding_dimensions_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    normalized.event_id,
                    candidate_id,
                    normalized.camera_id,
                    normalized.capture_session_id,
                    normalized.track_id,
                    normalized.timestamp,
                    normalized.end_timestamp,
                    sqlite3.Binary(appearance) if appearance is not None else None,
                    appearance_dimension,
                    sqlite3.Binary(gait) if gait is not None else None,
                    gait_dimension,
                    quality_json,
                    normalized.model_version,
                    normalized.threshold_version,
                    json.dumps(normalized.source_event_ids, ensure_ascii=False),
                    normalized.feature_schema,
                    normalized.calibration_version,
                    normalized.challenge_id,
                    json.dumps(normalized.challenge_response, ensure_ascii=False, default=_json_default),
                    normalized.appearance_request_id,
                    json.dumps(normalized.metadata, ensure_ascii=False, default=_json_default),
                    normalized.artifact_sha256,
                    normalized.preprocess_version,
                    normalized.joint_format,
                    normalized.sequence_length,
                    normalized.tta_mode,
                    normalized.coordinate_contract,
                    json.dumps(dict(normalized.embedding_dimensions), ensure_ascii=False),
                ),
            )

    def save_appearance_request(self, request: AppearanceAbsorptionRequest) -> None:
        """持久化一次性外观授权令牌。"""
        with self._write_scope():
            self.connection.execute(
                """
                INSERT INTO appearance_requests(
                    request_id,identity_id,issued_by_event_id,candidate_id,
                    gait_probability,gait_quality,issued_at,expires_at,status,
                    response_event_id,camera_id,track_id,metadata_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(request_id) DO UPDATE SET
                    identity_id=excluded.identity_id,
                    issued_by_event_id=excluded.issued_by_event_id,
                    candidate_id=excluded.candidate_id,
                    gait_probability=excluded.gait_probability,
                    gait_quality=excluded.gait_quality,
                    issued_at=excluded.issued_at,
                    expires_at=excluded.expires_at,
                    status=excluded.status,
                    response_event_id=excluded.response_event_id,
                    camera_id=excluded.camera_id,
                    track_id=excluded.track_id,
                    metadata_json=excluded.metadata_json
                """,
                (
                    request.request_id,
                    request.identity_id,
                    request.issued_by_event_id,
                    request.candidate_id,
                    request.gait_probability,
                    request.gait_quality,
                    request.issued_at,
                    request.expires_at,
                    request.status,
                    request.response_event_id,
                    request.camera_id,
                    request.track_id,
                    json.dumps(request.metadata, ensure_ascii=False, default=_json_default),
                ),
            )

    @staticmethod
    def _appearance_request_from_row(row: sqlite3.Row) -> AppearanceAbsorptionRequest:
        """从一行 SQLite 记录恢复外观请求。"""
        return AppearanceAbsorptionRequest(
            request_id=row["request_id"],
            identity_id=row["identity_id"],
            issued_by_event_id=row["issued_by_event_id"],
            candidate_id=row["candidate_id"],
            gait_probability=float(row["gait_probability"]),
            gait_quality=float(row["gait_quality"]),
            issued_at=float(row["issued_at"]),
            expires_at=float(row["expires_at"]),
            status=row["status"],
            response_event_id=row["response_event_id"],
            camera_id=row["camera_id"],
            track_id=row["track_id"],
            metadata=dict(json.loads(row["metadata_json"])),
        )

    def load_appearance_requests(self) -> list[AppearanceAbsorptionRequest]:
        """按签发顺序加载外观令牌，供引擎恢复状态。"""
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM appearance_requests ORDER BY issued_at"
            ).fetchall()
        return [self._appearance_request_from_row(row) for row in rows]

    def save_evidence(
        self,
        *,
        evidence_id: str,
        candidate_id: str | None,
        event_id: str,
        identity_id: str | None,
        kind: str,
        score: float | None,
        margin: float | None,
        independent: bool,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """记录一个带分数、间隔和独立性标记的证据项。"""
        with self._write_scope():
            self.connection.execute(
                """
                INSERT OR REPLACE INTO evidence(
                    evidence_id,candidate_id,event_id,identity_id,kind,score,margin,
                    independent,payload_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    evidence_id,
                    candidate_id,
                    event_id,
                    identity_id,
                    kind,
                    score,
                    margin,
                    int(independent),
                    json.dumps(payload or {}, ensure_ascii=False, default=_json_default),
                    time.time(),
                ),
            )

    def save_gait_event_proposal(
        self,
        *,
        candidate_id: str,
        event_key: str,
        vector: np.ndarray,
        stability: float,
        model_version: str,
        feature_schema: str,
        calibration_version: str = "heuristic-default-v1",
        artifact_sha256: str = "unverified",
        preprocess_version: str = "unversioned-v1",
        joint_format: str = "unknown",
        sequence_length: int | None = None,
        tta_mode: str = "unknown",
        coordinate_contract: str = "unknown",
        embedding_dimensions: dict[str, int] | None = None,
        camera_id: str = "unknown-camera",
        capture_session_id: str = "unknown-session",
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        created_at: float | None = None,
    ) -> None:
        """持久化一个跨会话自动建号事件，重复事件键只保留首次证据。"""

        packed, dimension = _pack_vector(vector)
        if packed is None or dimension is None:
            raise ValueError("gait event proposal vector must be finite and non-zero")
        expected_dimension = dict(embedding_dimensions or {}).get("gait")
        if expected_dimension is not None and int(expected_dimension) != int(dimension):
            raise ValueError(
                "gait event proposal dimension does not satisfy embedding contract: "
                f"gait={dimension}, expected={expected_dimension}"
            )
        with self._write_scope():
            self.connection.execute(
                """
                INSERT OR IGNORE INTO automation_gait_events(
                    candidate_id,event_key,vector,dimension,stability,
                    model_version,feature_schema,calibration_version,
                    artifact_sha256,preprocess_version,
                    joint_format,sequence_length,tta_mode,coordinate_contract,
                    embedding_dimensions_json,camera_id,capture_session_id,
                    start_timestamp,end_timestamp,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    candidate_id,
                    event_key,
                    sqlite3.Binary(packed),
                    dimension,
                    float(stability),
                    model_version,
                    feature_schema,
                    calibration_version,
                    artifact_sha256,
                    preprocess_version,
                    joint_format,
                    sequence_length,
                    tta_mode,
                    coordinate_contract,
                    json.dumps(dict(embedding_dimensions or {}), ensure_ascii=False),
                    camera_id,
                    capture_session_id,
                    start_timestamp,
                    end_timestamp,
                    time.time() if created_at is None else float(created_at),
                ),
            )

    def load_gait_event_proposals(
        self,
        candidate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """加载跨重启自动建号事件提案。"""

        with self._lock:
            if candidate_id is None:
                rows = self.connection.execute(
                    "SELECT * FROM automation_gait_events ORDER BY created_at"
                ).fetchall()
            else:
                rows = self.connection.execute(
                    """
                    SELECT * FROM automation_gait_events
                    WHERE candidate_id=? ORDER BY created_at
                    """,
                    (candidate_id,),
                ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            vector = _unpack_vector(row["vector"], row["dimension"])
            if vector is None:
                continue
            result.append(
                {
                    "candidate_id": row["candidate_id"],
                    "event_key": row["event_key"],
                    "vector": vector,
                    "stability": float(row["stability"]),
                    "model_version": row["model_version"],
                    "feature_schema": row["feature_schema"],
                    "calibration_version": row["calibration_version"],
                    "artifact_sha256": row["artifact_sha256"],
                    "preprocess_version": row["preprocess_version"],
                    "joint_format": row["joint_format"],
                    "sequence_length": row["sequence_length"],
                    "tta_mode": row["tta_mode"],
                    "coordinate_contract": row["coordinate_contract"],
                    "embedding_dimensions": dict(
                        json.loads(row["embedding_dimensions_json"] or "{}")
                    ),
                    "camera_id": row["camera_id"],
                    "capture_session_id": row["capture_session_id"],
                    "start_timestamp": row["start_timestamp"],
                    "end_timestamp": row["end_timestamp"],
                    "created_at": float(row["created_at"]),
                }
            )
        return result

    def delete_gait_event_proposals(self, candidate_id: str) -> None:
        """删除候选人已消费或明确作废的跨会话步态事件。"""

        with self._write_scope():
            self.connection.execute(
                "DELETE FROM automation_gait_events WHERE candidate_id=?",
                (candidate_id,),
            )

    def save_gait_enrollment_event(
        self,
        event: GaitEnrollmentEvent,
        *,
        replace_existing: bool = False,
    ) -> bool:
        """保存事件；同一事件可更新代表向量，但不会增加事件数量。"""

        packed, dimension = _pack_vector(event.vector)
        if packed is None or dimension is None:
            raise ValueError("gait enrollment event vector must be finite and non-zero")
        expected_dimension = dict(event.embedding_dimensions).get("gait")
        if expected_dimension is not None and int(expected_dimension) != int(dimension):
            raise ValueError(
                "gait enrollment event dimension does not satisfy embedding contract"
            )
        with self._write_scope():
            operation = "INSERT OR REPLACE" if replace_existing else "INSERT OR IGNORE"
            cursor = self.connection.execute(
                f"""
                {operation} INTO gait_enrollment_events(
                    identity_id,event_key,event_id,camera_id,capture_session_id,track_id,
                    vector,dimension,stability,quality,sample_count,view_angle,created_at,
                    model_version,feature_schema,calibration_version,artifact_sha256,
                    preprocess_version,joint_format,sequence_length,tta_mode,
                    coordinate_contract,embedding_dimensions_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.identity_id,
                    event.event_key,
                    event.event_id,
                    event.camera_id,
                    event.capture_session_id,
                    event.track_id,
                    sqlite3.Binary(packed),
                    dimension,
                    float(event.stability),
                    float(event.quality),
                    int(event.sample_count),
                    event.view_angle,
                    float(event.created_at),
                    event.model_version,
                    event.feature_schema,
                    event.calibration_version,
                    event.artifact_sha256,
                    event.preprocess_version,
                    event.joint_format,
                    event.sequence_length,
                    event.tta_mode,
                    event.coordinate_contract,
                    json.dumps(dict(event.embedding_dimensions), ensure_ascii=False),
                ),
            )
            return cursor.rowcount > 0

    def load_gait_enrollment_events(
        self,
        identity_id: str | None = None,
    ) -> list[GaitEnrollmentEvent]:
        """恢复视觉身份下已接受的独立步态事件。"""

        with self._lock:
            if identity_id is None:
                rows = self.connection.execute(
                    "SELECT * FROM gait_enrollment_events ORDER BY created_at"
                ).fetchall()
            else:
                rows = self.connection.execute(
                    """
                    SELECT * FROM gait_enrollment_events
                    WHERE identity_id=? ORDER BY created_at
                    """,
                    (identity_id,),
                ).fetchall()
        events: list[GaitEnrollmentEvent] = []
        for row in rows:
            vector = _unpack_vector(row["vector"], row["dimension"])
            if vector is None:
                continue
            events.append(
                GaitEnrollmentEvent(
                    identity_id=row["identity_id"],
                    event_key=row["event_key"],
                    event_id=row["event_id"],
                    camera_id=row["camera_id"],
                    capture_session_id=row["capture_session_id"],
                    track_id=row["track_id"],
                    vector=vector,
                    stability=float(row["stability"]),
                    quality=float(row["quality"]),
                    sample_count=int(row["sample_count"]),
                    view_angle=row["view_angle"],
                    created_at=float(row["created_at"]),
                    model_version=row["model_version"],
                    feature_schema=row["feature_schema"],
                    calibration_version=row["calibration_version"],
                    artifact_sha256=row["artifact_sha256"],
                    preprocess_version=row["preprocess_version"],
                    joint_format=row["joint_format"],
                    sequence_length=(
                        int(row["sequence_length"])
                        if row["sequence_length"] is not None
                        else None
                    ),
                    tta_mode=row["tta_mode"],
                    coordinate_contract=row["coordinate_contract"],
                    embedding_dimensions=dict(
                        json.loads(row["embedding_dimensions_json"] or "{}")
                    ),
                )
            )
        return events

    def delete_gait_enrollment_events(self, identity_id: str) -> None:
        """显式删除一个视觉身份的步态事件记录。"""

        with self._write_scope():
            self.connection.execute(
                "DELETE FROM gait_enrollment_events WHERE identity_id=?",
                (identity_id,),
            )

    def list_evidence(self, candidate_id: str) -> list[dict[str, Any]]:
        """返回候选证据行，并将载荷解码为字典。"""
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM evidence WHERE candidate_id=? ORDER BY created_at", (candidate_id,)
            ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]

    def event_metadata(self, event_id: str) -> dict[str, Any] | None:
        """返回供审计/调试检查使用的原始观察行。"""
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM observations WHERE event_id=?", (event_id,)
            ).fetchone()
        return dict(row) if row else None

    def observations_for_candidate(self, candidate_id: str) -> list[Observation]:
        """恢复与一个候选关联的所有观察值。"""
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM observations WHERE candidate_id=? ORDER BY timestamp",
                (candidate_id,),
            ).fetchall()
        observations: list[Observation] = []
        for row in rows:
            quality_values = dict(json.loads(row["quality_json"]))
            quality_values["reasons"] = tuple(quality_values.get("reasons", ()))
            quality = TrackQuality(**quality_values)
            observations.append(
                Observation(
                    event_id=row["event_id"],
                    camera_id=row["camera_id"],
                    capture_session_id=row["capture_session_id"],
                    track_id=row["track_id"],
                    timestamp=float(row["timestamp"]),
                    end_timestamp=row["end_timestamp"],
                    features=FeatureBundle(
                        appearance=_unpack_vector(row["appearance"], row["appearance_dimension"]),
                        gait=_unpack_vector(row["gait"], row["gait_dimension"]),
                    ),
                    quality=quality,
                    model_version=row["model_version"],
                    feature_schema=row["feature_schema"],
                    artifact_sha256=row["artifact_sha256"],
                    preprocess_version=row["preprocess_version"],
                    joint_format=row["joint_format"],
                    sequence_length=(
                        int(row["sequence_length"])
                        if row["sequence_length"] is not None
                        else None
                    ),
                    tta_mode=row["tta_mode"],
                    coordinate_contract=row["coordinate_contract"],
                    embedding_dimensions=dict(
                        json.loads(row["embedding_dimensions_json"] or "{}")
                    ),
                    calibration_version=row["calibration_version"],
                    threshold_version=row["threshold_version"],
                    source_event_ids=tuple(json.loads(row["source_event_json"])),
                    challenge_id=row["challenge_id"],
                    challenge_response=dict(json.loads(row["challenge_response_json"])),
                    appearance_request_id=row["appearance_request_id"],
                    metadata=dict(json.loads(row["metadata_json"])),
                )
            )
        return observations

    def snapshot_identity(self, identity_id: str, reason: str = "before-update") -> int:
        """在变更前保存正式原型，并返回快照 ID。"""
        prototypes = [p for p in self.load_prototypes() if p.identity_id == identity_id and p.zone == "formal"]
        payload = []
        for prototype in prototypes:
            payload.append(
                {
                    "identity_id": prototype.identity_id,
                    "modality": prototype.modality,
                    "vector": base64.b64encode(prototype.vector.astype(np.float32).tobytes()).decode("ascii"),
                    "dimension": int(prototype.vector.size),
                    "quality": prototype.quality,
                    "camera_id": prototype.camera_id,
                    "view_angle": prototype.view_angle,
                    "clothing_tag": prototype.clothing_tag,
                    "source_event_id": prototype.source_event_id,
                    "prototype_id": prototype.prototype_id,
                    "created_at": prototype.created_at,
                    "model_version": prototype.model_version,
                    "feature_schema": prototype.feature_schema,
                    "artifact_sha256": prototype.artifact_sha256,
                    "preprocess_version": prototype.preprocess_version,
                    "joint_format": prototype.joint_format,
                    "sequence_length": prototype.sequence_length,
                    "tta_mode": prototype.tta_mode,
                    "coordinate_contract": prototype.coordinate_contract,
                    "embedding_dimensions": dict(prototype.embedding_dimensions),
                }
            )
        with self._write_scope():
            cursor = self.connection.execute(
                "INSERT INTO snapshots(identity_id,reason,payload_json,created_at) VALUES(?,?,?,?)",
                (identity_id, reason, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            return int(cursor.lastrowid)

    def restore_snapshot(self, snapshot_id: int) -> list[Prototype]:
        """恢复之前的正式原型快照，并返回恢复后的值。"""
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown snapshot: {snapshot_id}")
        values = json.loads(row["payload_json"])
        restored: list[Prototype] = []
        for value in values:
            raw = base64.b64decode(value["vector"])
            vector = _unpack_vector(raw, value["dimension"])
            if vector is None:
                continue
            restored.append(
                Prototype(
                    identity_id=value["identity_id"],
                    modality=value["modality"],
                    vector=vector,
                    zone="formal",
                    quality=float(value["quality"]),
                    camera_id=value.get("camera_id"),
                    view_angle=value.get("view_angle"),
                    clothing_tag=value.get("clothing_tag"),
                    source_event_id=value.get("source_event_id"),
                    prototype_id=value["prototype_id"],
                    created_at=float(value["created_at"]),
                    model_version=value.get("model_version", "unconfigured"),
                    feature_schema=value.get("feature_schema", "unconfigured-v1"),
                    artifact_sha256=value.get("artifact_sha256", "unverified"),
                    preprocess_version=value.get("preprocess_version", "unversioned-v1"),
                    joint_format=value.get("joint_format", "unknown"),
                    sequence_length=value.get("sequence_length"),
                    tta_mode=value.get("tta_mode", "unknown"),
                    coordinate_contract=value.get("coordinate_contract", "unknown"),
                    embedding_dimensions=dict(value.get("embedding_dimensions", {})),
                )
            )
        with self._write_scope():
            self.connection.execute("DELETE FROM prototypes WHERE identity_id=? AND zone='formal'", (row["identity_id"],))
        self.save_prototypes(restored)
        return restored

    def audit(self, action: str, entity_id: str | None, payload: dict[str, Any] | None = None) -> None:
        """向只追加审计表写入结构化操作记录。"""
        with self._write_scope():
            self.connection.execute(
                "INSERT INTO audit_log(action,entity_id,payload_json,created_at) VALUES(?,?,?,?)",
                (action, entity_id, json.dumps(payload or {}, ensure_ascii=False, default=_json_default), time.time()),
            )

    def audit_log(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        """读取所有审计记录，可选限定到某个实体。"""
        with self._lock:
            if entity_id is None:
                rows = self.connection.execute("SELECT * FROM audit_log ORDER BY audit_id").fetchall()
            else:
                rows = self.connection.execute(
                    "SELECT * FROM audit_log WHERE entity_id=? ORDER BY audit_id", (entity_id,)
                ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]
