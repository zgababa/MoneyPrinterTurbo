"""ComfyUI Cloud text-to-video client.

Talks directly to the real ComfyUI Cloud API (Jobs/Assets, v2) via the
official ``comfy-sdk`` package, submitting a single vendored workflow
(``comfyui_workflows/api_minimax_h3_max_t2v.json``) that drives the
``MinimaxHailuo03TextToVideoNode`` Partner Node ("MiniMax H3 Max") with
``prompt_expansion_mode: "balanced"`` -- the model rewrites our short
LLM-generated search term into a fuller prompt server-side before
generation, since a bare 1-3 word term (see ``llm.generate_terms``) is not
enough prompt for good output on its own. The workflow's "Prompt Enhance"
and "Upscale to 2K" boolean toggles are left at their vendored default of
``False`` and never patched: both feed a ``ComfySwitchNode`` whose
``on_false``/``on_true`` inputs are lazy (confirmed via this node's
``/api/object_info`` entry), so the unused branches -- and their own paid
API calls -- never execute.

MoneyPrinterTurbo's other paid providers (``volcengine_seedance``, ``muapi``,
``metaso_minimax``) are hand-rolled against ``requests``; this one depends on
``comfy-sdk`` instead -- see docs/adr/0001-comfy-sdk-dependency.md for why.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect

DEFAULT_BASE_URL = "https://cloud.comfy.org"
# MiniMax H3 Max's own real bounds, confirmed via this node's
# /api/object_info entry (MinimaxHailuo03TextToVideoNode).
DEFAULT_MIN_DURATION_SECONDS = 5
DEFAULT_MAX_DURATION_SECONDS = 15
_WORKFLOW_PATH = (
    Path(__file__).parent / "comfyui_workflows" / "api_minimax_h3_max_t2v.json"
)
# Nodes are located by their ``_meta.title`` (the node's title in the ComfyUI
# editor, preserved by "Export (API)"), not by id: ids change on every
# re-export, titles are chosen by whoever builds the workflow.
#   MPT_PROMPT   (required) text input that receives the search term
#   MPT_OUTPUT   (required) SaveVideo node whose output is the clip
#   MPT_SETTINGS (optional) node whose ``model.duration`` / ``model.ratio``
#                inputs, when present, are patched per clip
#   MPT_RATIO    (optional) ResolutionSelector node; its ``aspect_ratio`` is
#                forced to 9:16 (the only aspect this provider supports)
PROMPT_NODE_TITLE = "MPT_PROMPT"
OUTPUT_NODE_TITLE = "MPT_OUTPUT"
SETTINGS_NODE_TITLE = "MPT_SETTINGS"
RATIO_NODE_TITLE = "MPT_RATIO"
_PORTRAIT_ASPECT_RATIO_LABEL = "9:16 (Portrait Widescreen)"
# Input names tried, in order, to carry the prompt text on the MPT_PROMPT node.
_PROMPT_INPUT_NAMES = ("value", "text", "prompt")
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
_IN_PROGRESS_STATUSES = frozenset({"queued", "running", "canceling"})
_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "canceled", "expired"})


class ComfyUIError(RuntimeError):
    """Deterministic configuration, request, or response error."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        self.task_id = task_id


class ComfyUIUnconfirmedTaskError(ComfyUIError):
    """The remote job may exist, but its final state cannot be confirmed."""


class ComfyUIDownloadError(ComfyUIError):
    """A billable job completed, but its output could not be downloaded."""


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    settings = config.app if settings is None else settings
    configured = str(settings.get("comfyui_api_key", "") or "").strip()
    environment_key = os.getenv("COMFYUI_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("comfyui_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


class _ComfyUIClient:
    """Thin wrapper giving ``comfy_low.transport.ComfyLow`` the
    submit_job/get_job shape this module calls, translating to the SDK's
    real method names (``post_jobs``/``get_job``) and unwrapping its
    pydantic response models to plain dicts."""

    def __init__(self, low, api_key: str) -> None:
        self._low = low
        self._api_key = api_key

    def submit_job(self, workflow: dict) -> str:
        # extra_data.api_key_comfy_org is required for ComfyUI's Partner
        # Nodes (e.g. MiniMax H3) to authorize themselves; the request's own
        # Authorization header only authorizes submitting the Cloud job
        # itself. A fresh idempotency key stops a network-level retry of
        # this call from silently submitting a duplicate paid job.
        job = self._low.post_jobs(
            workflow,
            idempotency_key=str(uuid.uuid4()),
            extra_data={"api_key_comfy_org": self._api_key},
        )
        return job.id

    def get_job(self, job_id: str) -> dict:
        return self._low.get_job(job_id).model_dump(mode="json")


def _client(api_key: str) -> _ComfyUIClient:
    from comfy_low.transport import ComfyLow

    return _ComfyUIClient(ComfyLow(base_url=_base_url(), api_key=api_key), api_key)


def _workflow_path() -> Path:
    # ``comfyui_workflow`` is a file name inside comfyui_workflows/ or a path.
    name = str(config.app.get("comfyui_workflow", "") or "").strip()
    if not name:
        return _WORKFLOW_PATH
    path = Path(name)
    return path if path.is_absolute() else _WORKFLOW_PATH.parent / path


def _load_workflow() -> dict:
    with open(_workflow_path(), encoding="utf-8") as f:
        return json.load(f)


def _find_node_id(workflow: Mapping[str, Any], title: str, *, required: bool = True):
    matches = [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, Mapping) and node.get("_meta", {}).get("title") == title
    ]
    if len(matches) > 1:
        raise ComfyUIError(
            f"ComfyUI workflow has {len(matches)} nodes titled {title!r}; "
            "exactly one is expected"
        )
    if not matches and required:
        raise ComfyUIError(
            f"ComfyUI workflow has no node titled {title!r}; rename the node "
            "in the ComfyUI editor and re-export the workflow (API format)"
        )
    return matches[0] if matches else None


def _duration_bounds() -> tuple[int, int]:
    def read(key: str, default: int) -> int:
        try:
            value = int(config.app.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value >= 1 else default

    minimum = read("comfyui_min_duration", DEFAULT_MIN_DURATION_SECONDS)
    maximum = read("comfyui_max_duration", DEFAULT_MAX_DURATION_SECONDS)
    return minimum, max(minimum, maximum)


def _poll_interval() -> float:
    try:
        return float(
            config.app.get("comfyui_poll_interval", DEFAULT_POLL_INTERVAL_SECONDS)
        )
    except (TypeError, ValueError):
        return DEFAULT_POLL_INTERVAL_SECONDS


def _run_timeout() -> float:
    try:
        return float(
            config.app.get("comfyui_run_timeout", DEFAULT_RUN_TIMEOUT_SECONDS)
        )
    except (TypeError, ValueError):
        return DEFAULT_RUN_TIMEOUT_SECONDS


def _wait_for_job(client, job_id: str) -> dict:
    deadline = time.monotonic() + _run_timeout()
    poll_interval = _poll_interval()
    while True:
        job = client.get_job(job_id)
        status = str(job.get("status") or "").strip().lower()
        if status == "succeeded":
            return job
        if status in _TERMINAL_FAILURE_STATUSES:
            error = job.get("error")
            message = (
                error.get("message") if isinstance(error, dict) else str(error or "")
            )
            raise ComfyUIError(
                f"ComfyUI job did not produce a video: id={job_id}, "
                f"status={status}, detail={message}",
                task_id=job_id,
            )
        if status not in _IN_PROGRESS_STATUSES:
            raise ComfyUIUnconfirmedTaskError(
                f"ComfyUI returned an unknown job status: id={job_id}, "
                f"status={status!r}",
                task_id=job_id,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ComfyUIUnconfirmedTaskError(
                "ComfyUI job is still running after the configured local "
                f"wait timeout: id={job_id}",
                task_id=job_id,
            )
        time.sleep(min(poll_interval, remaining))


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    api_key = get_api_key()
    if not api_key:
        raise ComfyUIError("ComfyUI Cloud requires an API key")

    term = str(search_term or "").strip()
    if not term:
        raise ComfyUIError("ComfyUI search term must not be empty")

    aspect = VideoAspect(video_aspect)
    if aspect != VideoAspect.portrait:
        raise ComfyUIError(
            "ComfyUI video generation only supports the 9:16 aspect ratio "
            f"in this version; got {aspect.value!r}"
        )

    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ComfyUIError("ComfyUI clip duration must be a positive integer") from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)

    workflow = _load_workflow()

    prompt_node_id = _find_node_id(workflow, PROMPT_NODE_TITLE)
    output_node_id = _find_node_id(workflow, OUTPUT_NODE_TITLE)
    settings_node_id = _find_node_id(workflow, SETTINGS_NODE_TITLE, required=False)
    ratio_node_id = _find_node_id(workflow, RATIO_NODE_TITLE, required=False)

    prompt_inputs = workflow[prompt_node_id]["inputs"]
    prompt_field = next((k for k in _PROMPT_INPUT_NAMES if k in prompt_inputs), None)
    if prompt_field is None:
        raise ComfyUIError(
            f"ComfyUI node {PROMPT_NODE_TITLE!r} has none of the text inputs "
            f"{_PROMPT_INPUT_NAMES}; found {sorted(prompt_inputs)}"
        )
    prompt_inputs[prompt_field] = term

    if settings_node_id is not None:
        settings_inputs = workflow[settings_node_id]["inputs"]
        if "model.duration" in settings_inputs:
            settings_inputs["model.duration"] = duration
        if "model.ratio" in settings_inputs:
            settings_inputs["model.ratio"] = aspect.value

    if ratio_node_id is not None:
        ratio_inputs = workflow[ratio_node_id]["inputs"]
        if "aspect_ratio" not in ratio_inputs:
            raise ComfyUIError(
                f"ComfyUI node {RATIO_NODE_TITLE!r} has no 'aspect_ratio' input; "
                f"found {sorted(ratio_inputs)}"
            )
        ratio_inputs["aspect_ratio"] = _PORTRAIT_ASPECT_RATIO_LABEL

    client = _client(api_key)
    try:
        job_id = client.submit_job(workflow)
    except Exception as exc:
        status_code = _status_code(exc)
        if status_code is not None and 400 <= status_code < 500:
            raise ComfyUIError(
                f"ComfyUI job submission rejected: HTTP {status_code}, {exc}"
            ) from exc
        raise ComfyUIUnconfirmedTaskError(
            "ComfyUI submission returned no confirmed response; a paid job "
            f"may already exist remotely: error={type(exc).__name__}, detail={exc}"
        ) from exc

    job = _wait_for_job(client, job_id)

    # The Cloud API's per-output "type" field does not reliably say "video"
    # for this workflow's SaveVideo node -- a real run returned type="image"
    # for a video/*.mp4 output. Since the workflow is fixed and vendored, the
    # producing node is the one titled MPT_OUTPUT, so match on its id instead.
    video_url = ""
    for output in job.get("outputs", []):
        if output.get("node_id") == output_node_id:
            video_url = str(output.get("url") or "")
            break
    if not video_url:
        raise ComfyUIError(
            f"ComfyUI job succeeded without a video output: id={job_id}",
            task_id=job_id,
        )

    return [
        MaterialInfo(
            provider="comfyui",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "comfyui",
                "search_term": term,
                "asset_id": job_id,
            },
        )
    ]
