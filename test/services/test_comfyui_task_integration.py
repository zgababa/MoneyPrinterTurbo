import unittest
from unittest.mock import patch

from app.models.schema import VideoParams
from app.services import comfyui
from app.services import state as sm
from app.services import task as task_service


class TestComfyUITaskIntegration(unittest.TestCase):
    def test_task_preflight_rejects_missing_api_key_before_script_generation(self):
        params = VideoParams(
            video_subject="ComfyUI preflight",
            video_source="comfyui",
        )
        memory_state = sm.MemoryState()
        with (
            patch.object(comfyui, "is_enabled", return_value=False),
            patch.object(task_service.sm, "state", memory_state),
            patch.object(task_service, "generate_script") as generate_script,
        ):
            result = task_service.start(
                "comfyui-preflight", params, stop_at="materials"
            )

        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("ComfyUI Cloud", result["error"])
        generate_script.assert_not_called()

    def test_task_records_comfyui_material_error_and_remote_job_id(self):
        params = VideoParams(
            video_subject="ComfyUI material failure",
            video_source="comfyui",
        )
        memory_state = sm.MemoryState()
        error = comfyui.ComfyUIUnconfirmedTaskError(
            "remote state unknown", task_id="job-recover"
        )
        with (
            patch.object(task_service.sm, "state", memory_state),
            patch.object(task_service.material, "download_videos", side_effect=error),
        ):
            result = task_service.get_video_materials(
                task_id="comfyui-material-error",
                params=params,
                video_terms=["scene"],
                audio_duration=5,
            )

        self.assertIsNone(result)
        failed = memory_state.get_task("comfyui-material-error")
        self.assertEqual(failed["failed_stage"], "materials")
        self.assertEqual(failed["comfyui_task_id"], "job-recover")

    def test_task_records_job_id_from_terminal_provider_error(self):
        params = VideoParams(
            video_subject="ComfyUI terminal failure",
            video_source="comfyui",
        )
        memory_state = sm.MemoryState()
        error = comfyui.ComfyUIError("remote job failed", task_id="job-terminal")
        with (
            patch.object(task_service.sm, "state", memory_state),
            patch.object(task_service.material, "download_videos", side_effect=error),
        ):
            result = task_service.get_video_materials(
                task_id="comfyui-terminal-error",
                params=params,
                video_terms=["scene"],
                audio_duration=5,
            )

        self.assertIsNone(result)
        failed = memory_state.get_task("comfyui-terminal-error")
        self.assertEqual(failed["failed_stage"], "materials")
        self.assertEqual(failed["comfyui_task_id"], "job-terminal")

    def test_task_records_job_id_when_generated_video_download_fails(self):
        params = VideoParams(
            video_subject="ComfyUI download failure",
            video_source="comfyui",
        )
        memory_state = sm.MemoryState()
        error = comfyui.ComfyUIDownloadError(
            "generated video download failed", task_id="job-paid-result"
        )
        with (
            patch.object(task_service.sm, "state", memory_state),
            patch.object(task_service.material, "download_videos", side_effect=error),
        ):
            result = task_service.get_video_materials(
                task_id="comfyui-download-error",
                params=params,
                video_terms=["scene"],
                audio_duration=5,
            )

        self.assertIsNone(result)
        failed = memory_state.get_task("comfyui-download-error")
        self.assertEqual(failed["failed_stage"], "materials")
        self.assertEqual(failed["comfyui_task_id"], "job-paid-result")
