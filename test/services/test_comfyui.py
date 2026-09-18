import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import config
from app.models.schema import VideoAspect
from app.services import comfyui


class TestComfyUIService(unittest.TestCase):
    def setUp(self):
        self.original_app = dict(config.app)
        config.app.update({"comfyui_api_key": "comfy-test-key"})

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app)

    def test_api_key_prefers_config_then_environment_variable(self):
        with patch.dict(os.environ, {"COMFYUI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(comfyui.get_api_key(), "comfy-test-key")
            config.app["comfyui_api_key"] = ""
            self.assertEqual(comfyui.get_api_key(), "env-key")
            os.environ["COMFYUI_API_KEY"] = ""
            self.assertEqual(comfyui.get_api_key(), "")

    def test_is_enabled_reflects_api_key_presence(self):
        self.assertTrue(comfyui.is_enabled())
        config.app["comfyui_api_key"] = ""
        with patch.dict(os.environ, {"COMFYUI_API_KEY": ""}, clear=False):
            self.assertFalse(comfyui.is_enabled())

    def test_submit_poll_and_parse_successful_video(self):
        client = MagicMock()
        client.submit_job.return_value = "job-123"
        client.get_job.return_value = {
            "status": "succeeded",
            "outputs": [
                {
                    "id": "asset-1",
                    "node_id": "2",
                    "type": "video",
                    "url": "https://cloud.comfy.org/assets/asset-1/content",
                }
            ],
        }

        with patch.object(comfyui, "_client", return_value=client):
            items = comfyui.generate_videos(
                search_term="a red bicycle",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.provider, "comfyui")
        self.assertEqual(item.url, "https://cloud.comfy.org/assets/asset-1/content")
        self.assertEqual(item.duration, 5)
        self.assertEqual(item.source_info["asset_id"], "job-123")

        submitted_workflow = client.submit_job.call_args.args[0]
        self.assertEqual(
            submitted_workflow["14"]["inputs"]["value"], "a red bicycle"
        )
        self.assertEqual(submitted_workflow["31"]["inputs"]["model.duration"], 5)
        self.assertEqual(
            submitted_workflow["31"]["inputs"]["model.ratio"], "9:16"
        )

    def test_missing_api_key_fails_before_submission(self):
        config.app["comfyui_api_key"] = ""
        with (
            patch.dict(os.environ, {"COMFYUI_API_KEY": ""}, clear=False),
            patch.object(comfyui, "_client") as client_factory,
        ):
            with self.assertRaises(comfyui.ComfyUIError):
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
        client_factory.assert_not_called()

    def test_empty_search_term_fails_before_paid_submission(self):
        with patch.object(comfyui, "_client") as client_factory:
            with self.assertRaises(comfyui.ComfyUIError):
                comfyui.generate_videos(
                    search_term="   ",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
        client_factory.assert_not_called()

    def test_landscape_aspect_is_rejected_before_paid_submission(self):
        with patch.object(comfyui, "_client") as client_factory:
            with self.assertRaises(comfyui.ComfyUIError):
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.landscape,
                )
        client_factory.assert_not_called()

    def test_clip_duration_is_clamped_to_the_configured_range(self):
        config.app["comfyui_min_duration"] = 3
        config.app["comfyui_max_duration"] = 10
        client = MagicMock()
        client.submit_job.return_value = "job-clamped"
        client.get_job.return_value = {
            "status": "succeeded",
            "outputs": [
                {"id": "asset-1", "node_id": "2", "type": "video", "url": "https://x/y"}
            ],
        }
        with patch.object(comfyui, "_client", return_value=client):
            items = comfyui.generate_videos(
                search_term="term",
                minimum_duration=99,
                video_aspect=VideoAspect.portrait,
            )
        self.assertEqual(items[0].duration, 10)
        submitted_workflow = client.submit_job.call_args.args[0]
        self.assertEqual(submitted_workflow["31"]["inputs"]["model.duration"], 10)

    def test_submission_server_error_is_unconfirmed_not_rejected(self):
        client = MagicMock()
        error = RuntimeError("server exploded")
        error.response = SimpleNamespace(status_code=500)
        client.submit_job.side_effect = error
        with patch.object(comfyui, "_client", return_value=client):
            with self.assertRaises(comfyui.ComfyUIUnconfirmedTaskError):
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )

    def test_submission_client_error_is_rejected_deterministically(self):
        client = MagicMock()
        error = RuntimeError("bad workflow")
        error.response = SimpleNamespace(status_code=422)
        client.submit_job.side_effect = error
        with patch.object(comfyui, "_client", return_value=client):
            with self.assertRaises(comfyui.ComfyUIError) as raised:
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
        self.assertNotIsInstance(
            raised.exception, comfyui.ComfyUIUnconfirmedTaskError
        )

    def test_submission_network_error_is_unconfirmed(self):
        client = MagicMock()
        client.submit_job.side_effect = ConnectionError("dns failure")
        with patch.object(comfyui, "_client", return_value=client):
            with self.assertRaises(comfyui.ComfyUIUnconfirmedTaskError):
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )

    def test_poll_continues_through_queued_and_running_until_succeeded(self):
        client = MagicMock()
        client.submit_job.return_value = "job-poll"
        client.get_job.side_effect = [
            {"status": "queued", "outputs": []},
            {"status": "running", "outputs": []},
            {
                "status": "succeeded",
                "outputs": [
                    {"id": "a", "node_id": "2", "type": "video", "url": "https://x/y"}
                ],
            },
        ]
        with (
            patch.object(comfyui, "_client", return_value=client),
            patch.object(comfyui.time, "sleep") as sleep,
        ):
            items = comfyui.generate_videos(
                search_term="term",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )
        self.assertEqual(items[0].url, "https://x/y")
        self.assertEqual(client.get_job.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_terminal_failure_status_raises_deterministic_error_with_job_id(self):
        client = MagicMock()
        client.submit_job.return_value = "job-failed"
        client.get_job.return_value = {
            "status": "failed",
            "error": {"message": "node crashed"},
            "outputs": [],
        }
        with patch.object(comfyui, "_client", return_value=client):
            with self.assertRaises(comfyui.ComfyUIError) as raised:
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
        self.assertNotIsInstance(
            raised.exception, comfyui.ComfyUIUnconfirmedTaskError
        )
        self.assertEqual(raised.exception.task_id, "job-failed")
        self.assertIn("node crashed", str(raised.exception))

    def test_run_timeout_raises_unconfirmed_with_job_id(self):
        config.app["comfyui_run_timeout"] = 0
        client = MagicMock()
        client.submit_job.return_value = "job-slow"
        client.get_job.return_value = {"status": "running", "outputs": []}
        with patch.object(comfyui, "_client", return_value=client):
            with self.assertRaises(comfyui.ComfyUIUnconfirmedTaskError) as raised:
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
        self.assertEqual(raised.exception.task_id, "job-slow")

    def test_output_is_matched_by_node_id_not_by_type_field(self):
        # A real ComfyUI Cloud run against this exact vendored workflow
        # returned type="image" for the SaveVideo node's output (a
        # video/*.mp4 file) -- the API's "type" field is not reliable here,
        # so the output must be matched by the known SaveVideo node id.
        client = MagicMock()
        client.submit_job.return_value = "job-real-shape"
        client.get_job.return_value = {
            "status": "succeeded",
            "outputs": [
                {
                    "id": "asset-1",
                    "node_id": "2",
                    "name": "video/MiniMax_H3_00001_.mp4",
                    "type": "image",
                    "content_type": "",
                    "url": "https://cloud.comfy.org/api/v2/assets/asset-1/content",
                }
            ],
        }
        with patch.object(comfyui, "_client", return_value=client):
            items = comfyui.generate_videos(
                search_term="term",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )
        self.assertEqual(
            items[0].url, "https://cloud.comfy.org/api/v2/assets/asset-1/content"
        )

    def test_succeeded_job_without_video_output_is_a_protocol_error(self):
        client = MagicMock()
        client.submit_job.return_value = "job-no-video"
        client.get_job.return_value = {"status": "succeeded", "outputs": []}
        with patch.object(comfyui, "_client", return_value=client):
            with self.assertRaises(comfyui.ComfyUIError) as raised:
                comfyui.generate_videos(
                    search_term="term",
                    minimum_duration=5,
                    video_aspect=VideoAspect.portrait,
                )
        self.assertEqual(raised.exception.task_id, "job-no-video")
