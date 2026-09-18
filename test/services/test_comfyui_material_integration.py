import unittest
from unittest.mock import patch

from app.config import config
from app.models.schema import MaterialInfo
from app.services import comfyui, material


class TestComfyUIMaterialIntegration(unittest.TestCase):
    def setUp(self):
        self.original_app = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app)

    @staticmethod
    def _item(term: str, url: str) -> MaterialInfo:
        return MaterialInfo(
            provider="comfyui",
            url=url,
            duration=5,
            source_info={
                "provider": "comfyui",
                "search_term": term,
                "asset_id": f"job-{term}",
            },
        )

    def test_on_demand_generation_stops_when_duration_is_covered(self):
        config.app["comfyui_api_key"] = "comfy-test-key"
        with (
            patch.object(
                comfyui,
                "generate_videos",
                side_effect=[
                    [self._item("one", "https://cdn.example.com/one.mp4")],
                    [self._item("two", "https://cdn.example.com/two.mp4")],
                ],
            ) as generate,
            patch.object(
                material,
                "save_video",
                side_effect=["/tmp/one.mp4", "/tmp/two.mp4"],
            ) as save_video,
            patch.object(material, "_persist_material_sources") as persist,
        ):
            result = material.download_videos(
                task_id="comfyui-materials",
                search_terms=["one", "two", "three"],
                source="comfyui",
                audio_duration=10,
                max_clip_duration=5,
            )

        self.assertEqual(result, ["/tmp/one.mp4", "/tmp/two.mp4"])
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(persist.call_args.args[0], "comfyui-materials")
        self.assertEqual(len(persist.call_args.args[1]), 2)
        # ComfyUI Cloud asset URLs need a Bearer token on download, unlike
        # every other provider's pre-signed download links.
        sent_headers = save_video.call_args.kwargs["headers"]
        self.assertEqual(sent_headers["Authorization"], "Bearer comfy-test-key")

    def test_unconfirmed_task_stops_later_paid_submissions(self):
        error = comfyui.ComfyUIUnconfirmedTaskError(
            "remote state unknown", task_id="job-two"
        )
        with (
            patch.object(
                comfyui,
                "generate_videos",
                side_effect=[
                    [self._item("one", "https://cdn.example.com/one.mp4")],
                    error,
                ],
            ) as generate,
            patch.object(material, "save_video", return_value="/tmp/one.mp4"),
            patch.object(material, "_persist_material_sources") as persist,
        ):
            with self.assertRaises(comfyui.ComfyUIUnconfirmedTaskError) as raised:
                material.download_videos(
                    task_id="comfyui-unconfirmed",
                    search_terms=["one", "two"],
                    source="comfyui",
                    audio_duration=15,
                    max_clip_duration=5,
                )

        self.assertEqual(raised.exception.task_id, "job-two")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(persist.call_args.args[1]), 1)

    def test_download_failure_raises_comfyui_download_error_with_job_id(self):
        with (
            patch.object(
                comfyui,
                "generate_videos",
                return_value=[self._item("one", "https://cdn.example.com/one.mp4")],
            ),
            patch.object(material, "save_video", return_value=""),
            patch.object(material, "_persist_material_sources"),
        ):
            with self.assertRaises(comfyui.ComfyUIDownloadError) as raised:
                material.download_videos(
                    task_id="comfyui-download-failure",
                    search_terms=["one"],
                    source="comfyui",
                    audio_duration=5,
                    max_clip_duration=5,
                )

        self.assertEqual(raised.exception.task_id, "job-one")
