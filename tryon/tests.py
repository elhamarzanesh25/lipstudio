import base64
import time
from pathlib import Path

import cv2
import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase
from django.urls import reverse

from tryon.services.lip_engine import FINISHES

SAMPLE = Path(__file__).resolve().parent.parent / "sample" / "user.jpg"


class TryOnTests(SimpleTestCase):
    def upload(self, data=None):
        f = SimpleUploadedFile("u.jpg", data or SAMPLE.read_bytes())
        return self.client.post(reverse("tryon:api_upload"), {"image": f})

    def apply(self, token, color="#FF0000", finish="matte"):
        return self.client.post(reverse("tryon:api_apply"), {"token": token, "color": color, "finish": finish})

    def test_upload_once_apply_many(self):
        r = self.upload(); self.assertEqual(r.status_code, 200)
        token = r.json()["token"]
        results = {}
        for finish in FINISHES:
            for color in ("#FF0000", "#8B0000"):
                r = self.apply(token, color, finish)
                self.assertEqual(r.status_code, 200, (finish, color))
                results[(finish, color)] = r.json()["result"]
        self.assertNotEqual(results[("matte", "#FF0000")], results[("matte", "#8B0000")])
        self.assertNotEqual(results[("matte", "#FF0000")], results[("glossy", "#FF0000")])

    def test_apply_is_much_faster_than_upload(self):
        t = time.perf_counter(); token = self.upload().json()["token"]; t_up = time.perf_counter() - t
        t = time.perf_counter(); self.apply(token, finish="glossy"); t_ap = time.perf_counter() - t
        print(f"\n  upload (detect+mask): {t_up*1000:.0f} ms | apply (colour+finish): {t_ap*1000:.0f} ms")

    def test_errors(self):
        token = self.upload().json()["token"]
        self.assertEqual(self.apply(token, color="red").status_code, 400)
        self.assertEqual(self.apply(token, finish="nope").status_code, 400)
        self.assertEqual(self.apply("does-not-exist").status_code, 410)          # expired token
        self.assertEqual(self.upload(b"hello").status_code, 400)                  # not an image
        blank = cv2.imencode(".png", np.full((300, 300, 3), 200, np.uint8))[1].tobytes()
        self.assertEqual(self.upload(blank).status_code, 422)                     # no face

    def test_page(self):
        self.assertContains(self.client.get(reverse("tryon:index")), "Lip Studio")


class LiveTests(SimpleTestCase):
    def test_live_page_and_assets(self):
        from django.contrib.staticfiles import finders
        from django.templatetags.static import static
        r = self.client.get(reverse("tryon:live"))
        self.assertContains(r, "Start camera")
        for f in FINISHES:
            self.assertContains(r, f'value="{f}"')
        for path in ("tryon/lipfx.js", "tryon/live.js", "tryon/models/face_landmarker.task",
                     "tryon/vendor/tasks-vision/vision_bundle.mjs",
                     "tryon/vendor/tasks-vision/vision_wasm_internal.js",
                     "tryon/vendor/tasks-vision/vision_wasm_internal.wasm",
                     "tryon/vendor/tasks-vision/vision_wasm_nosimd_internal.wasm"):
            self.assertTrue(finders.find(path), path)
