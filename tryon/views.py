import base64
import uuid

import cv2
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render as render_template
from django.views.decorators.http import require_GET, require_POST

from .forms import ApplyForm, UploadForm
from .services.lip_engine import FINISHES, NoFaceError, prepare_image, render

CACHE_PREFIX = "lip:"


def _data_uri(img_bgr, quality=92):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@require_GET
def index(request):
    return render_template(request, "tryon/index.html",
                           {"finishes": [(k, v[0], v[1]) for k, v in FINISHES.items()]})


@require_POST
def api_upload(request):
    """Step 1 - once per photo: detect face, build lip mask, keep it server-side."""
    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"errors": form.errors.get_json_data()}, status=400)
    try:
        prep = prepare_image(form.decoded)
    except NoFaceError:
        return JsonResponse({"errors": {"image": [{"message": "No face detected. Use a clear, front-facing photo."}]}}, status=422)
    token = uuid.uuid4().hex
    cache.set(CACHE_PREFIX + token, prep)
    return JsonResponse({"token": token, "original": _data_uri(prep.img)})


@require_POST
def api_apply(request):
    """Step 2 - as many times as you like: token + colour + finish -> result."""
    form = ApplyForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"errors": form.errors.get_json_data()}, status=400)
    prep = cache.get(CACHE_PREFIX + form.cleaned_data["token"])
    if prep is None:
        return JsonResponse({"errors": {"token": [{"message": "Session expired, please upload the photo again."}]}}, status=410)
    out = render(prep, form.cleaned_data["color"], form.cleaned_data["finish"])
    return JsonResponse({"result": _data_uri(out)})


@require_GET
def live(request):
    return render_template(request, "tryon/live.html",
                           {"finishes": [(k, v[0], v[1]) for k, v in FINISHES.items()]})
