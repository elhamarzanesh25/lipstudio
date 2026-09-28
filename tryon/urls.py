from django.urls import path

from . import views

app_name = "tryon"
urlpatterns = [
    path("", views.index, name="index"),
    path("live/", views.live, name="live"),
    path("api/upload/", views.api_upload, name="api_upload"),
    path("api/apply/", views.api_apply, name="api_apply"),
]
