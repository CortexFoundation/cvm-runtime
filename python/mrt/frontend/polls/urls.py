from django.urls import path

from . import views

urlpatterns = [
    path("prepare", views.views_prepare, name="prepare"),
]
