from django.urls import path
from . import views

urlpatterns = [
    path("", views.index),
    path("load_model",views.load_model)
]
