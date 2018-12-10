from django.shortcuts import *
from django.urls import path
from .views import *


urlpatterns = [
    path('all/', HeartRateView.as_view(), name="songs-all")
]