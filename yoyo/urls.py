from django.shortcuts import *
from django.urls import path
from .views import *


urlpatterns = [
    path('all/', AnimalTask.as_view(), name="songs-all")
]