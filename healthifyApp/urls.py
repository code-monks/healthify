from django.shortcuts import *
from django.urls import path
from .views import *


urlpatterns = [
    path('songs/', ListSongsView.as_view(), name="songs-all"),
	path('users/', ListUsersView.as_view(), name="users-all"),
	# path('userall/',userList , name="userall")
]