from rest_framework import serializers
from .models import Songs, User


class SongsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Songs
        fields = ("title", "artist")

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("url", "path", "timestamp", "heart_beat")