from rest_framework import generics
from .models import Songs, User
from rest_framework.views import APIView
from django.http import HttpResponse, JsonResponse
# from rest_framework.decorators import api_view
from rest_framework.response import Response

from rest_framework import status
from .serializers import SongsSerializer, UserSerializer
# from django.core import serializers
# import requests
import json


class ListSongsView(generics.ListAPIView):
    """
    Provides a get method handler.
    """
    queryset = Songs.objects.all()
    serializer_class = SongsSerializer

def download(url, path = '/home/nikhil/Github/healthify/healthifyApp/media/new2.mov', chunk=2048):
    req = requests.get(url, stream=True)
    if req.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in req.iter_content(chunk):
                f.write(chunk)
            f.close()
        return path
    raise Exception('Given url is return status code:{}'.format(req.status_code))

# @api_view(['GET'])
# def userList():
#     content = {
#             'status': 'Not Found'
#         }
#     return Response(content, status=status.HTTP_404_NOT_FOUND)
#     # try:
#         users = User.objects.all()
#         user_serialized = serializers.serialize('json', [users])
#         return JsonResponse(user_serialized)
#     except User.DoesNotExist:
#         content = {
#             'status': 'Not Found'
#         }
#         return Response(content, status=status.HTTP_404_NOT_FOUND)

# @api_view()
# def home(request):
#     return Response({"message": "Hello, world!"})
    # try:
    #     users = User.objects.all()
    #     user_serialized = serializers.serialize('json', [users])
    #     path = download('https://res.cloudinary.com/drbmjxpbv/video/upload/w_150,c_scale/v1544440511/ny8ygzkgrhowmgex8muo.mov')
    #     print(path)
    #     return JsonResponse(user_serialized)
    # except User.DoesNotExist:
    #     content = {
    #         'status': 'Not Found'
    #     }
    #     return Response(content, status=status.HTTP_404_NOT_FOUND)

class ListUsersView(generics.ListAPIView):
    """
    Provides a get method handler.
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer

class NewView(APIView):
    """
    Provides a get method handler.
    """
    def get(self, request, format=None):
        """
        Return a list of all users.
        """
        usernames = [user.username for user in User.objects.all()]
        return Response(usernames)

