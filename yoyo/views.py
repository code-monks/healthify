from rest_framework import status
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from django.http import Http404
from .models import Animal
from .serializers import AnimalSerializer
import requests

# Video Processing imports
from yoyo.vpg import main


# Indent Cmd + Alt + L

def download(url, path = '/home/nikhil/Github/healthify/yoyo/media/new2.mov', chunk=2048):
    req = requests.get(url, stream=True)
    if req.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in req.iter_content(chunk):
                f.write(chunk)
            f.close()
        return path
    raise Exception('Given url is return status code:{}'.format(req.status_code))

class HeartRateView(APIView):
    def get(self, request, format=None):
        path = download('https://res.cloudinary.com/drbmjxpbv/video/upload/v1544449847/rcngdj1e5vi0kitrnd6o.mp4')
        ans = main(path)
        return Response(ans)
class AnimalTask(APIView):
    
    def get(self, request, format=None):
        animals = Animal.objects.all()
        serializer = AnimalSerializer(animals, many=True)
        path = download('https://res.cloudinary.com/drbmjxpbv/video/upload/w_150,c_scale/v1544440511/ny8ygzkgrhowmgex8muo.mov')
        return Response(serializer.data)

    def put(self, request, format=None):
        name = request.GET.get('name')
        animal = Animal.objects.get(name=name)
        if animal:
            serializer = AnimalSerializer(animal, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_404_NOT_FOUND)

    def post(self, request, format=None):
        serializer = AnimalSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, format=None):
        name = request.GET.get('name')
        animal = Animal.objects.get(name=name)
        if animal:
            animal.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(status=status.HTTP_400_BAD_REQUEST)
