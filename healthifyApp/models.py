from django.db import models

# Create your models here.

class Songs(models.Model):
    # song title
    title = models.CharField(max_length=255, null=False)
    # name of artist or group/band
    artist = models.CharField(max_length=255, null=False)
	# video URL
    url = models.CharField(max_length=255, null=False)

    def __str__(self):
        return "{} - {}".format(self.title, self.artist)

class User(models.Model):
    url = models.CharField(max_length=255, null=False)
    path = models.CharField(max_length=255, null=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    heart_beat = models.DecimalField(default=0, max_digits=9, decimal_places=2)
    def __str__(self):
        return "{} - {}".format(self.timestamp, self.heart_beat)