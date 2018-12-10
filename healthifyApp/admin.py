from django.contrib import admin

# Register your models here.
from .models import Songs, User

admin.site.register(Songs)
admin.site.register(User)