from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(Task)
admin.site.register(DeviceTemplate)
admin.site.register(ProtectTemplate)
admin.site.register(TaskImage)