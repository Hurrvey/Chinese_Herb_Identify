from django.db import models
# Create your models here.


class Data(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()

    class Meta:
        app_label = 'image_recognizer'