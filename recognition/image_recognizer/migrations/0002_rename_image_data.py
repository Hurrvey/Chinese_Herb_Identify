# Generated by Django 4.2 on 2024-01-08 01:39

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('image_recognizer', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Image',
            new_name='Data',
        ),
    ]