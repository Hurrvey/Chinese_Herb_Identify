"""
URL configuration for recognition project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
import sys
sys.path.append("D:/PycharmProjects/pythonProject/Traditional Chinese Medicine Identification/recognition/image_recognizer/")
import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    path('result/', views.result, name='result_no_arg'),
    path('result/<str:result>/', views.result, name='result')
]

