"""firstproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from myapp.views import sayhello,hello2,hello3,hello4

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', sayhello),
    url(r'^hello/$', sayhello),
    url(r'^hello2/(\w+)/$', hello2),
    url(r'^hello3/(\w+)/$', hello3),
    url(r'^hello4/(\w+)/$', hello4),
]
