from django.conf.urls import url
from django.contrib import admin
from studentsapp import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
	url(r'^listone/$', views.listone),
	url(r'^listall/$', views.listall),
    url(r'^$', views.index),
    url(r'^index/$', views.index),
]
