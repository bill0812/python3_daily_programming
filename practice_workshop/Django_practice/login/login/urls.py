from django.conf.urls import url
from django.contrib import admin
from loginapp import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),

    url(r'^$', views.index),
    url(r'^index/$', views.index),

    url(r'^login/$', views.login),
    url(r'^logout/$', views.logout),	
	
	url(r'^adduser/$', views.adduser),	
]