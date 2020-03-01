from django.conf.urls import url
from django.contrib import admin
from passdataapp.views import dice,dice2,dice3,show,filter

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^dice/$', dice),
    url(r'^dice2/$', dice2),
    url(r'^dice3/$', dice3),
    url(r'^show/$', show),
    url(r'^filter/$', filter)
]
