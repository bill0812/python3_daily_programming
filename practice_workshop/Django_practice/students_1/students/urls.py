from django.conf.urls import url
from django.contrib import admin
from studentsapp import views

urlpatterns = [
	url(r'^admin/', admin.site.urls),
	url(r'^listone/$', views.listone),
	url(r'^listall/$', views.listall),
	url(r'^$', views.index),
	url(r'^index/$', views.index),
	
	url(r'^post/$', views.post), # POST 傳送表單
	url(r'^post1/$', views.post1), #資料新增，資料不驗證
	url(r'^post2/$', views.post2), #資料新增，資料作驗證

	url(r'^delete/(\d+)/$', views.delete),
	
	url(r'^edit/(\d+)/$', views.edit), # 由 瀏覽器 開啟
	url(r'^edit/(\d+)/(\w+)$', views.edit), # 由 edit.html 按 送出 鈕

	url(r'^edit2/(\d+)/(\w+)$', views.edit2),
	url(r'^postform/$', views.postform), # 表單驗證
]