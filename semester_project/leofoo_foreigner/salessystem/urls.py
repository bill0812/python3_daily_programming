"""salessystem URL Configuration

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
from salesapp import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^login/$', views.login),
    url(r'^vertify/$', views.vertify),
    url(r'^vertify_account/$', views.vertify_account),
    url(r'^register/$', views.register),
    url(r'^register_member/$', views.register_member),
    url(r'^register_address/$', views.register_address),
    url(r'^main/$', views.main_page),
    url(r'^reply/$', views.reply),
    url(r'^reply_status/$', views.reply_status),
    url(r'^logout/$', views.logout),
    url(r'^customer_page/$', views.customer_page),
    url(r'^customer_enter/$',views.customer_enter),
    url(r'^delivery/$',views.delivery),
    url(r'^cart/$',views.cart),
    url(r'^cart_upload/$',views.cart_upload),
    url(r'^make_order/$',views.make_order),
    url(r'^upload_normal/$',views.upload_normal),
    url(r'^order/$',views.order),
    url(r'^personal/$',views.personal),
    url(r'^edit_data/$',views.edit_data),
    url(r'^admin_page/$',views.admin),
    url(r'^admin_page_order/$',views.admin_page_order),
    url(r'^admin_page_structure/$',views.admin_page_structure),
    url(r'^product_structure/$',views.product_structure),
    url(r'^compose_product/$',views.compose_product),
    url(r'^sned_inventory/$',views.sned_inventory),
    url(r'^adminprocess/$',views.adminprocess),
    url(r'^get_product_order/$',views.get_product_order),
    url(r'^get_veg_order/$',views.get_veg_order),
    url(r'^show_rfm/$',views.show_rfm)
]
