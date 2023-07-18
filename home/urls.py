from home import views
from django.urls import re_path as url

urlpatterns =[
     url('^$',views.homepage),
     url('/predict',views.predict),
     url('/about',views.about),
     url('/firstaid',views.firstaid),
     url('/professionaltreatment',views.pt),
     
]
