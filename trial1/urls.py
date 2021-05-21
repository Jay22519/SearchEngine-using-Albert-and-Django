from django.urls import path
from trial1 import views

""""
1) load will use getData to load the data ( This is to be done only 1 time)
2) home will redirect me to home page 
3) content will redirect me to the page showing summary of all pages 
4) content_topic will redirect to pages of a given topic/keyword
5) Search will redirect to a searched page or if many pages are there then it will redirect to same as content_topic page
This search function will be in home page
6) about will redirect to a page with complete info about the page we searched for 
7) query will redirect me a query page with an answer and list of all referenced page used 
This query functionality will also be in home page 


About will also use search views as its more efficient to get result. More is written about it in the search view .
"""


urlpatterns = [
    path('load/',views.getData , name = 'getData') ,
    path('home/',views.home,name = 'home') ,
    path('content/',views.content,name = 'content'),
    path('contentTopic/',views.contentTopic , name = 'contentTopic'),
    path('search/',views.search , name = 'search'),
    path('about/',views.search , name = 'about'),
    path('query/',views.query , name = 'query') , 
]
