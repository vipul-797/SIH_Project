from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('history', views.history, name='history'),
    path('chatbot', views.chatbot, name='chatbot'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('newchat', views.newchat, name='newchat'),
    path('speak_text', views.speak_text, name='speak_text'),
    path('get_response', views.get_response, name='get_response'),
]