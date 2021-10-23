from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def redirect_to_year(request):
    year = 2006
    return HttpResponseRedirect(reverse('news-year-archieve'), args=(year,))
