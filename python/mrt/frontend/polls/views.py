from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse

def index(request):
    latest_question_list = Question.objects.order_by("-pub_data")[:5]
    output = ", ".join([q.question_text for q in latest_question_list])
    return HttpResponse(output)

def detail(request, question_id):
    return HttpResponse(
        "You're looking at question {}.".format(question_id))

def results(request, question_id):
    return HttpResponse(
        "You're looking at the results of question {}.".format(
            question_id))

def vote(request, question_id):
    return HttpResponse("You're voting on question: {}".format(question_id))
