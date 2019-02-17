from django.shortcuts import render
from django.http import HttpResponse

import collections
import detect.classifier as clr
import numpy as np

# Create your views here.
def test(request):
    test_case = request.POST
    test_case = collections.OrderedDict(sorted(test_case.items()))
    tmp = []
    for k, v in test_case.items():
        print(k)
        if k == 'csrfmiddlewaretoken':
            continue
        #print(v)
        try:
            tmp.append(float(v))
        except:
            tmp.append(int(v))
    test_case = np.asarray([tmp])
    #test_case = np.asarray([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    #test_case = np.asarray([[58, 1, 0, 100, 234, 0, 1, 156, 0, 0.1, 2, 1, 3]])
    print(test_case)
    diag = clr.classifier(test_case, 'detect/model/model_new.sav')
    res = int(diag.predict())
    print(res)

    if res == 1:
        return render(request, 'sorry.html')
    elif res == 0:
        return render(request, 'congratulations.html')

    return HttpResponse("error")


# login
def login(request):

    return render(request, 'login.html')


# home page
def home(request):

    return render(request, 'welcom.html')

# input page
def input(request):

    return render(request, 'test.html')


def contact(request):

    return render(request, 'contact.html')


def about(request):

    return render(request, 'about.html')
