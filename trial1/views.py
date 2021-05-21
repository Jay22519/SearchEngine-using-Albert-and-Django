from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
from trial1.models import Data,Keyword
import pandas as pd 
import numpy as np 
import os 
import json
from tqdm import tqdm
import wikipedia
from django.db.models import Q # new
from bs4 import BeautifulSoup 
import requests 
import torch
from transformers import AlbertTokenizer, AlbertForQuestionAnswering

import re
# Create your views here.

"""
Variables to be used for tokenizer and Model in Albert model 
And counter also to store the total number of pages we have 
"""

"""
List of all utility functions 
1) secure 
2) Answer 
"""
tokenizer = AlbertTokenizer.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')
model = AlbertForQuestionAnswering.from_pretrained('D:/albert') 
counter = 0
pattern = re.compile(r"== [0-9a-zA-Z]* ==")

def secure(a) :
    b = str(a) 
    if(pd.isna(b)) :
        return "not known"
    return b.lower()

def Answer(question, text):
        #print("Tokenizer is ...............",tokenizer,"\n") 
        #print("Mode is ....................",model,"\n\n")
        input_dict = tokenizer.encode_plus(question, text, return_tensors='pt')
        input_ids = input_dict["input_ids"].tolist()
        start_scores, end_scores = model(**input_dict,return_dict=False)
        
        ##Return_dict is important 
        start = torch.argmax(start_scores)
        end = torch.argmax(end_scores)
        
        #print("Scores are " ,start_scores ," ", end_scores,"\n")
        
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        answer = ''.join(all_tokens[start: end + 1]).replace('▁', ' ').strip()
        answer = answer.replace('[SEP]', '')
        return answer if answer != '[CLS]' and len(answer) != 0 else ''



def getData(request) :
    
    global tokenizer
    tokenizer = AlbertTokenizer.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')

    global model
    model = AlbertForQuestionAnswering.from_pretrained('D:/albert') 

    print("..............................................................................")
    print("Done with token and model","\n\n\n\n\n")

    data = pd.read_csv("D:/India_1.csv")  
    data_dict = {}  
    global counter 
    counter = 0
    data_keyword = []
    for i in range(1252) :
        data_dict['Title'] = secure(data['Title'][i])
        data_dict['Summary'] = secure(data['Summary'][i]) 
        data_dict['Categories'] = secure(data['Categories'][i])
        data_dict['Content'] = secure(data['Content'][i])
        data_dict['Related_links'] = secure(data['Related_links'][i]) 

        data_keyword.append(data_dict['Title']) 
        categ = (secure(data['Categories'][i]))
        for j in range(len(categ)) :
            data_keyword.append(categ[j])

        data_created = Data.create(**data_dict)  
        counter += 1 


        print("Done with ............",counter,"iterations in Data\n")


    data_keyword = set(data_keyword) 
    data_keyword = list(data_keyword)
    len_key = len(data_keyword) 

    for i in range(len_key) :
        data_dict['word'] = data_keyword[i]
        print(data_dict['word'])
        data_created = Keyword.create(**data_dict)

        print("Done with ............",i,"iterations in Keyword\n")

    data = {
        "datas" : Data.objects.all() ,
        'counter' : counter 
    }

    return render(request, "home.html", data)


def home(request) :

    data = {
        "datas" : Data.objects.all() ,
        'counter' : counter 
    }

    return render(request , "home.html",data)

def content(request) :
    data = {
        "datas" : Data.objects.all() ,
        'counter' : counter 
    }

    return render(request , "listing.html",data)

def contentTopic(request) :

    req = request.GET['query'] 
    answer = Data.objects.filter(
            Q(Title__icontains=req) | Q(Summary__icontains=req) | Q(Content__icontains = req) | Q(Categories__icontains = req)
        )

    data = {
        "datas" : answer,
        'counter' : counter 
    }

    return render(request , "listing.html",data) 


def search(request) :
    """
    This veiw is used for search purpose also and also for reading more about an article 
    So when we click on read more about article we are redirected to this view with req as page.Title so we'll 
    definitely be redirected to about.html 
    """

    req = request.GET['query'] 
    print("\n\n\n\n\n\n")
    print("Request query got is ....................",req,"\n\n")

    """
    1) First search through Title and if we get exact title then print it otherwise print first 10 results
    2) Then search through summary and print first 10 results 
    3) Still we haven't find anything then search through content and print first 10 results 
    4) And finally search using categories and print first 15 results 
    """

    answer = Data.objects.filter(
        Q(Title__icontains = req)
    )

    len_ans = len(answer)
    
    print("Length of Data got is .....",len_ans,"\n\n")
    got_exact =  0 #This is to find that whether an exact result is found or not  
    for i in range(len_ans) :
        if(answer[i].Title == str(req)):
            got_exact = 1 
            break 

    if(not got_exact) :
        if(len_ans > 10) :
            answer = answer[:10] 
    
    if(got_exact) :
        data = {
        "datas" : answer
        }
        return render(request ,"about.html",data)
    elif(len_ans) :
        data = {
        "datas" : answer
        }
        return render(request ,"listing.html",data)

    """ So we didn't get result using name so lets go for summary and do the same thing """ 

    answer = Data.objects.filter(
        Q(Summary__icontains = req)
    )

    len_ans = len(answer)
    if(len_ans > 10) :
        answer = answer[:10] 
    
    if(len_ans) :
        data = {
        "datas" : answer
        }
        return render(request ,"listing.html",data)

    """ So we didn't get result using name so lets go for Content and do the same thing """ 


    answer = Data.objects.filter(
        Q(Content__icontains = req)
    )

    len_ans = len(answer)
    if(len_ans > 10) :
        answer = answer[:10] 
    
    if(len_ans) :
        data = {
        "datas" : answer
        }
        return render(request ,"listing.html",data)


    """ So we didn't get result using name so lets go for Categories and do the same thing """ 



    answer = Data.objects.filter(
        Q(Categories__icontains = req)
    )

    len_ans = len(answer)
    if(len_ans > 10) :
        answer = answer[:10] 

    data = {
    "datas" : answer
    }
    return render(request ,"listing.html",data)





def query(request) :

    q = str(request.GET['query'])
    q = q.lower()
    """
    Now here solving query can be broken into 2 parts .
    1) Finding all relevant articles 
    2) Extracting important info from that article (Using bert model) 

    First part is same as search view except for the fact that we won't return article immediately after getting 
    answer object .

    First from the query we'll remove stop first and lemmetized it and then perform search type of view to get all relevant articles 
    and then finally do BERT(Albert) operations 

    """

    
    #Search for keywords only when the query isn't in any of the articles 

    keywords = [] 
    key_all = Keyword.objects.all()
    len_key = len(key_all) 
    print("Length of keyword is ............",len_key,"\n\n\n\n\n")
    for i in range(len_key) :
        print(key_all[i].keyword,"is a keyword")
        if(key_all[i].keyword in q ) :
            keywords.append(key_all[i].keyword) 

    """
    #After getting all the keyword I'll remove all the keyword with length less than 3 as they are of no use 
    """

    len_final_keyword = len(keywords) 
    final_keywords = [] 
    for i in range(len_final_keyword) :
        if(len(keywords[i]) > 2) :
            print("One of the final keyword is .......",keywords[i],"\n")
            final_keywords.append(keywords[i]) 


    ### Finding the largest keyword
    lar_index = 0 
    lar = 0
    print("LENGTH OF KEYWORD ............................",len(final_keywords),"\n")
    #print(len(final_keywords[0])," lENGTH................................................")
    for i in range(len(final_keywords)) :
        if(len(final_keywords[i]) > lar) :
            lar = len(final_keywords[i])
            lar_index = i 
            print("Now large is ..........",lar_index)

    if(len(final_keywords) > 0):
        final = final_keywords[lar_index]
        final_keywords = []
        final_keywords.append(final)

    ###That is no keyword found in query in query so we'll search manually through all the pages looking for this query
    if(len(final_keywords) == 0) : 
        final_keywords.append(q) 

    print("KeywordS got from query is .........",final_keywords , "\n\n\n")

    """
    Now left is to find relevant articles from those keywords 
    """

    """
    1) First search through Title and if we get exact title then print it otherwise print first 10 results
    2) Then search through summary and print first 10 results 
    3) Still we haven't find anything then search through content and print first 10 results 
    4) And finally search using categories and print first 15 results 
    """

    """
    And the above process will be iterated for all the keywords 
    """

    answers = []  ##### Will contains all the final Data Objects to be sent to bert 
    len_final_key = len(final_keywords) 
    for i in range(len_final_key) :
        answer = Data.objects.filter(
            Q(Title__icontains = final_keywords[i]) 
        )

        title = 0 
        summary = 0 
        content = 0 
        exact_got = 0 
        category = 0 
        if(len(answer) > 0) :
            title = 1 
        for j in range(len(answer)) :
            if(answer[j].Title == final_keywords[i]):
                exact_got = 1 
                break 

        if(exact_got == 1) :
            answer = answer[:1] 
        elif(exact_got == 0 and title == 1) :
            if(len(answer) > 5):
                answer = answer[:5]
 
        if(title == 0) :   #### That is no result found in title so searching in summary 
            answer = Data.objects.filter(
                Q(Summary__icontains = final_keywords[i])
            ) 

            if(len(answer) > 0) :
                summary = 1 
            if(len(answer) > 5) :
                answer = answer[:5]
            
        if(summary == 0 and title == 0) :  ### So no result found in title and summary so searching in content
            answer = Data.objects.filter(
                Q(Content__icontains = final_keywords[i])
            ) 

            if(len(answer) > 0) :
                content = 1
            if(len(answer) > 5) :
                answer = answer[:5]

        if(content == 0 and summary == 0 and title == 0) : ### So no result found in title and summary and content
                                                            #### so searching in categories 
            answer = Data.objects.filter(
                Q(Categories__icontains = final_keywords[i])
            ) 

            if(len(answer) > 0) :
                category = 1
            if(len(answer) > 5) :
                answer = answer[:5]

        



        answers += answer 
        print("\n\n\n Answer got for this is .................",answer,"\n\n\n\n\n\n")

    len_answers = len(answers) 
    Query_answer = "" 
    for i in range(len_answers) :
        text = answers[i].Content
        text = pattern.sub("",text)
        all_text = text.split(".")
        for j in range(len(all_text)) :
            this_answer = Answer(q,all_text[j]) 
            print("Answer got now for ",i,"th page and ",j,"th is ..................",this_answer , "\n\n\n\n")
            
            Query_answer += this_answer 
            Query_answer += ". "

        """
        for j in range(len(answers[i].Content)//512) :
            text1 = text[512*j : 512*(j+1)] 

            this_answer = Answer(q,text1) 
            print("Answer got now for ",i,"th page and ",j,"th is ..................",this_answer , "\n\n\n\n")
            
            Query_answer += this_answer 
            Query_answer += ". "

        """ 



    print("Final I get is ..............",Query_answer,"\n\n\n")
    qa = ""
    for i in range(len(Query_answer)//512) :
        text1 = Query_answer[512*j : 512*(j+1)] 

        this_answer = Answer(q,text1) 
        print("Answer got now is from Final is   ..................",this_answer , "\n\n\n\n")
        
        qa += this_answer



    
    data = {
        'datas' : answers ,
        'query_answer' : Query_answer,
        'query_short_answer' : qa

    }

    return render(request , "answer.html",data)

    

        

