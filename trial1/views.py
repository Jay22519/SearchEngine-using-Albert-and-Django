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


import gensim
from gensim.models import Word2Vec
from pyemd import emd


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

model_wmd = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin.gz', binary=True)  
counter = 0
pattern = re.compile(r"== [0-9a-zA-Z]* ==")



"""
Now we'll convert data_categories into proper list to be used 
"""

def to_list(s) :
    s = s.replace("[","")
    s = s.replace("]","")
    s = s.replace("'","")
    s = s.replace("'","")
    
    return list(s.split(","))



def secure(a) :
    b = str(a) 
    if(pd.isna(b)) :
        return "not known"
    return b.lower()



def find_relevant_page(query) :
    query = query.lower()
    keyword_list = [0 for I in range(10000)]
    counterL = 0 
    for i in range(counter) :
        if keyword[i] in query :
            #print(keyword[i])
            keyword_list[counterL] = keyword[i]
            counterL += 1
    return list(set(keyword_list[:counterL]))

def is_title_imp(query) :
    query = query.lower()
    keyword_list = [0 for I in range(10000)]
    counterL = 0 
    for i in range(10088) :
        if title[i].lower() in query :
            #print(keyword[i])
            for j in range(len(data_categories[i])) :
                if 'births' in data_categories[i][j].lower() :
                    keyword_list[counterL] = title[i]
                    counterL += 1
                    print("Added .......",title[i],"\n")
                    break 
                    
                if 'deaths' in data_categories[i][j].lower() :
                    keyword_list[counterL] = title[i]
                    counterL += 1
                    print("Added .......",title[i],"\n")
                    
    
    return list(set(keyword_list[:counterL]))



def find_keyword(query) :
    num = 1  #That is title found
    List = is_title_imp(query) 
    if( not len(List)) :
        print("No title found .........................\n\n")
        List = find_relevant_page(query)
        num = 0 
    #### Removing the keyword "Who" from it 
    if "who" in List :
        List.remove("who")
    return List ,num
            


### Now it may happen that List from find_keyword contains some subset of another keyword of list , so we have to remove that .
"""
For example if List contain = ['Cheif Minister' , 'Minister' ,  'Chief Minister of Chhattishgarh'] , so it should be known to 
the system that user is searching for 'Chief Minister of Chhattishgarh' and to implement this it will just contain the largest 
set of all the intersecting keywords 
"""
def unique(keywords,num):
    keys = []
    for s in keywords:
        if not any([s in r for r in keywords if s != r]):
            keys.append(s)
    
    return keys,num



"""
Here ->
1) Case 0 is when we had found a title in keyword 
2) Case 1 is when length of keyword is just 1 and if there is any title containing that match word 
3) Case 1 is searching for all the data_categories 

"""



def content_find(keywords,num) :
    if(num == 1) :
        print("Case 0\n")
        contentZero = [0 for i in range(10088)]
        contentLent = 0 
        for i in range(10088) :
            if keywords[0].lower() == title[i].lower() :
                contentZero[contentLent] = content[i].lower() 
                contentLent += 1
        if(contentLent) :
            return contentZero[:contentLent] , 1
    
    contents = {}
    content_final = [0 for i in range(1000)]
    final_counter =  0 
    
    
    list_of_title = [0 for i in range(100088)] 
    counterL = 0 
    done = 0 # Done is 1 if we have got titles matching keywords 
    ## First looking for any title matching [But only if number of keywords is 1]
    if(len(keywords) == 1) :
        print("Case 1")
        for i in range(10088) :
            if keywords[0].lower() in title[i].lower() :
                print(title[i].lower()," and index is ",i,"\n")
                done = 1
                contents[get_cosine(text_to_vector(keywords[0]),text_to_vector(title[i]))] = (content[i],title[i])
                    
        
        l = list(contents.items())
        l.sort(reverse=True)
        contents = dict(l)
       # print("Sorted and converted to dictionary again ")
        only5 = 0 
        for k in contents :
            only5+=1 
            content_final[final_counter] = contents[k][0]
            final_counter += 1
            #print("Content and title are  ..........",contents[k][0],"\n\n\n",contents[k][1])
            if(only5 == 5) :
                break 
        if(done == 1 ) :
            return content_final[:final_counter] ,1 
    
    
    ### The case we'll have to return list of answer is left 
    index_title = [0 for i in range(100088)] 
    
    if(done == 0) :
        print("Case 2 ")
        all_titles = [0 for i in range(10088)]
        for i in range(10088) : 
            done = 0 
            lenK = len(keywords)
            for k in range(lenK) :
                for z in range(len(data_categories[i])) :
                    if keywords[k] in data_categories[i][z].lower() :
                        done += 1 
                        break 
            if(done == lenK) :
                list_of_title[counterL] = title[i] 
                index_title[counterL] = i
                counterL += 1 
                print("Title added is ...................",title[i])
        if(counterL < 4) :
            print("Inside here")
            for i in range(4) :
                content_final[final_counter] = content[index_title[i]]
                print(content_final[final_counter])
                final_counter += 1 
            return content_final[:final_counter] , 1
            
            
        return list_of_title[:counterL],2





def all_preprocessing(query) :
    
    query = query.lower()
    
    #This sample keyword is to make sure that we aren't doing spelling correction if we found a title in query 
    sample_keyword , sample_num  = find_keyword(query.lower())
    if(sample_num == 0):
        ##Spelling correction 
        correct = correction(query) 
    else :
        print("Already corrected title")
    done = int(input()) 
    while(not done) :
        print("Input your query again .......................")
        query = input(query)
        #correct = correction(query)
        done = int(input())
    imp_keywords,num = find_keyword(query)
    imp_keywords,num = unique(imp_keywords,num)
    #### Removing the extra ' ' from front from all keywords 
    for i in range(len(imp_keywords)) :
        if(imp_keywords[i][0] == ' ') :
            #print(imp_keywords[i])
            imp_keywords[i] = imp_keywords[i][1:] 
    print("imp keyword is ................",imp_keywords)
    content_list , number = content_find(imp_keywords,num)
    
    return content_list , number 



def answer(question, text):
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
    return answer if answer != '[CLS]' and len(answer) != 0 else 'could not find an answer'  


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)



def finding_answer(Con_list,query) :
    vector1 = text_to_vector(query)
    pattern3 = re.compile(r"[\n]+")
    pattern4 = re.compile(r"[=]+") 
    
    for i in range(len(Con_list)) :
        Con_list[i] = pattern3.sub("",Con_list[i])
        Con_list[i] = pattern4.sub("",Con_list[i])
    
    text = ""
    for i in range(len(Con_list)) :
        text += Con_list[i] 
        text += " "
        
    text = text.split(".")
    
    
    max_score = {}
    for i in range(0,len(text)-1,1) :
        context = text[i] + text[i+1]
        vector2 = text_to_vector(context)
        max_score[get_cosine(vector1,vector2)] = i 

    l = list(max_score.items())
    l.sort(reverse=True)
    max_score= dict(l)
    
    
    ans = "could not find an answer"
    done = 0 
    for k in max_score :
        index = max_score[k] 
        context = ""
        if(index != len(text)-1) :
            context = context = text[index] + text[index+1]
        else :
            context = context = text[index] + text[index-1]
        ans = answer(query,context)
        #print(ans,context)
        #print(done)
        done += 1 
        if(ans != "could not find an answer" ) :
            ans = ans.replace("[CLS]","")
            ans = ans.replace(query.lower(),"")

            print(ans) 
            print("And context is ....................\n")
            print(context)

            return ans , context

            break 
        if(done == 20) :
            print("Sorry couldn't find a perfect answer !!!! :((")

            return "Sorry couldn't find a perfect answer !!!! :((" , "No context"

            break 

    
    

def complete(query) :
    query = query.lower()
    print("Query is ..............",query)
    Con_list , num_type = all_preprocessing(query)
    print()
    #print("Done preprocessing")
    #print(Con_list ,"is the content")
    if(num_type == 1) :
        answer , context = finding_answer(Con_list,query)
        
    
        return answer , context
    else :
        return "Sorry couldn't find a perfect answer !!!! :((" , "No context"
    




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

    answer , context = complete(q) ; 




    data = {
    'answer' : answer  , 
    'context' : context 

    }



    return render(request , "answer.html",data)


    

    

        

