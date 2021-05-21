from django.db import models

# Create your models here.
class Data(models.Model) :
    Title = models.CharField(max_length = 250)
    Summary = models.TextField()
    Categories = models.TextField() 
    Content = models.TextField() 
    Related_links = models.TextField() 


    @classmethod 
    def create(cls, **kwargs) :

        dataaa = cls.objects.create(
            Title = kwargs['Title'] , 
            Summary = kwargs['Summary'] ,
            Categories = kwargs['Categories'] ,
            Content = kwargs['Content'] ,
            Related_links = kwargs['Related_links']

        )


        return dataaa


class Keyword(models.Model) :
    keyword = models.TextField() 

    @classmethod 
    def create(cls , **kwargs) :

        dataaa = cls.objects.create(
            keyword = kwargs['word'] 
        )

        return dataaa