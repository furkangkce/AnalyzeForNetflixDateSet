## NETFLIX DataSet


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols





pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

netflix = pd.read_csv("netflix_data.csv", sep=';')
print(netflix.head())


yil_sayim = netflix["release_year"].value_counts().sort_index()

plt.plot(yil_sayim.index, yil_sayim.values,color="#D9581E")
plt.xlabel("Year",font="Arial",fontsize=15,fontweight="bold")
plt.ylabel("Total Contents",font="Arial",fontsize=15,fontweight="bold")
plt.title("Total Contents By Year",font="Arial",fontsize=18,fontweight="bold")
plt.grid()
plt.savefig("TotalContentsByYear.png", dpi=300, bbox_inches='tight')
plt.show()

netflix_movie = netflix[netflix["type"] == "Movie"]
print(netflix_movie.head())
netflix_movies2 = netflix_movie[["title","country","genre","release_year","duration"]]
print(netflix_movies2.head())


## GRAPH duration by year
plt.scatter(netflix_movies2["release_year"],netflix_movies2["duration"],color="darkred",alpha=0.7)
plt.title("Film Durations By Year",font="Arial",fontsize=20,fontweight="bold")
plt.xlabel("Release Year")
plt.grid()
plt.xlabel("Release Year",font="Arial",fontsize=18,fontweight="bold")
plt.ylabel("Duration",font="Arial",fontsize=18,fontweight="bold")
plt.savefig("Durations By Year.png",dpi=300, bbox_inches='tight')
plt.show()

## durations<60
short_movies = netflix_movies2[netflix_movies2["duration"] <60 ]
print(short_movies.head())

## graph for type(duration<60)

colors=[]

for i in short_movies["genre"]:
    if i=="Children":
        colors.append("red")
    elif i=="Documentaries":
        colors.append("blue")
    elif i=="Stand-Up":
        colors.append("green")
    else:
        colors.append("black")

plt.scatter(short_movies["duration"],short_movies["genre"],color=colors,alpha=0.7)
plt.xlabel("Duration (min)",font="Arial",fontsize=15,fontweight="bold")
plt.title("Categorical Graph Of Short Films",font="Arial",fontsize=18,fontweight="bold")
plt.grid()
plt.savefig("Types By Duration.png",dpi=300, bbox_inches='tight')
plt.show()



# ANOVA

#H₀ (Null): Ülkelerin filmlerin zamanı için bir etkisi yoktur
#H₁ (Alternatif): En az bir ülkenin ortalama süresi farklıdır


#H₀ (Null): Film Türlerinin film uzunluğuna etkisi yoktur
#H₁ (Alternatif): En az biri farklıdır

#H₀ (Null): Ülke ve Film Türleri arasında etkileşim yoktur
#H₁ (Alternatif): Ülke ve Film Türleri arasında etkileşim vardır

print(type(netflix_movie))

netflix_cntry_drtn = netflix_movie[["country","duration","genre"]]
print(netflix_cntry_drtn.head())
netflix_cntry_drtn["duration"] = pd.to_numeric(netflix_cntry_drtn["duration"]) ## transform numeric

netflix_cntry_drtn["country"] = netflix_cntry_drtn["country"].str.split(", ")


netflix_cntry_drtn = netflix_cntry_drtn.explode("country")


netflix_cntry_drtn = netflix_cntry_drtn.groupby("country").filter(lambda x: len(x) > 20)
netflix_cntry_drtn = netflix_cntry_drtn.groupby("genre").filter(lambda x: len(x) > 20)

model = ols('duration ~ C(country) + C(genre) + C(country):C(genre)',
    data=netflix_cntry_drtn).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

anova_table.to_csv("anova_table.csv")
