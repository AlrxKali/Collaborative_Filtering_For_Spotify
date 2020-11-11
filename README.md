[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="#">
    <img src="img/LogoSample_ByTailorBrands (1).jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Collaborative Filtering for Spotify</h3>

  <p align="center">
    Alejandro Alemany
    <br />
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">View Demo</a>
    ·
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">Report Bug</a>
    ·
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">Request Feature</a>
  </p>
</p>

[license-shield]: img/license.svg
[license-url]: https://github.com/AlrxKali/AB_testing_Mobile_game/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/alejandro-alemany/

<hr>

Collaborative filtering (CF) is a technique used by recommender systems. Collaborative filtering has two senses, a narrow one and a more general one.

In the newer, narrower sense, collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue than that of a randomly chosen person. For example, a collaborative filtering recommendation system for television tastes could make predictions about which television show a user should like given a partial list of that user's tastes (likes or dislikes). Note that these predictions are specific to the user, but use information gleaned from many users. This differs from the simpler approach of giving an average (non-specific) score for each item of interest, for example based on its number of votes.

In the more general sense, collaborative filtering is the process of filtering for information or patterns using techniques involving collaboration among multiple agents, viewpoints, data sources, etc. Applications of collaborative filtering typically involve very large data sets. Collaborative filtering methods have been applied to many different kinds of data including: sensing and monitoring data, such as in mineral exploration, environmental sensing over large areas or multiple sensors; financial data, such as financial service institutions that integrate many financial sources; or in electronic commerce and web applications where the focus is on user data, etc. The remainder of this discussion focuses on collaborative filtering for user data, although some of the methods and approaches may apply to the other major applications as well.

Collaborative filtering is an algorithm used for the implementation of recommender systems. Many wells know apps like Netflix and Youtube use these algorithms to recommend new content to their users.

There are two different types of collaborative filtering, a narrow one and a general one. 

The narrow algorithms are used to get automated predictions about the user's interest by collecting the behavior and preferences of many users. For example, this method might predict which television show you may like based on your previous likes and dislikes. 

The general one will filter for information and patterns involving collaboration from different data sources. For example, this method is useful when finding oil, mineral exploration, or environmental sensing over large areas. 

# Data Analysis

In this project, I will be implementing a narrow collaborative filter for Spotify. Like Youtube and Facebook, Spotify also recommends to their user. I will be using data collected from the ['Million Song Dataset'](http://millionsongdataset.com/). 


```python
!pip install fuzzywuzzy
```

    Requirement already satisfied: fuzzywuzzy in /usr/local/lib/python3.6/dist-packages (0.18.0)
    


```python
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import numpy as np
```

    /usr/local/lib/python3.6/dist-packages/fuzzywuzzy/fuzz.py:11: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning
      warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')
    


```python
#Read userid-songid-listen_count
song_info = pd.read_csv('https://static.turi.com/datasets/millionsong/10000.txt',
                        sep='\t', header=None)
song_info.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_actual =  pd.read_csv('https://static.turi.com/datasets/millionsong/song_data.csv')
song_actual.drop_duplicates(['song_id'], inplace=True)

#Merge the two dataframes above to create input dataframe for recommender systems
songs = pd.merge(song_info, song_actual, on="song_id", how="left")
```


```python
songs.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>song_id</th>
      <th>listen_count</th>
      <th>title</th>
      <th>release</th>
      <th>artist_name</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOAKIMP12A8C130995</td>
      <td>1</td>
      <td>The Cove</td>
      <td>Thicker Than Water</td>
      <td>Jack Johnson</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOBBMDR12A8C13253B</td>
      <td>2</td>
      <td>Entre Dos Aguas</td>
      <td>Flamenco Para Niños</td>
      <td>Paco De Lucia</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOBXHDL12A81C204C0</td>
      <td>1</td>
      <td>Stronger</td>
      <td>Graduation</td>
      <td>Kanye West</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOBYHAJ12A6701BF1D</td>
      <td>1</td>
      <td>Constellations</td>
      <td>In Between Dreams</td>
      <td>Jack Johnson</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SODACBL12A8C13C273</td>
      <td>1</td>
      <td>Learn To Fly</td>
      <td>There Is Nothing Left To Lose</td>
      <td>Foo Fighters</td>
      <td>1999</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SODDNQT12A6D4F5F7E</td>
      <td>5</td>
      <td>Apuesta Por El Rock 'N' Roll</td>
      <td>Antología Audiovisual</td>
      <td>Héroes del Silencio</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SODXRTY12AB0180F3B</td>
      <td>1</td>
      <td>Paper Gangsta</td>
      <td>The Fame Monster</td>
      <td>Lady GaGa</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOFGUAY12AB017B0A8</td>
      <td>1</td>
      <td>Stacked Actors</td>
      <td>There Is Nothing Left To Lose</td>
      <td>Foo Fighters</td>
      <td>1999</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOFRQTD12A81C233C0</td>
      <td>1</td>
      <td>Sehr kosmisch</td>
      <td>Musik von Harmonia</td>
      <td>Harmonia</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOHQWYZ12A6D4FA701</td>
      <td>1</td>
      <td>Heaven's gonna burn your eyes</td>
      <td>Hôtel Costes 7 by Stéphane Pompougnac</td>
      <td>Thievery Corporation feat. Emiliana Torrini</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>10</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOIYTOA12A6D4F9A23</td>
      <td>1</td>
      <td>Let It Be Sung</td>
      <td>If I Had Eyes</td>
      <td>Jack Johnson / Matt Costa / Zach Gill / Dan Le...</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>11</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOIZAZL12A6701C53B</td>
      <td>5</td>
      <td>I'll Be Missing You (Featuring Faith Evans &amp; 1...</td>
      <td>No Way Out</td>
      <td>Puff Daddy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOJNNUA12A8AE48C7A</td>
      <td>1</td>
      <td>Love Shack</td>
      <td>Original Hits - Rock</td>
      <td>The B-52's</td>
      <td>1989</td>
    </tr>
    <tr>
      <th>13</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOJPFQG12A58A7833A</td>
      <td>1</td>
      <td>Clarity</td>
      <td>As/Is: Cleveland/Cincinnati_ OH - 8/03-8/04/04</td>
      <td>John Mayer</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>b80344d063b5ccb3212f76538f3d9e43d87dca9e</td>
      <td>SOKRIMP12A6D4F5DA3</td>
      <td>5</td>
      <td>I?'m A Steady Rollin? Man</td>
      <td>Diggin' Deeper Volume 7</td>
      <td>Robert Johnson</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
songs.shape
```




    (2000000, 7)



This dataset has two million rows and seven variables. Also, we have 9567 unique songs and 76,353 individual users. 


```python
un_users = songs['user_id'].nunique()
un_songs = songs['title'].nunique()

print(f'Number of unique users: {un_users}')
print(f'Number of unique songs: {un_songs}')
```

    Number of unique users: 76353
    Number of unique songs: 9567
    

### So how do we know if a user might like a recommended song?

There is an explicit rating in which we ask the users to provide ratings so we can predict how much they might like the song. 

<span>
<p align="center">
  <a href="#">
    <img src="img/rating.jpg" alt="rating" width="250" height="250" align="center">
  </a>
</span>
<br>

Photo by [mohamed hassan](https://pxhere.com/en/photographer/767067) form [PxHere](https://pxhere.com/en/photo/1584361)

Moreover, there is an implicit rating in which we use other features to know how much a user likes an item.

In our case, we will use listen_count. If a user listens to the same song repeatedly, the chances are that they like it. 



```python
song_mean = round(songs.groupby(by=['user_id']).count().mean())
song_median = round(songs.groupby(by=['user_id']).count().median())

print(f'Users listen to the same song by an average of {song_mean[0]} times.')
print(f'Users listen to the same song by a median of {song_median[0]} times.')
```

    Users listen to the same song by an average of 26.0 times.
    Users listen to the same song by a median of 16.0 times.
    

We can see that users listen to the song they like 26 times on average, and the median value is 16. 

Now that we have rating, we can create interation matrices.

### What is a interaction matrix?

In our case, an interaction matrix is the collection of many entries paring users' ratings to songs. This matrix might come with its problems and challenges. They might have missing values, and the data can be very sparse. 

No users listen to all the songs. Therefore, our data is very sparse, and trying to implement a model such as scant information is costly in terms of time and computer resources. Consequently, we should take the data from users that have listened to at least 16 songs. 

We need to reshape the data frame, so we have song_id as the index and the user_id as the variables. To accomplish this task, we are going to convert the data frame to a pivot table. 


```python
df_unique_songs = songs.drop_duplicates(subset=['song_id']).reset_index(drop=True)[['song_id', 'title']]
```


```python
# Get number of songs each user has listened
song_user = songs.groupby('user_id')['song_id'].count()

# Get users which have listen to at least 16 songs
song_ten_id = song_user[song_user > 16].index.to_list()

# Filtered the dataset to keep only those users with more than 16 listened
df_song_id_more_ten = songs[songs['user_id'].isin(song_ten_id)].reset_index(drop=True)
```


```python
df_songs_features = df_song_id_more_ten.pivot(index='song_id', columns='user_id', values='listen_count').fillna(0)

# obtain a sparse matrix
mat_songs_features = csr_matrix(df_songs_features.values)

# take a look at the pivot table
df_songs_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>000e2c2a8c7870ff9121f212b35c8b3a20cc0e67</th>
      <th>000ebc858861aca26bac9b49f650ed424cf882fc</th>
      <th>000ef25cc955ad5841c915d269432eea41f4a1a5</th>
      <th>0012bf75d43a724f62dc746d9e85ae0088a3a1d6</th>
      <th>001322829b5dc3edc59bf78189617ddd8f23c82a</th>
      <th>00185e316f07f0f00c325ca034be59c15b362401</th>
      <th>0019740e3e8c24e223a6f88e3faa7c144ec5a014</th>
      <th>001b005fe5e80d3cb45f320f5658fc8e2e72794e</th>
      <th>001f22c638730aed5659034c447d3cf0e658898e</th>
      <th>0021d9a4628624f6d70237f9c200ab82e766bf26</th>
      <th>002543003041db1d049206b09426d5cdffc0f451</th>
      <th>0028292aa536122c1f86fd48a39bd83fe582d27f</th>
      <th>00292cf9c6d6e99c5ddbece7e37f957ab1362d25</th>
      <th>00296f66ed7fb84c876486aecc9fab2d5809576d</th>
      <th>0030822badc23ef6500a72ce7feda1c63faf2262</th>
      <th>0031572620fa7f18487d3ea22935eb28410ecc4c</th>
      <th>003412e33eb3d05573f7811c1ba61d6a15be5690</th>
      <th>00342a0cdf56a45465f09a39040a5bc25b7d0046</th>
      <th>00388e5764c59488ec06a109c88b39f59a2b6361</th>
      <th>0039bd8483d578997718cdc0bf6c7c88b679f488</th>
      <th>003ac50a4e6ed0c9085fecb7a1738730e7ea4942</th>
      <th>003bfb50126f91f6389aaee733f5b3e0a8d5cbe0</th>
      <th>003d0f3aac94fd261bb74c0124a90750579972d4</th>
      <th>003d21762b29fe2ffe20fb9a51eb1e02ebeb3242</th>
      <th>003d6d799d58e1fce362f5f4f6c7bcc26c8f3546</th>
      <th>003e3919f41dbb8ff05a75623d205f6abcede4fc</th>
      <th>003f1064ed75d1156352cd89b25fb752bcc10b13</th>
      <th>003f1e939952a57d1a5bc990727acad5ceea97b4</th>
      <th>00409f6a83c2bf4299ab6ae2dea958050537b5a9</th>
      <th>0041925615557845642a7b1257fdc6229fe1ee5d</th>
      <th>00454c72c0b4b99f9cc81ba0b1989597a43669d6</th>
      <th>0045c60d98ced5efb3cbf1e0b4b7de1da3f1a506</th>
      <th>00488ec44caa0d2fa669780f8cb604bf39e94392</th>
      <th>004c7be9336ca88824e1e0b09ef9a2168200fa33</th>
      <th>004dc9f93f5ad4a75f9a3ba0da5dd887b31d6bd2</th>
      <th>004f6065fa9840913f62e52d94d9c29ea1d26fe1</th>
      <th>004fcdf8829d68f4e45ef846ad9f308c4493ed8e</th>
      <th>0051a2e7b452e3dc67f48688442032df557897f4</th>
      <th>00544d8bde0d7985e8d703c1eb676d41cad33c67</th>
      <th>00546de8971645143eead323561d0298d5b0f2be</th>
      <th>...</th>
      <th>ffb63da2222280f299a7a896edea073728aab343</th>
      <th>ffb7096f3eeee706825b4a8c3fab98ce0e0b4216</th>
      <th>ffb8299fcb3f31716c93fb8a77dee0d1dbb210f9</th>
      <th>ffba3563fd590a51dabc1dfcb8a27119e6035241</th>
      <th>ffbc58b89c81227a08ee05d90259bdd9172b9479</th>
      <th>ffbe437f18c3bea5c78596c5f3183ee62d440b6c</th>
      <th>ffc2563e7dd136bf371371cd21f0cc404a1d7499</th>
      <th>ffc564b85d81f0ea427cee3eea2415cc2fc5c4be</th>
      <th>ffc5b3bd0575330eb8c4ae4f3b9cabd3ba315dbb</th>
      <th>ffc66b4520671da6b2a67a0326201565160d9650</th>
      <th>ffc9966885909a0b42493b2558be9dc451317488</th>
      <th>ffcc2cff250ea22471df09e76f59e2be0debae72</th>
      <th>ffcfb0b34a47fdf55b3d96c1799cd196677f8261</th>
      <th>ffd1d617221f5bf00de80ee3eb5d2a17fc8d077d</th>
      <th>ffd25d7da8b4e54ea2cde25dd3b52d0e0aef7a5d</th>
      <th>ffd458f903d49854685cf4540245c1d297e8bafa</th>
      <th>ffd6f34b343cee62cd7dbbf20fb1ab1119b299e5</th>
      <th>ffda2736b2204ece3b19e941caacad4106d00ed5</th>
      <th>ffdaab327f2fc6b9fa01a4e3e7f41fdd0e468046</th>
      <th>ffdb5557e4e24da051ebd6b45ec18c524c55dc40</th>
      <th>ffdbaeb5cf2081eb34053a655c20f57524de11ba</th>
      <th>ffde97c0d23bf5ce249ce73e630ccb4e7293cc32</th>
      <th>ffe2a7e7b6689071f8c699d944b52ce590ae4636</th>
      <th>ffe33dce4e652a4dc4824cc39680a9f709cfdfb3</th>
      <th>ffebfec313fd515a11faba060b022f030b57fac0</th>
      <th>ffecbb84f3dde31b0b2e64d70b7b7e5092bf7427</th>
      <th>ffef9c3e59ab44554a9775af5e3b2ac149111bb6</th>
      <th>fff03efd1550136063389fa71125194614e1c68f</th>
      <th>fff0b1ab076f0b71cbde9c7dcbcfca400708d845</th>
      <th>fff22417a61c1ba3ee2592b22a052ed6a27a8e91</th>
      <th>fff300cd094fe04030b79fae550dc9d065190182</th>
      <th>fff4676dacb2e9a7217702f62ee70e88aa512ecc</th>
      <th>fff4e1a7dacbe9c13051c08f09bf66d76cbee35e</th>
      <th>fff543db7918cb8f4f56f7470903eb2f1d5a6dd8</th>
      <th>fff6c30c773e6ffafcac213c9afd9666afaf6d63</th>
      <th>fffb701ee87a32eff67eb040ed59146121f01571</th>
      <th>fffc0df75a48d823ad5abfaf2a1ee61eb1e3302c</th>
      <th>fffce9c1537fbc350ea68823d956eaa8f5236dbe</th>
      <th>fffd9635b33f412de8ed02e44e6564e3644cf3c6</th>
      <th>fffea3d509760c984e7d40789804c0e5e289cc86</th>
    </tr>
    <tr>
      <th>song_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SOAAAGQ12A8C1420C8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SOAACPJ12A81C21360</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SOAACSG12AB018DC80</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SOAAEJI12AB0188AB5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SOAAFAC12A67ADF7EB</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36561 columns</p>
</div>



### Ok, but how are you going to recommend?

There are two approaches to recommend items in a collaborative filter. First, a user-based system in which we analyze users with similar interests and behaviors. We collect those users that we think have a similar taste and have listened to similar songs, and we recommend based on their previous likes. 

The second approach is a technique known as item-based. In this algorithm, we consider songs that the user might have liked in the past.

# Which machine learning algorithm can help us accomplish this task?

Three ML algorithms can help us to solve this project. We will be using the Nearest Neighbors(KNN) for this project. 

Matrix Factorization is another algorithm that can help us to get similar results. It is a powerful tool that is very used in recommender systems. It works by representing items and users in a lower-dimensional latent space. Therefore, this method decomposes the sparsity in the user-item matrix into a lower dimensionality in rectangular matrices with latent features. 

We will be building a recommender system for youtube with this method to show how it works. 

Another method for building recommender systems is clustering. We will be creating a recommender system for Netflix using this algorithm. 

### Why K-Nearest Neighbors?

It is a well-known algorithm that is considered a standard approach for building collaborative filters. Also, it is a good starting point for this series of projects making collaborative filtering. 

The KNN is a supervised method used for regression and classification. A technique is supervised when it needs a labeled data point to work. Besides, this is a non-parametric method. That means we do not need to make any assumption regarding the distribution of the data. 

In other words, it will assume that similar things are located near to each other, and the distance among the points measure how similar or "close" they are. 

# Insert an images showing knn

Therefore, the knn will calculate the distance between the target song and all other songs in our dataset to make predictions. The algorithms will then pick the closest position and return the top k nearest neighbor as recommendations. 


```python
decode_id_song = {
    song: i for i, song in 
    enumerate(list(df_unique_songs.set_index('song_id').loc[df_songs_features.index].title))
}
```


```python
class Recommender:
    def __init__(self, metric, algorithm, k, data, decode_id_song):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_song = decode_id_song
        self.data = data
        self.model = self._recommender().fit(data)
    
    def make_recommendation(self, new_song, n_recommendations):
        recommended = self._recommend(new_song=new_song, n_recommendations=n_recommendations)
        print("... Done")
        return recommended 
    
    def _recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)
    
    def _recommend(self, new_song, n_recommendations):
        # Get the id of the recommended songs
        recommendations = []
        recommendation_ids = self._get_recommendations(new_song=new_song, n_recommendations=n_recommendations)
        # return the name of the song using a mapping dictionary
        recommendations_map = self._map_indeces_to_song_title(recommendation_ids)
        # Translate this recommendations into the ranking of song titles recommended
        for i, (idx, dist) in enumerate(recommendation_ids):
            recommendations.append(recommendations_map[idx])
        return recommendations
                 
    def _get_recommendations(self, new_song, n_recommendations):
        # Get the id of the song according to the text
        recom_song_id = self._fuzzy_matching(song=new_song)
        # Start the recommendation process
        print(f"Starting the recommendation process for {new_song} ...")
        # Return the n neighbors for the song id
        distances, indices = self.model.kneighbors(self.data[recom_song_id], n_neighbors=n_recommendations+1)
        return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    def _map_indeces_to_song_title(self, recommendation_ids):
        # get reverse mapper
        return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}
    
    def _fuzzy_matching(self, song):
        match_tuple = []
        # get match
        for title, idx in self.decode_id_song.items():
            ratio = fuzz.ratio(title.lower(), song.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print(f"The recommendation system could not find a match for {song}")
            return
        return match_tuple[0][1]
```

# Implementing the model

This project is a real-world application. Therefore, it requires a more efficient design that can be easily maintained and improved over time. For these purposes, I created a class "Recommender." A class is a blueprint for creating objects. It has methods and properties of the object. 

The functions inside the class are methods that will be used in the object and its properties.


```python
def _fuzzy_matching(self, song):
    match_tuple = []
    # get match
    for title, idx in self.decode_id_song.items():
        ratio = fuzz.ratio(title.lower(), song.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
            
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    
    return match_tuple[0][1]
```


```python
model = Recommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, 
                    decode_id_song=decode_id_song)
```

We created an object called "model." We pass the metric, algorithm, k, data, and decode_id_song parameters to the constructor to build the object. The next step is to apply the methods to the object. 

We are going to call the make_recommendation method, with a song to see how the model work.


```python
song = 'same old love'
```


```python
new_recommendations = model.make_recommendation(new_song=song, n_recommendations=10)

print(f"The recommendations for {song} are:")
print(f"{new_recommendations}")
```

    Starting the recommendation process for same old love ...
    ... Done
    The recommendations for same old love are:
    ['Shake A Tail Feather', 'Fire And Rain', 'You Picked Me', 'Lived In Bars', 'I Found A Reason', 'Do-Wah-Doo', 'Hard To Concentrate (Album Version)', 'Metal Heart', "What I Wouldn't Do", 'Happier']
    

The model is working correctly, and it is recommending ten songs. 

# Other appliations of collaborative filtering

This type of recommender system is used to analyze the interactions that users have with items. The advantage of collaborative filters is that they do not require features about the user and the objects. Also, These recommenders do not put weight on the user profile and recommend items that might be completely different from what they have used before. 

They are widely used, but they have their challenges as well. Data sparsity can affect the model and lower the quality of the system. Scaling is also problematic in a huge dataset, and other models like matrix factorization might be used. 

# Conclusion

We have built a collaborative filter for Spotify. We have seen what type of model can be used, the mathematics behind the models, and how to apply it with the model. This project still has room for improvement, and I will be posting the new versions soon. 
