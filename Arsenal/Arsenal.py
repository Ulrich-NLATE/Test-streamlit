#!/usr/bin/env python
# coding: utf-8

# Commandes pour lancer le fichier

# cd C:\Users\matba\Desktop\DataScientest\Projet\Arsenal
# streamlit run Arsenal.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from streamlit_option_menu import option_menu
pd.plotting.register_matplotlib_converters()

#Partie ML
from pycaret.datasets import get_data
from pycaret.classification import *
from pycaret.classification import setup, compare_models
from scipy.stats import chi2_contingency
import scipy.stats as stats
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LassoCV, Lasso 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import datetime as dt

# Eviter les messages d'erreur sur certains graphiques
st.set_option('deprecation.showPyplotGlobalUse', False)

import warnings
warnings.filterwarnings('ignore')

# Importer glob pour gérer les images
import glob

#########################################

st.set_page_config(
    page_title="Qu'est-il arrivé à Arsenal ?",
    page_icon="⚽",
)

########################################

header = st.container()
logo = st.container()
intro = st.container()

import base64
from PIL import Image

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

image1_base64 = img_to_base64("Matthieu2.png")
image2_base64 = img_to_base64("Ulrich2.png")
datascientest_logo_base64 = img_to_base64("datascientest.png")
canon_logo_base64 = img_to_base64("canon.png")
premier_league_logo_base64 = img_to_base64("premier_league_logo.png")

# Ajouter le code CSS ici
st.markdown(
    """
    <style>
    .sidebar {
        font-family: 'ff-meta-web-pro', sans-serif;
    }
    .css-vk3wp9 {
        background-color: #9C824A;
        color: white;
    }
    .team-players-title::before {
        content: "";
        display: block;
        width: 100px;
        height: 5px;
        background-color: white;
        margin-bottom: 10px;
    }
    .bi-linkedin {
        color: #063672;
    }
    .team-images {
        display: flex;
        justify-content: center;
    }
    .team-image {
        width: 120px;
        height: auto;
        display: inline-block;
        margin-bottom: 10px;
        margin-right: 10px;
    }
     .team-players-title {
        font-weight: bold;
        margin-top: 10px;
        font-size: 20px;
    }
    .css-promotion {
        margin-top: 20px;
    }
    .css-promotion-text {
        color: #063672;
    }
    .datascientest-logo {
        width: 25px;
        height: auto;
        display: inline-block;
        margin-left: 5px;
    }
    .canon-logo {
        width: 150px;
        height: auto;
        display: block;
        margin: 0px auto;
    }
    .sidebar-content {
        display: flex;
        flex-direction: column;
    }
    .premier-league-logo-container {
        text-align: center;
    }
    </style>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    """,
    
    unsafe_allow_html=True,
)
   
# Sidebar
with st.sidebar:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-content">
            <div class="canon-logo-container">
                <img src="data:image/png;base64,{canon_logo_base64}" alt="Canon" class="canon-logo" />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    selected = option_menu(
        menu_title=None,
        options=["Introduction", "Dataset", "Storytelling / Dataviz", "Machine Learning", "Conclusion"],
        icons=["ballon", "ballon", "ballon", "ballon", "ballon"],
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#DB0007"},
                "nav-link": {"font-size": "16px", "margin": "5px", "--hover-color": "#063672", "color": "white"},
                "container": {"padding": "500", "background-color": "#9C824A", "border-radius": "0"}}
    )

st.sidebar.markdown(
    f"""
    <p class="css-vk3wp9 team-players-title">Team players :</p>
    <div class="team-images">
        <img src="data:image/png;base64,{image1_base64}" alt="Image 1" class="team-image" />
        <img src="data:image/png;base64,{image2_base64}" alt="Image 2" class="team-image" />
    </div>
    <p class="css-vk3wp9">Matthieu BADINA  <a href="https://www.linkedin.com/in/matthieubadina/"><i class="bi bi-linkedin"></i></a></p>
    <p class="css-vk3wp9">Ulrich NLATE MVOLO  <a href="https://www.linkedin.com/in/ulrich-nlate-mvolo-933b2b123/"><i class="bi bi-linkedin"></i></a></p>
    <p class="css-vk3wp9 css-promotion"><span class="css-promotion-text">Promotion Continue DA - Octobre 2002</span><a href="https://datascientest.com/" target="_blank"><img src="data:image/png;base64,{datascientest_logo_base64}" alt="Datascientest" class="datascientest-logo" /></a></p>
    """,
    unsafe_allow_html=True,
)
    
########################################
# Intro
########################################
if selected == "Introduction":
    # Header
    with logo:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            arsenal_link = "https://www.arsenal.com/"
            st.markdown(
                f'<a href="{arsenal_link}" target="_blank"><img src="data:image/png;base64,{img_to_base64("Arsenal.png")}" width="200" /></a>',
                unsafe_allow_html=True,
            )
        with col3:
            st.write(' ')

    with header:
        st.markdown("<h1 style='text-align: center; color: #DB0007;'>Qu'est-il arrivé à Arsenal ?</h1>", unsafe_allow_html=True)

    # Intro
    with intro:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write("<div style='text-align: justify;'> Ce projet a été réalisé dans le cadre de la formation Data Analyst de <a href='https://datascientest.com/'>DataScientest.com</a>. L’objectif est de développer un data storytelling permettant d’expliquer, grâce aux données, comment le niveau du club de football Arsenal s’est dégradé pendant la dernière décennie.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>Un peu d'histoire</span>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'> Arsenal Football Club, communément appelé Arsenal, est un club anglais de football, fondé le 1er décembre 1886 à Londres. Son siège est situé dans le borough londonien d'Islington, au nord de la capitale britannique.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.markdown(
            f"""
            <div class="premier-league-logo-container">
                <a href="https://www.premierleague.com/home" target="_blank">
                    <img src="data:image/png;base64,{premier_league_logo_base64}" alt="Premier League" width="100" />
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(' ')
        st.write(' ')
        st.write("<div style='text-align: justify;'> Depuis sa création en 1992, la <a href='https://www.premierleague.com/'>Premier League</a> a vu Arsenal FC briller à plusieurs reprises. Sous la houlette de l'emblématique entraîneur Arsène Wenger, le club a décroché trois titres de champion, dont un mémorable en 2004, lorsqu'il est resté invaincu tout au long de la saison. En plus de ces succès en championnat, Arsenal détient également un record impressionnant de quatorze coupes d'Angleterre (FA Cup) à son actif.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'> Pendant de nombreuses années, Arsenal a été dirigé par les familles Bracewell-Smith et Hill-Wood, qui ont su conserver une certaine stabilité au sein du club. Cependant, en 2011, l'homme d'affaires américain Stan Kroenke a racheté la majorité des parts du club, modifiant ainsi sa structure de gestion. Depuis lors, les résultats d'Arsenal sur le terrain ont connu une baisse notable. Au cours de ce projet, nous nous pencherons sur les raisons qui pourraient expliquer cette dégradation des performances et tenterons de mieux comprendre les facteurs qui ont contribué à cette situation.</div>", unsafe_allow_html=True)

    
########################################
# Dataset
########################################
if selected == "Dataset":
    # Header
    with header:
        st.markdown("<h1 style='text-align: center; color: #DB0007;'>Dataset</h1>", unsafe_allow_html=True)
  
    # Intro
    with intro:
        st.write(' ')
        st.write("<div style='text-align: justify;'>Dans cette section, nous détaillerons les jeux de données utilisés pour analyser les performances d'Arsenal FC, ainsi que les étapes suivies pour les préparer et les enrichir.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>KAGGLE</span>", unsafe_allow_html=True)
        st.write("<h3><span style='color: #EF0107;'>1) Premier jeu de données, exploration et préparation</span></h3>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Pour commencer, nous nous sommes appuyés sur un jeu de données provenant de <a href='https://www.kaggle.com/pablohfreitas/all-premier-league-matches-20102021' target='_blank'>Kaggle</a>, qui contient l'ensemble des matchs de Premier League des saisons 2010 à 2011. Ce jeu de données comprend 4070 lignes, correspondant aux matchs des 10 dernières saisons, et 114 colonnes de statistiques. Après une première exploration approfondie, nous avons identifié des données manquantes et constaté que les 117 derniers matchs de la saison 2020/2021 étaient absents.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Afin de mieux comprendre la structure et la qualité de nos données, nous avons réalisé plusieurs analyses, notamment sur les valeurs manquantes et la corrélation entre les différentes variables. Nous avons également éliminé certaines colonnes moins pertinentes pour notre étude, comme les liens vers les matchs sur le site de la <a href='https://www.premierleague.com/'>Premier League</a>.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<h3><span style='color: #EF0107;'>2) Webscraping et ajout des données manquantes</span></h3>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Pour récupérer les 117 matchs manquants, nous avons décidé d'extraire les données directement depuis le site de la <a href='https://www.premierleague.com/'>Premier League</a>. Bien que la bibliothèque <b><i>Beautifulsoup</i></b> ne nous ait pas donné les résultats escomptés, nous avons réussi à récupérer les informations nécessaires en utilisant la bibliothèque <b><i>Selenium</i></b>. Une fois les données obtenues, nous les avons ajoutées à notre jeu de données initial.</div>",unsafe_allow_html=True)
        st.write(' ')
        st.write("<h3><span style='color: #EF0107;'>3) Transformation des données</span></h3>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Afin de tirer le meilleur parti de nos données et de les adapter à notre étude, nous avons procédé à plusieurs transformations. Nous avons notamment regroupé les données en utilisant la fonction groupby et restructuré certaines colonnes pour obtenir une base solide sur laquelle s'appuyer. Grâce à ces transformations, nous avons finalement obtenu le jeu de données souhaité pour analyser les performances d'Arsenal FC au cours de la dernière décennie.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Vous pouvez consulter les différentes étapes de notre travail sur les données en cochant les cases correspondantes pour afficher les dataframes initiaux, complets et finaux, ainsi que les résumés, les valeurs manquantes et les matrices de corrélation.</div>", unsafe_allow_html=True)
        st.write(' ')
        
        df = pd.read_csv('df_full_premierleague.csv')
        
        def summary(df):
    
            table = pd.DataFrame(
                index=df.columns,
                columns=['type_info', '%_missing_values', 'nb_unique_values', 'list_unique_values', "mean_or_mode", "flag"])
            table.loc[:, 'type_info'] = df.dtypes.values
            table.loc[:, '%_missing_values'] = df.isna().sum().values / len(df)
            table.loc[:, 'nb_unique_values'] = df.nunique().values
    
            def get_list_unique_values(colonne):
                if colonne.nunique() < 6:
                    return colonne.unique()
                else:
                    return "Too many categories..." if colonne.dtypes == "O" else "Too many values..."             

            def get_mean_mode(colonne):
                return colonne.mode()[0] if colonne.dtypes == "O" else colonne.mean()

            def alerts(colonne):
                thresh_na = 0.25
                thresh_balance = 0.8
                if (colonne.count() / len(colonne)) < thresh_na:
                     return "Too many missing values ! "
                elif colonne.value_counts(
                        normalize=True).values[0] > thresh_balance:
                    return "It's imbalanced !"
                else:
                    return "Nothing to report"
        
            table.loc[:, 'list_unique_values'] = df.apply(get_list_unique_values)
            table.loc[:, 'mean_or_mode'] = df.apply(get_mean_mode)
            table.loc[:, 'flag'] = df.apply(alerts)
    
            return table
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe initial") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=df.shape[0])
            st.dataframe(df.head(line_to_plot))
            
        ### Showing summary df
        if st.checkbox("Résumé des données") :
            st.write(summary(df))
    
        ### Showing missing values
        if st.checkbox("Valeurs manquantes") : 
                ### Display graph
                fig, ax = plt.subplots(figsize=(20, 10))
                sns.heatmap(df.isna(), cbar=False, ax=ax)
                st.pyplot(fig)
                
                
        ### Showing heatmap
        if st.checkbox("Matrice de Corrélation") :
            
                df.drop(df.iloc[:, 38:], inplace=True, axis=1)
                df = df.drop(['Unnamed: 0','link_match','sg_match_ft','sg_match_ht'], axis=1)
                
                ### Display graph
                plt.figure(figsize=(16, 6))

                # Calculez la matrice de corrélation
                correlation_matrix = df.corr()

                # Appliquez un masque triangulaire supérieur
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

                # Appliquez un masque pour filtrer les corrélations faibles (par exemple, seuil de 0.5)
                threshold = 0.5
                mask_weak_corr = np.abs(correlation_matrix) > threshold
                filtered_corr_matrix = correlation_matrix * mask_weak_corr

                # Créez la heatmap en utilisant Seaborn
                heatmap = sns.heatmap(filtered_corr_matrix, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
                heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
                st.pyplot()

        match = pd.read_csv('match.csv')
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe complet") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=match.shape[0])
            st.dataframe(match.head(line_to_plot))

        data = pd.read_csv('arsenal.csv')       
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe final") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=data.shape[0])
            st.dataframe(data.head(line_to_plot))
            
        st.write(' ')
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>TRANSFERMARKT</span>", unsafe_allow_html=True)
        st.write("<h3><span style='color: #EF0107;'>1) Web scraping des données de transfert</span></h3>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Afin d'analyser l'impact des transferts sur les performances d'Arsenal FC, nous avons décidé d'extraire des données sur les mouvements de joueurs entrants et sortants du club. Pour cela, nous avons utilisé le site web <a href='https://www.transfermarkt.fr/' target='_blank'>Transfermarkt</a>, une référence en matière d'informations sur les transferts de football. Nous avons extrait les données de transfert des saisons 2010/11 à 2020/21 en utilisant les bibliothèques <b><i>requests</i></b> et <b><i>BeautifulSoup</i></b> pour effectuer le web scraping.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<h3><span style='color: #EF0107;'>2) Préparation et nettoyage des données</span></h3>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Une fois les données extraites, nous les avons nettoyées et préparées pour l'analyse. Cela inclut la conversion des montants de transfert au format numérique et le calcul des sommes totales de transfert pour chaque saison et type de transfert (arrivées et départs). Nous avons également filtré les données pour ne conserver que les saisons de 2010/11 à 2020/21, qui sont les saisons d'intérêt pour notre étude.</div>", unsafe_allow_html=True)
        st.write(' ')

        transfert = pd.read_csv('transferts.csv')
        totals = pd.read_csv('totals.csv')
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe des transferts") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=transfert.shape[0])
            st.dataframe(transfert.head(line_to_plot))
            
        ### Showing the data
        if st.checkbox("Afficher le Dataframe des montants totaux") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=totals.shape[0])
            st.dataframe(totals.head(line_to_plot)) 
            
########################################
# Dataviz
########################################
if selected == "Storytelling / Dataviz":
    # Header
    with header:
        st.markdown("<h1 style='text-align: center; color: #DB0007;'>Storytelling / Dataviz</h1>", unsafe_allow_html=True)
        
    # Intro
    with intro:
        st.write(' ')
        st.write("<div style='text-align: justify;'>Arsenal FC, l'un des clubs de football les plus célèbres et titrés d'Angleterre, a connu des hauts et des bas au cours de la dernière décennie. Dans cette étude, nous allons explorer les performances d'Arsenal en Premier League à travers différents événements clés, tels que les changements de propriété, les transferts de joueurs importants et les changements d'entraîneurs. Nous analyserons également l'impact des transferts sur les performances du club au fil des saisons.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Pour cela, nous avons rassemblé un ensemble de données sur les matchs de Premier League d'Arsenal, les résultats et les statistiques des saisons 2010/11 à 2020/21. Nous avons également extrait des données sur les transferts de joueurs d'Arsenal au cours de ces saisons à partir de Transfermarkt. À l'aide de ces données, nous avons créé des visualisations et des analyses pour raconter l'histoire des performances d'Arsenal au cours de la dernière décennie.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Dans la section suivante, nous présenterons nos analyses et visualisations, en expliquant les tendances et les événements clés qui ont façonné les performances d'Arsenal. Nous inclurons également des graphiques comparatifs avec le Big 6 de la Premier League, qui regroupe les six meilleures équipes du championnat anglais : Arsenal, Chelsea, Liverpool, Manchester City, Manchester United et Tottenham Hotspur. Ces comparaisons permettront aux néophytes et aux amateurs de football de mieux comprendre la place d'Arsenal parmi les meilleures équipes de la ligue. Accompagnez-nous dans cette exploration des hauts et des bas d'Arsenal FC.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>Exploration des statistiques</span>", unsafe_allow_html=True)
        st.write(' ')
        
        arsenal = pd.read_csv('arsenal.csv')

        arsenal['result'] = np.nan
        for i in range(len(arsenal['Team'])):
            if (arsenal['Team_Goal'][i]>arsenal['Opponent_Goal'][i]):
                arsenal['result'][i] = 'H'
            elif (arsenal['Team_Goal'][i]<arsenal['Opponent_Goal'][i]):
                arsenal['result'][i] = 'A'
            elif (arsenal['Team_Goal'][i]==arsenal['Opponent_Goal'][i]):
                arsenal['result'][i] = 'D'
            
        arsenal['points'] = 0
        arsenal.loc[arsenal.result=='H', 'points'] = 3
        arsenal.loc[arsenal.result=='D', 'points'] = 1    
        
        # create the scores table
        scores = pd.DataFrame()
        for t in arsenal.Team.unique():
            arsenal1 = arsenal[arsenal['Team']==t]
            
            team_score = arsenal1.groupby('season').sum()
            team_score['team'] = t
            scores = pd.concat([scores, team_score], axis=0)
            
        # make the pivot table of scores
        scores['season'] = scores.index
        scores = scores.pivot(index='team', columns='season', values='points')
        scores.fillna(0, inplace=True)
        
        ranks = pd.DataFrame()
        for c in scores.columns:
            rank = scores[c].rank(method='min', ascending=False)
            ranks = pd.concat([ranks, rank], axis=1)
            
        arsenal_data = arsenal.loc[arsenal.Team=='Arsenal']
        arsenal_data['goals'] = arsenal.Team_Goal
        arsenal_data.drop(['Team_Goal', 'Opponent_Goal'], axis=1, inplace=True)
        arsenal_scores = scores.loc['Arsenal']
        arsenal_goals = arsenal_data.groupby('season').sum()
        
        # define plot parameters
        colors = sns.color_palette('Paired') 
        labels = ranks.loc['Arsenal']
        xpoints = arsenal_scores.index
        ypoints = arsenal_scores
        ytext = [-15, 5, -15, -5, 0, 5, -15, 5, 0, -10, 0, 5, 8, -13, -5, 5]
        box_params = dict(boxstyle="round,pad=0.3", fc=colors[6], ec="b")
        arrow_params1=dict(arrowstyle="-|>", connectionstyle="arc3, rad=-0.3", color='black',ls=(0, (5, 10)))
        arrow_params2=dict(arrowstyle="-|>", connectionstyle="arc3, rad=0.3", color='black',ls=(0, (5, 10)))
        
        # plot the lines
        sns.set_style("ticks",{'axes.grid' : True})
        fig,ax = plt.subplots(1, 1, figsize=(24,18))
        ax.plot(xpoints, ypoints, linewidth=3, label='score', color='#DB0007')
        ax.plot(arsenal_goals.index, arsenal_goals.goals, linewidth=2, linestyle='--', label='goals', color="#063672")
        
        #plot the annotations 
        for label, x, y, yt in zip(labels, xpoints, ypoints, ytext):
            plt.annotate(
              'Rank: {}'.format(int(label)),
              xy=(x, y), xytext=(0, yt),
              textcoords='offset points',
              ha='center', fontsize=20,
              color=colors[9], weight='bold')
                
        ax.annotate('Stan Kroenke \n becomes majority owner',
                    xy=(1, 70), xytext=(2,65),
                    ha="center", va="center", size=20,
                    bbox=box_params,
                    arrowprops=arrow_params2)

        ax.annotate('Van Persie left Arsenal', 
                    xy=(2, 73), xytext=(1,75),
                    ha="center", va="center", size=20,
                    bbox=box_params,
                    arrowprops=arrow_params2)

        ax.annotate('Özil joined Arsenal for 50 M€', 
                    xy=(3, 79), xytext=(4,75),
                    ha="center", va="center", size=20,
                    bbox=box_params,
                    arrowprops=arrow_params2)

        ax.annotate('Wenger left Arsenal', 
                    xy=(7, 63), xytext=(6,61),
                    ha="center", va="center", size=20,
                    bbox=box_params,
                    arrowprops=arrow_params2)

        ax.annotate('Unai Emery \n became the manager', 
                    xy=(7, 63), xytext=(6,70),
                    ha="center", va="center", size=20,
                    bbox=box_params,
                    arrowprops=arrow_params2)

        ax.annotate('Mikel Arteta \n became the manager', 
                    xy=(8.5, 63), xytext=(9,65),
                    ha="center", va="center", size=20,
                    bbox=box_params,
                    arrowprops=arrow_params2)
                
        plt.ylabel('Premier League rank and goals', fontsize=20)
        plt.xlabel('Season', fontsize=20)
        plt.xticks(rotation=-30, fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 20})        
        
        # Render the plot in Streamlit
        st.pyplot(fig)
        
        st.write(' ')
        st.write("<div style='text-align: justify;'>Ce graphique représente l'évolution du classement et des buts marqués par Arsenal en Premier League au fil des saisons, ainsi que les moments clés qui ont marqué l'histoire du club. On peut voir que les performances de l'équipe ont fluctué au fil des ans, atteignant leur apogée au début des années 2000, avec une baisse de régime à partir de la saison 2012-2013. Les flèches et les boîtes de couleur indiquent des événements importants tels que le départ de Robin van Persie et le recrutement d'Özil pour 50 millions d'euros, ainsi que les changements de management avec l'arrivée d'Unai Emery et de Mikel Arteta. Mais pour mieux comprendre ce qu'il est arrivé à Arsenal, il nous faut nous pencher premièrement sur les résultats du club.</div>", unsafe_allow_html=True)
        st.write(' ')
        
        arsenal_games = arsenal.loc[(arsenal['Team'] == 'Arsenal')]

        arsenal_games.reset_index(inplace=True, drop=True)
        
        arsenal_games['result'] = np.nan
        for i in range(len(arsenal_games['Team'])):
            if (arsenal_games['Team_Goal'][i]>arsenal_games['Opponent_Goal'][i]):
                arsenal_games['result'][i] = 'W'
            elif (arsenal_games['Team_Goal'][i]<arsenal_games['Opponent_Goal'][i]):
                arsenal_games['result'][i] = 'L'
            elif(arsenal_games['Team_Goal'][i]==arsenal_games['Opponent_Goal'][i]):
                arsenal_games['result'][i] = 'D'
        
        # Créer une colonne pour la saison correspondante
        arsenal_games['Season'] = ['{}-{}'.format(str(int(x.split('/')[1])-1)[-2:], x[-2:]) for x in arsenal_games['season']]

        # Compter les victoires, défaites et nuls par saison
        win_loss_draw_season = arsenal_games.groupby(['Season', 'result'])['result'].count().unstack().fillna(0)

        # Tracer le graphique
        plt.figure(figsize=(10, 6))

        # Ajouter des barres pour chaque résultat
        plt.bar(win_loss_draw_season.index, win_loss_draw_season['W'], label='Wins')
        plt.bar(win_loss_draw_season.index, win_loss_draw_season['D'], bottom=win_loss_draw_season['W'], label='Draws')
        plt.bar(win_loss_draw_season.index, win_loss_draw_season['L'], bottom=win_loss_draw_season['W']+win_loss_draw_season['D'], label='Losses')

        # Ajouter les chiffres sur les barres
        for i, result in enumerate(win_loss_draw_season.index):
            w = win_loss_draw_season.loc[result, 'W']
            d = win_loss_draw_season.loc[result, 'D']
            l = win_loss_draw_season.loc[result, 'L']
            plt.text(i, w/2, int(w), ha='center', va='center', color='white', fontweight='bold')
            plt.text(i, w+d/2, int(d), ha='center', va='center', color='white', fontweight='bold')
            plt.text(i, w+d+l/2, int(l), ha='center', va='center', color='white', fontweight='bold')

            plt.xlabel('Season')
            plt.ylabel('Number of matches')
            plt.title('Arsenal wins, losses and draws by season')
            plt.xticks(rotation=45)
            plt.legend()

        # Render the plot in Streamlit
        st.pyplot()
        
        st.write(' ')
        st.write("<div style='text-align: justify;'>On peut remarquer que le nombre de victoires est globalement en diminution depuis la saison 2012-2013, avec une légère remontée lors de la saison 2014-2015, avant de baisser à nouveau. Le nombre de défaites suit quant à lui une tendance inverse, en augmentant globalement depuis 2012-2013, avec une forte hausse lors des saisons 2017-2018 et 2018-2019. Enfin, le nombre de matchs nuls est relativement stable, avec quelques variations d'une saison à l'autre.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>On peut également remarquer que la saison 2019-2020 est marquée par un nombre particulièrement élevé de matchs nuls (14), ce qui peut s'expliquer en partie par l'arrêt prématuré du championnat en raison de la pandémie de COVID-19.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>En somme, ce graphique permet de visualiser l'évolution des performances d'Arsenal au cours des dernières saisons, et va nous servir de base pour une analyse plus approfondie des facteurs ayant influencé ces performances.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
        
        data = pd.read_csv('arsenal.csv')
        
        # Grouper les données par équipe et saison, puis appliquer la somme sur les colonnes Team_Goal et Opponent_Goal
        goals = data.groupby(['Team']).agg({'Team_Goal': 'sum', 'Opponent_Goal': 'sum'})

        #Filtrer pour le big 6
        teams_to_keep = ["Arsenal", "Liverpool", "Chelsea", "Manchester City", "Tottenham Hotspur", "Manchester United"]
        Big_6 = goals.loc[goals.index.isin(teams_to_keep)]

        #Ajout d'une colonne path pour les images
        Big_6['path'] = Big_6.index + '.png'

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.scatter(Big_6['Team_Goal'],Big_6['Opponent_Goal'], color='white')

        plt.plot(Big_6['Team_Goal'],Big_6['Opponent_Goal'],"o", color='white')

        # Plot badges
        def getImage(path):
            return OffsetImage(plt.imread('images/' + path), zoom=.05, alpha = 1)

        for index, row in Big_6.iterrows():
            ab = AnnotationBbox(getImage(row['path']), (row['Team_Goal'], row['Opponent_Goal']), frameon=False)
            ax.add_artist(ab)

        # Add average lines
        plt.hlines(Big_6['Opponent_Goal'].mean(), Big_6['Team_Goal'].min(), Big_6['Team_Goal'].max(), color='#c2c1c0')
        plt.vlines(Big_6['Team_Goal'].mean(), Big_6['Opponent_Goal'].min(), Big_6['Opponent_Goal'].max(), color='#c2c1c0')

        # Avg line explanation
        fig.text(.74,.600,'Avg. Goals Against', size=8, color='#c2c1c0')
        fig.text(.340,.17,'Avg. Goals For', size=8, color='#c2c1c0',rotation=90)

        # Title and Axes titles
        fig.text(.15,.98,'Goals Performance since season 10/11',size=15)
        fig.text(.15,.93,'Goals For & Against', size=10)

        ax.set_xlabel("Goals For")
        ax.set_ylabel("Goals Against")

        ax.text(770,475,"Poor attack, poor defense",color="red",size="10")
        ax.text(860,380,"Strong attack, strong defense",color="red",size="10")
        
        # Render the plot in Streamlit
        st.pyplot(fig)
        
        st.write(' ')
        st.write("<div style='text-align: justify;'>Afin d'essayer d'expliquer les résultats d'Arsenal, on se penche sur la qualité de son attaque et de sa défense. On peut voir que Manchester City est l'équipe qui a la meilleure performance globale en termes d'attaque et de défense. Arsenal, en revanche, se situe dans le quadrant supérieur gauche, ce qui signifie que l'équipe a des problèmes à la fois en attaque et en défense. Cela peut être dû à un manque de qualité dans les joueurs de l'équipe ou à une mauvaise stratégie de jeu mise en place par l'entraîneur. En tout cas, cela peut expliquer en partie pourquoi les résultats d'Arsenal ont été moins bons que ceux des autres membres du Big 6 ces dernières années. </div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        def main():
            
            data = pd.read_csv('arsenal.csv')
            
            col1, col2 = st.columns([1, 1])
        
            # Graphique 1
        
            opponent_goals = data.loc[data.Team=='Arsenal']
            opponent_goals['goals_against'] = arsenal.Opponent_Goal
            opponent_goals.drop(['Team_Goal', 'Opponent_Goal'], axis=1, inplace=True)
            opponent_goals = opponent_goals.groupby('season').sum()
            opponent_goals = opponent_goals[['goals_against']]
        
            # Créer un graphique en barre colorée avec Seaborn
            plt.figure(figsize=(14,10))
            sns.set_style("whitegrid")
            sns.barplot(x=opponent_goals.index, y="goals_against", data=opponent_goals, palette="Set2")
        
            # Ajouter un titre et des labels pour les axes
            plt.title("Goals Conceded by Season")
            plt.xlabel("Season")
            plt.ylabel("Goals Conceded")
            
            plt.text(0, 44, "Arsène Wenger", fontsize=10, fontweight='bold', ha='center')
            plt.text(1, 50, "Arsène Wenger", fontsize=10, fontweight='bold', ha='center')
            plt.text(2, 39, "Arsène Wenger", fontsize=10, fontweight='bold', ha='center')
            plt.text(3, 42, "Arsène Wenger", fontsize=10, fontweight='bold', ha='center')
            plt.text(4, 38, "Arsène Wenger", fontsize=10, fontweight='bold', ha='center')
            plt.text(5, 37, "Arsène Wenger", fontsize=10, fontweight='bold', ha='center')
            plt.text(6, 45, "Arsène Wenger", fontsize=10, fontweight='bold', ha='center')
            plt.text(7, 52, "Unai Emery", fontsize=10, fontweight='bold', ha='center')
            plt.text(8, 52, "Unai Emery", fontsize=10, fontweight='bold', ha='center')
            plt.text(9, 49, "Unai Emery \n \ Mikel Arteta", fontsize=10, fontweight='bold', ha='center')
            plt.text(10, 40, "Mikel Arteta", fontsize=10, fontweight='bold', ha='center')
        
            col1.pyplot()
            
            def add_manager_column(data):
                # Filter for Arsenal in Team
                arsenal_goals = data[data['Team'] == 'Arsenal'].copy()
    
                # Convert date column to datetime
                arsenal_goals['date'] = pd.to_datetime(arsenal_goals['date'])
    
                # Add manager column based on date
                arsenal_goals['manager'] = pd.cut(arsenal_goals['date'], 
                                       bins=[pd.Timestamp('1900-01-01'), pd.Timestamp('2018-05-13'), pd.Timestamp('2018-08-12'), pd.Timestamp('2019-11-23'), pd.Timestamp('2100-01-01')], 
                                       labels=['Arsene Wenger', 'Unai Emery', 'Unai Emery', 'Mikel Arteta'], 
                                       include_lowest=True, ordered=False)
    
                return arsenal_goals
    
            arsenal_goals = add_manager_column(data)

            # Group by manager and season, and count the number of goals
            goals_by_manager_season = arsenal_goals.groupby(['manager', 'season'])['Team_Goal'].sum().reset_index()

            # Pivot the table to have managers as columns
            goals_by_manager_season_pivot = goals_by_manager_season.pivot(index='season', columns='manager', values='Team_Goal')

            # Create a bar plot with stacked bars
            plt.figure(figsize=(14,10))
            ax = goals_by_manager_season_pivot.plot(kind='bar', stacked=True, color=['#DB0007', '#063672', '#9C824A'])

            # Add axis labels and a title
            plt.xlabel('Season')
            plt.ylabel('Number of goals')
            plt.title('Number of goals scored by manager and season')

            # Modify the x-axis labels to display the seasons
            start_year = 2010
            tick_labels = [f"{str(start_year + i)[-2:]}/{str(start_year + i + 1)[-2:]}" for i in range(len(goals_by_manager_season_pivot.index))]
            ax.set_xticklabels(tick_labels)

            # Add labels to the bars
            for container in ax.containers:
                ax.bar_label(container)
            
            col2.pyplot()
        
        if __name__ == "__main__":
            main()   

        st.write(' ')
        st.write(' ')
        st.write("<div style='text-align: justify;'>On peut voir que les performances défensives sont variables d'une saison à l'autre, avec des pics à la hausse ou à la baisse. On remarque également que la période de l'entraîneur Arsène Wenger est plutôt marquée par une stabilité des résultats sur le plan défensif, tandis que la période d'Unai Emery est marquée par une augmentation des buts encaissés. On peut aussi noter que Mikel Arteta semble avoir redressé la situation depuis sa prise de poste, avec une diminution significative du nombre de buts encaissés en 2020-2021. On peut observer que le nombre de buts marqués par saison est assez similaire entre Arsène Wenger et Unai Emery, avec une diminution sous la direction de Mikel Arteta.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
        
        data = pd.read_csv('arsenal.csv')
        
        # Sélectionner les équipes et les saisons d'intérêt
        teams = ['Arsenal', 'Manchester United', 'Manchester City', 'Liverpool', 'Tottenham Hotspur', 'Chelsea']
        seasons = data['season'].unique()

        # Créer un sous-ensemble de données avec les moyennes des tirs cadrés par saison et par équipe
        avg_shots_on_target = data.groupby(['Team', 'season'])['home_shots_on_target'].mean().reset_index()

        # Définir les couleurs pour chaque équipe
        team_colors = {'Arsenal': "#EF0107", 'Manchester United': '#DA291C', 'Manchester City': "#6CABDD",
               'Liverpool': "#C8102E", 'Tottenham Hotspur': '#132257', 'Chelsea': "#034694"}

        # Charger les images correspondantes à chaque équipe
        team_images = {'Arsenal': 'images/Arsenal.png', 'Manchester United': 'images/Manchester United.png',
               'Manchester City': 'images/Manchester City.png', 'Liverpool': 'images/Liverpool.png',
               'Tottenham Hotspur': 'images/Tottenham Hotspur.png', 'Chelsea': 'images/Chelsea.png'}

        # Définir la fonction pour charger les images
        def load_image(path):
            return plt.imread(path)

        # Définir la fonction d'affichage
        def display_image(ax, image_path, xy):
            image = load_image(image_path)
            im = OffsetImage(image, zoom=0.05)
            ab = AnnotationBbox(im, xy, xybox=(0., 0.), xycoords='data', boxcoords="offset points", frameon=False)
            ax.add_artist(ab)

        # Définir le graphique
        fig, ax = plt.subplots(figsize=(10, 6))

        # Afficher les lignes pour chaque équipe
        for team in teams:
            team_data = avg_shots_on_target[avg_shots_on_target['Team'] == team]
            team_color = team_colors[team]
            if len(team_data) > 0:
                ax.plot(team_data['season'], team_data['home_shots_on_target'], label=team, color=team_color)
        
                # Afficher l'image de l'équipe
                last_season = team_data['season'].iloc[-1]
                last_shots_on_target = team_data['home_shots_on_target'].iloc[-1]
                team_image_path = team_images[team]
                display_image(ax, team_image_path, (last_season, last_shots_on_target))
    
        # Afficher la légende et les titres
        ax.set_title('Shots on target per season and per team')
        ax.set_xlabel('Season')
        ax.set_ylabel('Shots on target')
        ax.legend()
        
        # Render the plot in Streamlit
        st.pyplot(fig)
        
        st.write(' ')
        st.write(' ')        
        st.write("<div style='text-align: justify;'>Ce graphique représente le nombre de tirs cadrés par saison pour le Big 6. Chaque ligne représente une équipe différente et est colorée avec la couleur de leur maillot. On peut voir que le nombre de tirs cadrés varie considérablement d'une saison à l'autre pour chaque équipe. En ce qui concerne Arsenal, on peut constater que le nombre de tirs cadrés est en baisse depuis la saison 2010/2011, avec une exception notable lors de la saison 2017-2018. Cela nous montre que l'équipe est devenue au fil du temps moins performante, ou peut-être que son style de jeu a changé. En tout cas, ce graphique permet de mieux comprendre les performances d'Arsenal et de ses concurrents en termes d'efficacité devant le but.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
    
        # Sélectionner les équipes d'intérêt
        teams = ['Arsenal', 'Manchester United', 'Manchester City', 'Liverpool', 'Tottenham Hotspur', 'Chelsea']

        # Créer un sous-ensemble de données avec les moyennes des passes par saison et par équipe
        avg_home_passes = data.groupby(['Team', 'season'])['home_passes'].mean().reset_index()
        avg_away_passes = data.groupby(['Opponent', 'season'])['away_passes'].mean().reset_index()

        # Renommer les colonnes pour les équipes à l'extérieur pour qu'elles correspondent aux équipes à domicile
        avg_away_passes = avg_away_passes.rename(columns={'Opponent': 'Team', 'away_passes': 'home_passes'})

        # Fusionner les données des passes à domicile et à l'extérieur pour chaque équipe et chaque saison
        avg_passes = pd.concat([avg_home_passes, avg_away_passes], ignore_index=True)

        # Sélectionner les données pour les équipes d'intérêt
        avg_passes = avg_passes[avg_passes['Team'].isin(teams)]

        # Regrouper les données par saison et calculer la moyenne des passes pour chaque équipe
        season_avg_passes = avg_passes.groupby(['season', 'Team'])['home_passes'].mean().reset_index()

        # Modifier le nom de la colonne "home_passes" pour la renommer en "Average Passes"
        season_avg_passes = season_avg_passes.rename(columns={'home_passes': 'Average Passes'})

        # Définir les couleurs pour chaque équipe
        team_colors = {'Arsenal': "#EF0107", 'Manchester United': '#DA291C', 'Manchester City': "#6CABDD",
               'Liverpool': "#C8102E", 'Tottenham Hotspur': '#132257', 'Chelsea': "#034694"}

        # Créer le graphique en barres
        fig = px.bar(season_avg_passes, x='season', y='Average Passes', color='Team', color_discrete_map=team_colors,
                     category_orders={'Team': ['Arsenal', 'Manchester United', 'Manchester City', 'Liverpool', 'Tottenham Hotspur', 'Chelsea']},
                     barmode='group')

        # Ajouter un titre et des titres d'axes
        fig.update_layout(title='Average Passes per Season and per Team', xaxis_title='Season', yaxis_title='Average Passes')

        # Render the plot in Streamlit
        st.plotly_chart(fig)
              
        st.write("<div style='text-align: justify;'>Ce graphique représente les performances du Big 6 en termes de passes moyennes par saison. Nous pouvons observer une tendance à la baisse sur les 2 années de management d'Unai Emery, et semble remonter avec la prise de poste de Mikel Arteta. Mais cette diminution pourrait être due à des facteurs tels que des changements de style de jeu, des blessures ou des variations de performance individuelle. Nous allons regarder la possession pour avoir une idée plus précise.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')        
        
        arsenal_data_home = data[data['Team'] == 'Arsenal']
        arsenal_data_away = data[data['Opponent'] == 'Arsenal']
        
        # Créer une figure avec deux sous-graphes
        fig, axs = plt.subplots(1, 2, figsize=(12,6))

        # Tracer le premier graphique
        axs[0].set_title('Arsenal: Home possession over the seasons')
        sns.boxplot(data=arsenal_data_home, x='season', y='home_possession', ax=axs[0])

        # Tracer le deuxième graphique
        axs[1].set_title('Arsenal: Away possession over the seasons')
        sns.boxplot(data=arsenal_data_away, x='season', y='away_possession', ax=axs[1])
        
        st.pyplot(fig)
        
        st.write("<div style='text-align: justify;'>La possession de balle est une statistique importante dans le football, car elle mesure le pourcentage de temps qu'une équipe passe en possession de la balle pendant un match. Dans ce graphique en boîte, nous pouvons voir que la possession de balle d'Arsenal à domicile est généralement plus élevée que sa possession à l'extérieur. le graphique montre également que la possession de balle d'Arsenal varie considérablement d'une saison à l'autre, avec des hauts et des bas. Cela peut être dû à plusieurs facteurs tels que les changements dans la composition de l'équipe, les tactiques de l'entraîneur, ou la qualité des adversaires. Mais cette possession n'est pas significative dans l'explication de la baisse des performances de l'équipe.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>Exploration des transferts</span>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Pour aller plus loin dans notre analyse, nous pouvons également examiner les données financières d'Arsenal, notamment les dépenses de transfert des joueurs. Cela nous permettra de voir si le club a investi suffisamment dans son effectif pour pouvoir rivaliser avec les autres membres du Big 6 et si cela a eu un impact sur les performances sur le terrain.</div>", unsafe_allow_html=True)
        st.write(' ')
        
        totals = pd.read_csv('totals.csv')        
        
        # Filtrer les saisons de 10/11 à 20/21
        filtered_totals = totals[totals["Season"] <= "20/21"]

        # Préparer les données pour le graphique
        arrivals = filtered_totals[filtered_totals["Transfer Type"] == "Arrivées"].set_index("Season")["Transfer Sum"]
        departures = filtered_totals[filtered_totals["Transfer Type"] == "Départs"].set_index("Season")["Transfer Sum"]

        # Créer un graphique à barres côte à côte
        fig, ax = plt.subplots(figsize=(8, 9))

        seasons = arrivals.index
        bar_width = 0.4
        bar_positions = np.arange(len(seasons))

        ax.bar(bar_positions - bar_width / 2, arrivals / 1e6, width=bar_width, label="Arrivals", color="#DB0007")
        ax.bar(bar_positions + bar_width / 2, departures / 1e6, width=bar_width, label="Departures", color="#063672")

        # Configurer les étiquettes et le titre
        ax.set_ylabel("Transfer Amount (million €)")
        ax.set_xlabel("Season")
        ax.set_title("Transfer Amounts per Season and Type")
        plt.xticks(bar_positions, seasons, rotation=45)

        # Ajouter les montants sur les barres
        for i, (arr, dep) in enumerate(zip(arrivals / 1e6, departures / 1e6)):
            ax.text(i - bar_width / 2, arr + 0.5, f"{arr:.1f}", ha="center", fontsize=9)
            ax.text(i + bar_width / 2, dep + 0.5, f"{dep:.1f}", ha="center", fontsize=9)

        # Ajouter une légende
        ax.legend()
        
        # Créer un DataFrame pour la balance des transferts
        df_balance = pd.DataFrame({"Season": seasons, "Balance": arrivals - departures})

        # Créer un graphique Polar Bar Chart
        fig2 = px.bar_polar(df_balance, r="Balance", theta="Season", color="Balance",
                           color_continuous_scale=px.colors.sequential.Plasma, template="plotly_dark",
                           range_color=(-max(abs(df_balance['Balance'])), max(abs(df_balance['Balance']))))

        # Configurer le titre et la taille
        fig2.update_layout(title="Transfer Balance per Season (Polar Bar Chart)",
                           width=500, height=500)  # Modifier les valeurs de width et height pour ajuster la taille

        # Créer deux colonnes pour afficher les graphiques côte à côte
        col1, col2 = st.columns(2)
        
        # Afficher les graphiques dans les colonnes respectives
        with col1:
            st.pyplot(fig)

        with col2:
            st.plotly_chart(fig2)
            
        st.write("<div style='text-align: justify;'>Ces graphiques montrent les dépenses de transfert d'Arsenal au fil des saisons. On peut constater que les dépenses ont considérablement augmenté ces dernières années, avec des montants de plus en plus importants investis dans l'achat de nouveaux joueurs. Cela pourrait indiquer que le club a cherché à renforcer son effectif pour améliorer ses performances sur le terrain, mais cela n'a pas nécessairement eu l'effet escompté, comme le montrent les résultats précédemment analysés.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("<div style='text-align: justify;'>En conclusion, l'histoire récente d'Arsenal montre une série de hauts et de bas en termes de performances sur le terrain, avec des fluctuations dans le classement et le nombre de buts marqués. Plusieurs facteurs pourraient expliquer ces performances, notamment les changements de management, les blessures de joueurs clés, la qualité de l'effectif et la stratégie de transfert. Cela soulève la question de savoir si le club doit revoir sa stratégie d'investissement et s'il doit se concentrer sur d'autres aspects, tels que la formation de jeunes joueurs et la fidélisation des joueurs clés plutôt que de chercher constamment à acheter de nouveaux talents.</div>", unsafe_allow_html=True)
        
        
########################################
# Machine Learning
########################################
if selected == "Machine Learning":
    # Header
    with header:
        st.markdown("<h1 style='text-align: center; color: #DB0007;'>Machine Learning</h1>", unsafe_allow_html=True)
        
    # Intro
    with intro:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write("<div style='text-align: justify;'>Avec nos données, nous nous sommes demandés si elles pouvaient être utilisées pour entraîner un modèle de machine learning afin d'analyser la performance de l'équipe lors des différentes saisons étudiées.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>Données</span>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Afin de pouvoir entrainer notre modèle, nous avons du déterminer une variable cible. Pour ce faire nous avons ajouté une colonne Target à notre jeu de données.</div>", unsafe_allow_html=True)
        st.write(' ')
        
        group = pd.read_csv('arsenal.csv')
        
        # Ajout d'une colonne "result" avec le vainqueur du match ou match nul

        group['result'] = np.nan
        for i in range(len(group['Team'])):
            if (group['Team_Goal'][i]>group['Opponent_Goal'][i]):
                group['result'][i] = 'H'
            elif (group['Team_Goal'][i]<group['Opponent_Goal'][i]):
                group['result'][i] = 'A'
            elif(group['Team_Goal'][i]==group['Opponent_Goal'][i]):
                group['result'][i] = 'D'

        # Définir la fonction de transformation
        def get_target(result):
            if result == "D":
                return 0
            elif result == "H":
                return 1
            elif result == "A":
                return 2

        # Appliquer la fonction à chaque ligne du dataframe et créer la colonne "target"
        group["target"] = group["result"].apply(get_target)

        group = group.drop(['goal_home_ht','goal_away_ht','result','result_full','result_ht'], axis=1)
        
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe avec la target") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=group.shape[0])
            st.dataframe(group.head(line_to_plot))
              
        st.write("<div style='text-align: justify;'>Pour pouvoir lisser les données, nous avons ajouter des moyennes roulantes pour chaque caractéristique en utilisant une fenêtre de taille 3, c'est-à-dire qu'elle calcule la moyenne des trois valeurs précédentes, incluant la valeur actuelle</div>", unsafe_allow_html=True)
        st.write(' ')
        
        def rolling_averages(group, cols, new_cols):
            group = group.sort_values("date")
            rolling_stats = group[cols].rolling(3, closed='left').mean()
            group[new_cols] = rolling_stats
            group = group.dropna(subset=new_cols)
            return group
        
        cols = ["Team_Goal", "Opponent_Goal", "home_clearances", "home_corners", "home_fouls_conceded", "home_offsides",
        "home_passes", "home_possession", "home_shots", "home_shots_on_target", "home_tackles", "home_touches",
       "away_clearances", "away_corners", "away_fouls_conceded", "away_offsides",
        "away_passes", "away_possession", "away_shots", "away_shots_on_target", "away_tackles", "away_touches"]
        new_cols = [f"{c}_rolling" for c in cols]

        rolling_averages(group, cols, new_cols)
        
        matches_rolling = group.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols))

        matches_rolling = matches_rolling.droplevel('Team')

        matches_rolling.index = range(matches_rolling.shape[0])
        
        ml = pd.read_csv('matches_rolling.csv')
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe avec les moyennes roulantes") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=ml.shape[0])
            st.dataframe(ml.head(line_to_plot))
        
        st.write("<div style='text-align: justify;'>Nous avons utilisé la matrice de corrélation pour extraire les variables ayant une corrélation supérieure ou inférieure à un certain seuil avec la variable cible, puis identifié les couples de variables ayant une corrélation supérieure ou inférieure à un certain seuil. Ensuite, nous avons créé une matrice de corrélation filtrée en fonction d'un seuil de corrélation, afin d'obtenir une vue d'ensemble des relations entre les variables. Enfin, nous avons éliminé certaines colonnes de données à partir de la matrice des matchs, à savoir celles ayant une forte corrélation avec d'autres colonnes ou celles qui ne sont pas considérées comme pertinentes pour la prédiction de la variable cible.</div>", unsafe_allow_html=True)
        st.write(' ') 
        
        ### Showing heatmap
        if st.checkbox("Afficher la matrice de corrélation"):
            # Définir la taille de la figure
            plt.figure(figsize=(16, 10))

            # Appliquer un masque pour le triangle supérieur
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Filtrer les corrélations faibles en fonction d'un seuil (0.75)
            threshold = 0.75
            mask_weak_corr = np.abs(corr_matrix) > threshold
            filtered_corr_matrix = corr_matrix * mask_weak_corr

            # Créer la heatmap avec Seaborn
            heatmap = sns.heatmap(filtered_corr_matrix, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
            heatmap.set_title('Matrice de corrélation', fontdict={'fontsize': 18}, pad=16)
            
        pycaret = pd.read_csv('df_pycaret.csv')
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe pour le machine learning") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=pycaret.shape[0])
            st.dataframe(pycaret.head(line_to_plot))
        
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>Pycaret</span>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Au cours de nos recherches, nous avons découvert une bibliothèque de machine learning open-source en Python, basée sur une approche low-code, nommée <a href='https://pycaret.org/' target='_blank'>PyCaret</a>. Celle-ci permet de créer des modèles de machine learning plus rapidement et efficacement en utilisant moins de code. Elle automatise les workflows de machine learning et simplifie le processus d'expérimentation en remplaçant des centaines de lignes de code par seulement quelques lignes. PyCaret est une alternative 'low-code' qui vous permet de tirer parti de plusieurs bibliothèques de machine learning en utilisant une interface simplifiée et cohérente. En somme, PyCaret vous aide à accélérer et à simplifier le processus de développement de modèles de machine learning.</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.image("Pycaret1.jpg", width=200)

        with col2:
            st.image("Pycaret3.jpg", width=400)
            
        # Affichage de la liste des modèles retenus
        st.write("<div style='text-align: justify;'>L'utilisation de la bibliothèque pycaret nous a permis de retenir trois modèles sur lesquels nous pouvons choisir de concentrer notre analyse:</div>", unsafe_allow_html=True)
        models_list = ["Linear Discriminant Analysis", "Logistic Regression", "Ridge Classifier"]
        for model in models_list:
            st.write(f"- {model}")
       
        st.write(' ')        
        st.write(f"<span style='font-weight:bold;color:#063672;font-size:24px;'>Etude des modèles</span>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Pour étudier la significativité des variables catégorielles dans la prédiction du churn, nous avons utilisé le test de chi2. Nous avons calculé le coefficient de V de Cramer pour chaque variable catégorielle et avons trié les résultats par ordre décroissant de V de Cramer. Les p-valeurs étant < 5% on contate que les variables Team , Opponent , Venue et Season sont significatives.</div>", unsafe_allow_html=True)
        st.write(' ') 
        
        ### Test de chi2
        data_corr = pd.read_csv('df_pycaret.csv', index_col=0)
        data_corr['date']= pd.to_datetime(data_corr['date'])
        cat_vars= data_corr.select_dtypes(include = ['object','datetime64']).columns
        num_vars = data_corr.select_dtypes(include = ["float", "int",'datetime64']).columns
        
        from scipy.stats import chi2_contingency
        #variables catégorielles dans cat_vars
        # Initialiser les listes pour stocker les résultats
        var_names = []
        chi2_stats = []
        p_values = []
        cramer_vs = []

        # Parcourir toutes les variables catégorielles
        for var in cat_vars:
            # Calculer le tableau de contingence
            contingency_table = pd.crosstab( data_corr['target'],  data_corr[var])
            # Calculer la statistique de test du Chi-deux et la p-valeur
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            # Calculer le coefficient V de Cramer
            n = contingency_table.sum().sum()
            phi2 = chi2/n
            r,k = contingency_table.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rc = r-((r-1)**2)/(n-1)
            kc = k-((k-1)**2)/(n-1)
            cramer_v = np.sqrt(phi2corr/min(rc-1,kc-1))
            # Ajouter les résultats aux listes correspondantes
            var_names.append(var)
            chi2_stats.append(chi2)
            p_values.append(p)
            cramer_vs.append(cramer_v)

        # Créer un DataFrame avec les résultats
        results_df = pd.DataFrame({
            'Variable': var_names,
            'Chi2': chi2_stats,
            'P-valeur': p_values,
            'V de Cramer': cramer_vs
        })

        # Trier le DataFrame par ordre croissant de V de Cramer
        results_df.sort_values(by='V de Cramer', inplace=True,ascending=False)

        # Afficher le tableau des résultats
        st.write(results_df)
        
        ###test de kruskal-Wallis
        
        # Initialiser les listes pour stocker les résultats
        var_names = []
        kw_stats = []
        p_values = []
        
        # Parcourir toutes les variables numériques
        for var in num_vars:
            # Calculer les groupes de valeurs
            groups = [data_corr[data_corr['target'] == 0][var], data_corr[data_corr['target'] == 1][var], data_corr[data_corr['target'] == 2][var]]
            # Appliquer le test de Kruskal-Wallis
            kw_stat, p = stats.kruskal(*groups)
            # Ajouter les résultats aux listes correspondantes
            var_names.append(var)
            kw_stats.append(kw_stat)
            p_values.append(p)

        # Créer un DataFrame avec les résultats
        results_df = pd.DataFrame({
            'Variable': var_names,
            'Kruskal-Wallis': kw_stats,
            'P-valeur': p_values
        })

        # Trier le DataFrame par ordre croissant de p-valeur
        results_df.sort_values(by='P-valeur', inplace=True)
        
        st.write("<div style='text-align: justify;'>Le test de Kruskal-Wallis a été appliqué à toutes les variables numériques pour évaluer si les différences de moyennes entre les groupes étaient statistiquement significatives. Nous constatons pour ce test que nos variables sont pertinentes sauf : 'home_tackles_rolling' ,'away_fouls_conceded_rolling', 'home_tackles' , 'away_tackles' ,'away_clearances_rolling'.</div>", unsafe_allow_html=True)
        st.write(' ') 
        st.write(results_df)
        
        #Suppression des colonnes non pertinentes de notre dataset
        data_corr = data_corr.drop(['home_tackles_rolling' ,'away_fouls_conceded_rolling', 'home_tackles' , 'away_tackles' ,'away_clearances_rolling'], axis=1)
        
        st.write("<span style='font-weight:bold;color:#063672;font-size:24px;'>Encodage des données</span>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Afin d'entrainer notre modèle, nous nous sommes occupés des variables qualitatives. Pour cela, nous les avons regroupées afin de diminuer le nombre de colonnes utilisées par le modèle. Nous avons également regroupé les dates en parties de saison, en suivant les règles suivantes :</div>", unsafe_allow_html=True)
        st.markdown("* Si le mois est compris entre août et octobre, nous le classerons dans la partie 1 (P1).\n* Si le mois est compris entre novembre et décembre, nous le classerons dans la partie 2 (P2).\n* Si le mois est compris entre janvier et mars, nous le classerons dans la partie 3 (P3).\n* Si le mois est compris entre avril et mai, nous le classerons dans la partie 4 (P4).\n* Si le mois est compris entre juin et juillet, nous le classerons dans la partie 5 (P5).")
        st.write(' ')
        
        data_corr['Month'] = data_corr['date'].dt.month
        categ=["P3","P4","P5","P1","P2"]
        bins= [1,3,6,8,10,12]
        data_corr["Decoupage_date"]= pd.cut(data_corr['Month'],bins,labels=categ,include_lowest=True)
        
        #Regroupement des équipes à domicile en 3 chapeaux en fonction de l'occurence des matchs joués
        A = [cat for cat in data_corr['Team'].value_counts().sort_values(ascending = False).head(13).index]
        B = [cat for cat in data_corr['Team'].value_counts().sort_values(ascending = False).tail(10).index]
        list =[None]*8249
        for i in range(0,8249):
            if data_corr['Team'][i] in A: 
                list[i] = "Chap1" 
            elif data_corr['Team'][i] in B: 
                list[i]= "Chap3" 
            else:
                list[i]= "Chap2"
        
        data_corr['Team_dom']=list
        
        #Regroupement des équipes à l'extérieur en 3 chapeaux en fonction de l'occurence des matchs joués
        C = [cat for cat in data_corr['Opponent'].value_counts().sort_values(ascending = False).head(13).index]
        D = [cat for cat in data_corr['Opponent'].value_counts().sort_values(ascending = False).tail(10).index]
        list2 =[None]*8249
        for i in range(0,8249):
            if data_corr['Opponent'][i] in C: 
                list2[i] = "Chap1" 
            elif data_corr['Opponent'][i] in D: 
                list2[i]= "Chap3" 
            else:
                list2[i]= "Chap2"
        
        data_corr['Team_ext']=list2

        # Affichage des tableaux côte à côte
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(data_corr["Decoupage_date"].value_counts())

        with col2:
            st.write(data_corr['Team_dom'].value_counts())

        with col3:
            st.write(data_corr['Team_ext'].value_counts())
            
        st.write("<div style='text-align: justify;'>Suite aux regroupements effectués nous avons supprimé les colonnes date, Team, Opponent et Month.</div>", unsafe_allow_html=True)
        st.write(' ')
        
        data_corr= data_corr.drop(['date','Team','Opponent','Month'], axis=1)
        
        #On conserve les K-1 catégories de nos variables.
        season = pd.get_dummies(data_corr['season'], drop_first = True)
        
        venue = pd.get_dummies(data_corr['Venue'],drop_first=True)

        Decoupage_date =  pd.get_dummies(data_corr['Decoupage_date'])
        Decoupage_date =  Decoupage_date.drop('P5', axis=1)

        Team_dom = pd.get_dummies(data_corr['Team_dom'],prefix= 'Dom')
        Team_dom =  Team_dom.drop('Dom_Chap3', axis=1)

        Team_ext = pd.get_dummies(data_corr['Team_ext'],prefix='Ext')
        Team_ext = Team_ext.drop('Ext_Chap3', axis=1)
        
        # on supprime les anciennes catégories
        df_dum = data_corr.drop(['season', 'Venue', 'Decoupage_date', 'Team_dom', 'Team_ext'],axis=1)
        # on concatène avec les datasets résultant de l'encodage
        df_dum = pd.concat([df_dum,season,venue,Decoupage_date,Team_dom,Team_ext],axis =1)
        

        st.write("<div style='text-align: justify;'>Nous avons encodé les variables catégorielles en utilisant la méthode de <i>one-hot encoding</i> afin de les transformer en variables numériques exploitables par notre modèle.</div>", unsafe_allow_html=True)
        st.markdown("* La variable 'season' a été encodée en 2 catégories.\n* La variable 'Venue' a été encodée en 1 catégorie.\n* La variable 'Decoupage_date' a été encodée en 4 catégories.\n* Les variables 'Team_dom' et 'Team_ext' ont été encodées en 2 catégories chacune.")
        st.write("<div style='text-align: justify;'>Le nouveau jeu de données est alors composé de {} colonnes et {} lignes.</div>".format(df_dum.shape[1], df_dum.shape[0]), unsafe_allow_html=True)
        st.write(' ')
        
        ### Showing the data
        if st.checkbox("Afficher le Dataframe après encodage") :
            line_to_plot = st.slider("Selectionner le nombre de lignes à afficher", min_value=3, max_value=df_dum.shape[0])
            st.dataframe(df_dum.head(line_to_plot))
            
        # Séparation et mise à l'échelle des données
        X= df_dum.drop('target',axis =1)
        y= df_dum.target
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 45)

        # mise à l'échelle des variables quantitatives 
        # ces variables seront repérées dans num_train et num_test

        cat_train = X_train[X_train.columns[30:]]
        cat_test = X_test[X_test.columns[30:]]

        # instanciation de StandardScaler 
        scaler = StandardScaler()

        num_train = pd.DataFrame(scaler.fit_transform(X_train[X_train.columns[0:30]]),columns = X_train.columns[0:30], index= X_train.index)
        num_test = pd.DataFrame(scaler.transform(X_test[X_test.columns[0:30]]), columns = X_test.columns[0:30],index= X_test.index )

        # on regroupe les 2 dataframes num et cat 
        X_train_sc = pd.concat([num_train,cat_train], axis=1)
        X_test_sc = pd.concat([num_test,cat_test ], axis=1)

        # Affichage du résultat
        st.write("<div style='text-align: justify;'>Nous avons ensuite séparé notre jeu de données en jeu d'entrainement et jeu de test avec respectivement 70% et 30% des données, puis nous avons mis à l'échelle les variables quantitatives en utilisant la méthode StandardScaler.</div>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>Le nouveau jeu de données est composé de {} colonnes et {} lignes.</div>".format(X_train_sc.shape[1], X_train_sc.shape[0]), unsafe_allow_html=True)
        st.write(' ')
        
        if st.checkbox("Afficher les données après séparation et mise à l'échelle"):
            st.write("X_train_sc :", X_train_sc.shape)
            st.dataframe(X_train_sc.head())
            st.write("y_train :", y_train.shape)
            st.dataframe(y_train.head())
            st.write("X_test_sc :", X_test_sc.shape)
            st.dataframe(X_test_sc.head())
            st.write("y_test :", y_test.shape)
            st.dataframe(y_test.head())
            
        st.write("<span style='font-weight:bold;color:#063672;font-size:24px;'>Entrainement des modèles</span>", unsafe_allow_html=True)   
        st.write("<span style='font-weight:bold;color:#EF0107;font-size:18px;'>Ridge Classifier</span>", unsafe_allow_html=True)
        
        # Entrainement du Ridge Classifier

        # On instancie le modèle
        ridge_clf= RidgeClassifier()
        ridge_clf.fit(X_train_sc, y_train)

        # On évalue le modèle
        y_pred1 = ridge_clf.predict(X_test_sc)
        confusion_matrix = pd.crosstab(y_test, y_pred1, rownames=['Realité'], colnames=['Prédiction'])
        classification_report_str = classification_report(y_test, y_pred1)

        # Affichage de la matrice de confusion et du rapport de classification côte à côte
        columns = st.columns(2)
        with columns[0]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Matrice de confusion</span>", unsafe_allow_html=True)
            st.table(confusion_matrix)
        with columns[1]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Rapport de classification</span>", unsafe_allow_html=True)
            st.text(classification_report_str)
            
        st.write("<div style='text-align: justify;'>Après ce premier entrainement, nous constatons que la classe 0 (match nul) est mal représentée.</div>", unsafe_allow_html=True)
        
        st.write(' ')             
        st.write("<span style='font-weight:bold;color:#EF0107;font-size:18px;'>Régression Logistique</span>", unsafe_allow_html=True)
        st.write(' ') 
        
        # Entrainement du la regression logistique
        
        # On instancie le modèle
        reglog= LogisticRegression()
        reglog.fit(X_train_sc, y_train)

        # On évalue le modèle
        y_pred2 = reglog.predict(X_test_sc)
        confusion_matrix = pd.crosstab(y_test,y_pred2, rownames=['Realité'], colnames=['Prédiction'])
        classification_report_str = classification_report(y_test, y_pred2)
        
        # Affichage de la matrice de confusion et du rapport de classification côte à côte
        
        columns = st.columns(2)
        with columns[0]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Matrice de confusion</span>", unsafe_allow_html=True)
            st.table(confusion_matrix)
        with columns[1]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Rapport de classification</span>", unsafe_allow_html=True)
            st.text(classification_report_str)
        
        st.write("<div style='text-align: justify;'>Dans ce modèle, la classe 0 (match nul) est mieux représentée.</div>", unsafe_allow_html=True)
        st.write(' ') 
            
        # Optimisation en utilisant un RFECV

        st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Optimisation en utilisant un RFECV</span>", unsafe_allow_html=True)
        st.write(' ')
        
        @st.cache_data
        def rfe_cv():
            #Test du RFECV (Recursive Feature Elimination with Cross-validation)
            rfe = RFECV(reglog, step=1, cv=5, scoring='accuracy')
            rfe.fit(X_train_sc, y_train)
            return rfe

        # Appel de la fonction cachee
        rfe = rfe_cv()
        
        st.write("<div style='text-align: justify;'>Nous avons appliqué la méthode Recursive Feature Elimination with Cross-Validation (RFECV), qui est une technique de sélection de caractéristiques basée sur la récursivité. Cette méthode vise à sélectionner les caractéristiques les plus pertinentes pour un modèle d'apprentissage automatique. Grâce à cette approche, nous avons identifié les variables les plus importantes pour notre jeu de données.</div>", unsafe_allow_html=True)
        st.write(' ')
        st.write("Meilleures caractéristiques:", X_train_sc.columns[np.where(rfe.support_)[0]])
        st.write("Meilleur score: 0.665800865")
        
        # On evalue la regression logistique avec les features du rfecv

        X_rfe= X_train_sc[['home_clearances', 'home_passes', 'home_red_cards', 'home_shots_on_target', 'away_clearances', 'away_passes', 'away_red_cards', 'away_shots_on_target', 'home_shots_on_target_rolling', '13/14', '14/15', '16/17', '18/19', 'Home', 'Dom_Chap1', 'Ext_Chap1', 'Ext_Chap2']]

        X_test_rfe= X_test_sc[['home_clearances', 'home_passes', 'home_red_cards', 'home_shots_on_target', 'away_clearances', 'away_passes', 'away_red_cards', 'away_shots_on_target', 'home_shots_on_target_rolling', '13/14', '14/15', '16/17', '18/19', 'Home', 'Dom_Chap1', 'Ext_Chap1', 'Ext_Chap2']]

        reglog_rfe= LogisticRegression()
        reglog_rfe.fit(X_rfe, y_train)
        
        # On évalue le modèle

        y_pred_rfe = reglog_rfe.predict(X_test_rfe)

        columns = st.columns(2)
        with columns[0]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Matrice de confusion</span>", unsafe_allow_html=True)
            confusion_matrix = pd.crosstab(y_test,y_pred_rfe, rownames=['Realité'], colnames=['Prédiction'])
            st.table(confusion_matrix)
        with columns[1]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Rapport de classification</span>", unsafe_allow_html=True)
            classification_report_rfe = classification_report(y_test, y_pred_rfe)
            st.text(classification_report_rfe)
            
        st.write(' ')             
        st.write("<span style='font-weight:bold;color:#EF0107;font-size:18px;'>Régression Logistique avec poids de classe équilibrés</span>", unsafe_allow_html=True)
        st.write(' ') 

        # Entrainement de la régression logistique avec poids de classe équilibrés

        # On instancie le modèle avec les paramètres donnés
        reglog2 = LogisticRegression(C=0.14285714285714285, solver='newton-cg', class_weight='balanced')
        reglog2.fit(X_rfe, y_train)

        # On évalue le modèle
        y_pred_rfe2 = reglog2.predict(X_test_rfe)
        confusion_matrix2 = pd.crosstab(y_test, y_pred_rfe2, rownames=['Realité'], colnames=['Prédiction'])
        classification_report_str2 = classification_report(y_test, y_pred_rfe2)

        # Affichage de la matrice de confusion et du rapport de classification côte à côte
        columns2 = st.columns(2)
        with columns2[0]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Matrice de confusion</span>", unsafe_allow_html=True)
            st.table(confusion_matrix2)
        with columns2[1]:
            st.write("<span style='font-weight:bold;color:#063672;font-size:14px;'>Rapport de classification</span>", unsafe_allow_html=True)
            st.text(classification_report_str2)

        st.write("<div style='text-align: justify;'>En ajoutant un poids à chacune de nos classes de notre modèle, nous observons une meilleure représentativité.</div>", unsafe_allow_html=True)
        st.write(' ') 
        
        st.write("<span style='font-weight:bold;color:#063672;font-size:24px;'>Interprétation</span>", unsafe_allow_html=True)
        st.write("<div style='text-align: justify;'>L’interprétation du modèle permet de comprendre comment nos variables explicatives ont contribué aux choix de prédiction de notre modèle. Dans notre cas, le modèle étudié étant une régression logistique, nous allons nous intéresser aux odd-ratios ou rapport de chances de nos variables par rapport à notre cible.</div>", unsafe_allow_html=True)

        # Récupération des coefficients de la régression et des noms des colonnes
        coef = reglog2.coef_
        col = X_test_rfe.columns

        # Classe 0
        coefficient_0 = pd.Series(np.exp(coef[0]), col)
        fig0, ax0 = plt.subplots(figsize=(10, 10))
        coefficient_0.sort_values().plot.barh(color='pink', ax=ax0)
        st.pyplot(fig0)
        list_class0 = [index for index, value in coefficient_0.sort_values(ascending=False).items() if value > 1]
        st.write("Variables pour la classe 0 avec un OR > 1 :", list_class0)

        # Classe 1
        coefficient_1 = pd.Series(np.exp(coef[1]), col)
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        coefficient_1.sort_values().plot.barh(color='grey', ax=ax1)
        st.pyplot(fig1)
        list_class1 = [index for index, value in coefficient_1.sort_values(ascending=False).items() if value > 1]
        st.write("Variables pour la classe 1 avec un OR > 1 :", list_class1)

        # Classe 2
        coefficient_2 = pd.Series(np.exp(coef[2]), col)
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        coefficient_2.sort_values().plot.barh(color='blue', ax=ax2)
        st.pyplot(fig2)
        list_class2 = [index for index, value in coefficient_2.sort_values(ascending=False).items() if value > 1]
        st.write("Variables pour la classe 2 avec un OR > 1 :", list_class2)

        A = pd.concat([coefficient_0, coefficient_1, coefficient_2], axis=1)
        st.write("Tableau des coefficients de chaque classe :", A)

        figA, axA = plt.subplots(figsize=(10, 10))
        A.plot.barh(linewidth=0.5, width=0.7, edgecolor='black', ax=axA)
        st.pyplot(figA)
        
        st.write("<span style='font-weight:bold;color:#063672;font-size:24px;'>Observations</span>", unsafe_allow_html=True)

        st.write("""
        Nous constatons que les variables ayant le plus d'influence sur notre cible (0, 1, 2) sont les suivantes :
        - « away_shots_on_target »
        - « home_shots_on_target »
        - « away_clearances »
        - « home_clearances »
        - « home_red_cards »
        - « away_red_cards »
        - « away_passes »
        - « home_passes »
        - « Home »
        - « Dom_Chap1 »
        - « Ext_Chap1 »
        - « Ext_Chap2 »
        """)
        st.write(' ')
        st.write("<div style='text-align: justify;'>Ces observations corroborent les analyses effectuées lors de l'étape de visualisation, qui mettaient en lumière certains facteurs tels que la meilleure attaque ou la pire défense. En somme, nous pouvons affirmer que la capacité d'une équipe de football à se maintenir parmi les meilleures de son championnat repose sur sa performance dans ces différentes variables.</div>", unsafe_allow_html=True)
        
       
    
########################################
# Conclusion
########################################
if selected == "Conclusion":
    # Header
    with header:
        st.markdown("<h1 style='text-align: center; color: #DB0007;'>Conclusion</h1>", unsafe_allow_html=True)
        
        
    st.write("<div style='text-align: justify;'>En conclusion, notre analyse statistique sur les matchs d'Arsenal a permis de mettre en évidence plusieurs éléments clés expliquant la dégradation du niveau de jeu de l'équipe au cours de la dernière décennie. Nous avons constaté une baisse du nombre de victoires et une augmentation du nombre de défaites, ainsi qu'une performance globale faible en termes d'attaque et de défense. Nous avons également analysé des facteurs tels que les performances défensives et offensives.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write("<div style='text-align: justify;'>Nous avons également exploré les dépenses de transfert d'Arsenal au fil des saisons, et constaté que le club a investi des montants de plus en plus importants pour renforcer son effectif, mais sans pour autant améliorer significativement ses performances sur le terrain.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write("<div style='text-align: justify;'>Nous souhaitions également étudier les statistiques xG et XGA, qui permettent d'évaluer la qualité des occasions de but créées et concédées par une équipe. Toutefois, ces données délivrées par Opta Football, ne sont disponibles que depuis la saison 2017/2018, ce qui limite notre analyse de la dégradation du niveau de jeu d'Arsenal. </div>", unsafe_allow_html=True)
    st.write(' ')
    st.write("<div style='text-align: justify;'>En utilisant des techniques de machine learning, nous avons pu identifier des éléments clés ayant une influence significative sur les résultats d'Arsenal, tels que la capacité de l'équipe à concéder des tirs cadrés, son manque d'efficacité devant les buts adverses, le fait de jouer contre les équipes du Big 6, la possession de balle et le fait de se retrouver en infériorité numérique.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write("<div style='text-align: justify;'>Pour aller plus loin, il serait intéressant de s'intéresser aux raisons de cette dégradation en examinant les tendances des thématiques d'actualité liées à Arsenal FC dans les articles de presse au cours de la dernière décennie. En utilisant le web scraping et le text mining, pourquoi pas extraire les sujets récurrents tels que la mauvaise gestion, la faiblesse de l'effectif, la mauvaise stratégie de transfert, la discorde entre les joueurs et le manager, etc. </div>", unsafe_allow_html=True)
    st.write(' ')
    st.write("<div style='text-align: justify;'>Il serait également possible d'approfondir l'analyse des performances de l'équipe en examinant les données de manière plus fine, par exemple en étudiant l'impact des blessures sur les résultats ou en analysant les performances des joueurs individuellement. Enfin, il serait utile de se pencher sur les pratiques de gestion du club pour identifier d'éventuels problèmes de gouvernance ou de gestion qui pourraient contribuer à la dégradation des performances de l'équipe.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write("<div style='text-align: justify;'>Nous espérons que cette analyse a permis de mieux comprendre la dégradation du niveau de jeu d'Arsenal et que cela permettra aux fans du club de mieux appréhender les enjeux pour l'avenir de leur équipe. </div>", unsafe_allow_html=True) 