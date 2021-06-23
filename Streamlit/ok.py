import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, roc_curve, roc_auc_score, recall_score, plot_confusion_matrix



data = pd.read_csv('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Datasets/bank-full-transformed.csv')
data5 = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/MachineLearning/cenario5-jony-finals.xlsx')
cluster1 = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/cluster0.xlsx')
cluster2 = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/cluster1.xlsx')
cluster3 = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/cluster2.xlsx')

dataf0 = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/dataf0.xlsx')
dataf1 = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/dataf1.xlsx')
dataf2 = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/dataf2.xlsx')

data_yes = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Description/data_yes.xlsx')
sub = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Description/subscrição.xlsx')
todos = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Description/todos_os_clientes.xlsx')

clusters = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/df3clusters.xlsx')
st.title("Análise de uma Campanha de Telemarketing de uma Intituição Bancária Portuguesa")
#pickle_in = open('decisionpickle.pkl', 'wb') 
#classifier = pickle.load(pickle_in)
    
st.sidebar.title("Visualização da Análise")
z = st.sidebar.selectbox('Escolha',('Gráficos', 'Classificação', 'Clusters'))
st.sidebar.header(z)

job_dict = {"blue-collar":0,"management":1,"technician":2,"admin.":3,"services":4,"retired":5,"self-employed":6,"entrepreneur":7,"unemployed":8,"housemaid":9,"student":10,"unknown":11}
marital_dict = {"married":0,"single":1, "divorced":2}
education_dict = {"primary": 0,"secondary":1,"tertiary": 2, "other": 3}
month_dict = {"Primavera": 0,"Outono":1,"Primavera": 2, "Verão": 3}


def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def encode(dataF, col):
    return pd.concat([dataF, pd.get_dummies(col, prefix=col.name)], axis=1)

def roc_auc_graph():
    data_train = pd.read_csv('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Data Transformation/bank-full-transformed.csv')
    data_train = encode(data_train, data_train.job)
    data_train = encode(data_train, data_train.marital)
    data_train = encode(data_train, data_train.education)
    data_train = encode(data_train, data_train.month)
    data_train.drop(['job', "marital", "education", "month"], axis=1, inplace=True)
    data_x = data_train.drop(["y", "pdays"], axis =1)
    data_y = data_train["y"]
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle = True)
    smote = SMOTE()
    X_resample, y_resample = smote.fit_resample(X_train, y_train)
    models = []
    models.append(('LogisticRegression', LogisticRegression(penalty = 'l2', max_iter = 100, C = 20)))
    models.append(('KNeighborsClassifier', KNeighborsClassifier(weights= 'distance', n_neighbors= 35, n_jobs= -1, leaf_size= 3)))
    models.append(('Decison-Tree', DecisionTreeClassifier(criterion='gini', max_depth= None, min_samples_leaf= 3)))
    models.append(('RandomForest',RandomForestClassifier(n_estimators= 50, min_samples_split= 2, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 10, bootstrap= True)))
    fig = go.Figure()
    for name, model in models:
        #kfold = model_selection.KFold(n_splits=10)
        model = model.fit(X_resample, y_resample)
        #proba = cross_val_predict(model, X_resample, y_resample, cv = kfold, n_jobs=-1, method='predict_proba')
        prob_prediction = model.predict_proba(X_test)[::,1]
        #prob_prediction = proba[:,1]
        fpr, tpr, _ = roc_curve(y_test, prob_prediction)
        auc = roc_auc_score(y_test, prob_prediction)
        fig.add_trace(go.Scatter(x = fpr, y = tpr, name = name + ": AUROC = %.2f" % auc))
    fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(title = 'Curva ROC AUC', xaxis_title = "Taxa de falsos Positivos", yaxis_title = "Taxa de falsos Negativos")
    fig.update_layout(height = 500, width=1000)

    for name, model in models:
        y_pred = model.predict(X_test)
        conf = confusion_matrix(y_test, y_pred)
        titulo = name
        nome_quadrantes = ['True Negative','False Positive','False Negative','True Positive']
        valor_quadrantes = ["{0:0.0f}".format(value) for value in
        conf.flatten()]
        percentagem_quadrantes = ["{0:.2%}".format(value) for value in
        conf.flatten()/np.sum(conf)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
        zip(valor_quadrantes,percentagem_quadrantes,nome_quadrantes)]
        labels = np.asarray(labels).reshape(2,2)
        fig1 = plt.figure(figsize=(9,6))
        plt.title(titulo, fontsize =20)
        sns.heatmap(conf, annot=labels, fmt="", cmap='Greens')
        st.write(fig1)
        
    st.write(fig)
    


def importar_modelo():
    if n == 'Decison-tree':
        model = joblib.load('Decison-tree.mdl')
             
    if n == 'LogisticRegression':
        model = joblib.load('logistic.mdl')
            
    if n == 'Knn':
        model = joblib.load('KNeighborsClassifier.mdl')
        
    if n == 'RandomForest':
        model = joblib.load('RandomForest.mdl')
        
        
        
    #return prediction_proba

def user_input_features():
        job = st.selectbox('Profissão', ('blue-collar','management','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'))
        marital = st.selectbox('Estado Cívil', ('married', 'single', 'divorced'))
        education = st.selectbox('Educação', ('primary', ' secondary', 'tertiary', 'other'))
        month = st.selectbox('Estação do ano', ('Inverno', 'Primavera', 'Verão', 'Outono'))
        housing = st.selectbox('Crédito Habitação?', ('sim', 'Não'))
        loan = st.selectbox('Possuí empréstimo?', ('Sim', 'Não'))
        day = st.selectbox('Altura do Mês', ('Primeira Quinzena', 'Segunda Quinzena'))
        age = st.selectbox('Faixa Etária', (20, 30, 40, 50, 60))
        balance = st.selectbox('Balanço Bancário', ('Inferior a 0', 'Entre 0 e 2500', 'Entre 2500 e 5000', 'Superior a 5000'))
        previous = st.selectbox('Número de Contactos anteriores de outras campanhas', ('Nenhum' , 'Um Contacto' ,'Mais do que um'))
        default = st.selectbox('Possui crédito em default?', ('Sim' ,'Não'))
        campaign = st.selectbox('Número de contactos durante esta campanha', ('Um contacto' ,'Mais do que um contacto'))
        if housing == 'sim':
            housing = 1
        else:
            housing = 0
        
        if previous == 'Nenhum':
            previous = 0
        if previous == 'Um Contacto':
            previous = 1
        if previous == 'Mais do que um':
            previous = 2
        
        if loan == 'sim':
            loan = 1
        else:
            loan = 0

        if default == 'sim':
            default = 1
        else:
            default = 0
        
        if day == 'Primeira Quinzena':
            day = 1
        else:
            day = 2 
        
        if campaign == 'Um contacto':
            campaign = 1
        else:
            campaign = 2
        
        if balance == 'Inferior a 0':
            balance = -1
        if balance == 'Entre 0 e 2500':
            balance = 0
        if balance == 'Entre 2500 e 5000':
            balance = 1      
        if balance == 'Superior a 5000':
            balance = 2 
        #job_en = get_value(job,job_dict)
        #marital_en = get_value(marital,marital_dict)
        #education_en = get_value(education,education_dict)
        #month_en = get_value(month,month_dict)
        dataF = {'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'campaign': campaign,
                'default' : default,
                'month' : month,
                'balance': balance,
                'housing': housing,
                'loan': loan,
                'day': day,
                "previous": previous}                
        features = pd.DataFrame(dataF, index=[0])
        #st.write(dataF)
        #st.write(features)
        data2 = pd.read_csv('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Data Transformation/bank-full-transformed.csv')
        data2 = data2.drop(["y", "pdays"], axis =1)
        df = pd.concat([features,data2],axis=0)
        encode = ['job','marital','education', 'month']
        for col in encode:
            dummy = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df,dummy], axis=1)
            del df[col]
        df = df[:1]
        #st.write(df)
        data = np.array(df).reshape(1,-1)
        prediction_proba = round(model.predict_proba(data)[0][1], 2 )
        st.write(prediction_proba) 
        if st.button("Previsão"):
            if prediction_proba >= 0.5:
                st.write(f'Sucesso: Este cliente vai aceitar o depósito a prazo. Probabilidade:  {round(prediction_proba * 100, 2)}%')
                prediction = 1
            else:
                st.write(f'Insucesso: Este cliente não vai aceitar o depósito a prazo. Probabilidade:  {100 - round(prediction_proba * 100, 2)}%')
                prediction = 0 
        
        #if st.button('Prever Modelo'):
                
        
        return data

def concat():
    input_df = user_input_features()     
    data2 = pd.read_csv('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Datasets/bank-full-transformed.csv')
    data2 = data2.drop(["y", "pdays", "month", "default", "campaign"], axis =1)
    df = pd.concat([input_df,data2],axis=0)
    encode = ['job','marital','education']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df[:1]
    st.write(data2)
    st.write(df)
    return df
       


if z == 'Gráficos':
    
    t = st.sidebar.selectbox('Gráficos',('job','age','marital', 'loan', 'balance', 'campaign', 'day', 'education', 'housing', 'month', 'previous', 'pdays'))
    comparar = st.sidebar.selectbox('Comparacao',('job','age','marital', 'loan', 'balance', 'campaign', 'day', 'education', 'housing', 'month', 'previous', 'pdays'))
    labels = data[t].value_counts().index
    values = data[t].value_counts().values
    fig = px.pie(data, values= values, names= labels,title = 'Pie chart: '+ t)
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='show')

    joby = pd.DataFrame(data.groupby(t)['y'].sum()).reset_index()
    joby = joby.sort_values(by = 'y', ascending = False)
    fig1 = px.bar(joby.iloc[:], 
                x = t, y = 'y', title = 'Número de subscritos X Categoria ' + t )
   
    count_job_response_pct = pd.crosstab(data['y'],data[t]).apply(lambda x: x/x.sum() * 100)
    count_job_response_pct = count_job_response_pct.transpose()
    plot_job = count_job_response_pct[1].sort_values(ascending = True).plot(kind ='barh',figsize = (12,6))             
    plt.title('Taxa de Subscrição por Profissão')
    plt.xlabel('Taxa de Subscrição')
    plt.ylabel('Job Category')
    for rec, label in zip(plot_job.patches,count_job_response_pct[1].sort_values(ascending = True).round(1).astype(str)):
        plot_job.text(rec.get_width()+0.8,rec.get_y()+ rec.get_height()-0.5,label+'%',ha = 'center',va='bottom')
    

    a = px.histogram(data, x=t, color="y", barmode = 'group', title = 'Distribuição da categoria ' + t + ' com subscrição') 
    y = (px.histogram(data, x=t, color=t, title = 'Categoria ' + t))
    x = (px.histogram(data, x=t, color=comparar, title = 'Relação categoria ' + t + ' com a categoria ' + comparar)) 
    b = (px.histogram(data, x=t, color=comparar, barmode = 'group', title = 'Relação categoria ' + t + ' com a categoria ' + comparar))
    st.write(y)
    st.write(a)
    st.write(fig1)
    st.write(x)
    st.write(b)
    st.write(fig)
    
    
    
    

if z == 'Classificação':
    if st.sidebar.button("Verificar valores de classificação"):
        st.write(data5)
        st.write(roc_auc_graph())
    
        
    n = st.sidebar.selectbox('Modelos',('Escolha o modelo abaixo', 'Knn', 'LogisticRegression', 'Decison-tree', 'RandomForest'), index = 0 )      
    if n == 'Escolha o modelo abaixo':
        st.write("""A equipa definiu 5 cenários e 4 modelos para realizar os testes de previsão dos perfis que nos tinhamos previamente proposto a identificar.
        """)
        st.write("""Pode agora selecionar o modelo que pretende analisar com os atributos do cenário em que foram obtidas as melhores classificações.
        """)
        st.write("""A opção "Verificar valores de classificação" permite verificar os resultados obtidos nas métricas em todos os modelos bem como os gráficos do AUC e da Confusion Matrix.
        """)
        if st.button("Possiveis clientes alvo"):
            st.write("""Atributos mais verificados de clientes que subscreveram à campanha:
            """)
            st.write(data_yes)
            st.write("""Uma vez que os atributos acima correspondem aos atributos mais verificados não significa que serão os atributos com uma maior percentagem de sucesso porque estes podem ter sido contactados muitas mais vezes.
            """)
            st.write("""Dito isto, os atributos que realmente nos são interessantes são aqueles que não precisam de ser contactados muitas vezes para subscreverem
            """)
            st.write("""Os atributos que correspondem a uma maior taxa de subscrição são:
            """)
            st.write(sub)
            st.write("""Todos os clientes que responderam afirmativamente à campanha:
            """)
            st.write(todos)
    if n == 'Decison-tree':
        st.write("""
        Escolheu o Modelo: 
        """ + n)
        st.write("""
        Escolha agora os atributos que pretende avaliar neste modelo e veja a probabilidade de esse perfil de cliente aceitar o depóstio a prazo.
        """)
        model = joblib.load('Decison-tree.mdl')
        user_input_features()
        
    if n == 'LogisticRegression':
        st.write("""
        Escolheu o Modelo: 
        """ + n)
        st.write("""
        Escolha agora os atributos que pretende avaliar neste modelo e veja a probabilidade de esse perfil de cliente aceitar o depóstio a prazo.
        """)
        model = joblib.load('logistic.mdl')   
        user_input_features()
        
    if n == 'Knn':
        st.write("""
        Escolheu o Modelo: 
        """ + n)
        st.write("""
        Escolha agora os atributos que pretende avaliar neste modelo e veja a probabilidade de esse perfil de cliente aceitar o depóstio a prazo.
        """)
        model = joblib.load('KNeighborsClassifier.mdl')   
        user_input_features()
        
    if n == 'RandomForest':
        st.write("""
        Escolheu o Modelo: 
        """ + n)
        st.write("""
        Escolha agora os atributos que pretende avaliar neste modelo e veja a probabilidade de esse perfil de cliente aceitar o depóstio a prazo.
        """)
        model = joblib.load('RandomForest.mdl')   
        user_input_features()
        

if z == 'Clusters':
    #clusters = pd.read_excel('C:/Users/joao_/Desktop/4 ANO JOAO/SINO/VS CODE PROJETO/Jupyter Notebook/Clustering/df3clusters.xlsx')
    #st.write(clusters)
    c = st.sidebar.selectbox('Modelos',('Escolha o cluster abaixo', 'Cluster1', 'Cluster2', 'Cluster3'), index = 0 )
    if c == 'Escolha o cluster abaixo':
        st.write("""Escolha o Cluster que pretende visualizar.
        """)
    
    if c == 'Cluster1':
        st.write("""Atributos mais verificados no Cluster 1
        """)
        st.write(cluster1)
        l = st.sidebar.selectbox('Verificar categorias Cluster',('job','age','marital', 'loan', 'balance','education', 'housing','y'))
        n = st.sidebar.selectbox('Verificar outra categoria Cluster se pretender comparar',('job','age','marital', 'loan', 'balance','education', 'housing', 'y'))
        i = px.histogram(dataf0, x=l, color="cluster_predicted", barmode = 'group', title = 'Categoria ' + l + ' e os seus clusters correspondentes') 
        u = px.histogram(dataf0, x=n, color="cluster_predicted", barmode = 'group', title = 'Categoria ' + n + ' e os seus clusters correspondentes')
        st.write("""Percentagem de aceitação: 8%
        """)
        st.write(i) 
        st.write(u)
    
    if c == 'Cluster2':
        st.write("""Atributos mais verificados no Cluster 2
        """)
        st.write(cluster2)
        l = st.sidebar.selectbox('Verificar categorias Cluster',('job','age','marital', 'loan', 'balance','education', 'housing','y'))
        n = st.sidebar.selectbox('Verificar outra categoria Cluster se pretender comparar',('job','age','marital', 'loan', 'balance','education', 'housing','y'))
        i = px.histogram(dataf1, x=l, color="cluster_predicted", barmode = 'group', title = 'Categoria ' + l + ' e os seus clusters correspondentes') 
        u = px.histogram(dataf1, x=n, color="cluster_predicted", barmode = 'group', title = 'Categoria ' + n + ' e os seus clusters correspondentes')
         
        st.write("""Percentagem de aceitação: 18%
        """)
        st.write(i)
        st.write(u)
    
    if c == 'Cluster3':
        st.write("""Atributos mais verificados no Cluster 3
        """)
        st.write(cluster3)
        l = st.sidebar.selectbox('Verificar categorias Cluster',('job','age','marital', 'loan', 'balance','education', 'housing','y'))
        n = st.sidebar.selectbox('Verificar outra categoria Cluster se pretender comparar',('job','age','marital', 'loan', 'balance','education', 'housing','y'))
        i = px.histogram(dataf2, x=l, color="cluster_predicted", barmode = 'group', title = 'Categoria ' + l ) 
        u = px.histogram(dataf2, x=n, color="cluster_predicted", barmode = 'group', title = 'Categoria ' + n )
        
        st.write("""Percentagem de aceitação: 12%
        """)
        st.write(i) 
        st.write(u)
    
    
  
   
   
    
