import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bcb import sgs
from pmdarima.arima import auto_arima
import base64

paginas = ['Previsão']
pagina = st.sidebar.radio('Selecione uma página', paginas)



if pagina == 'Previsão':
	st.title("Previsão IPCA para 12 meses")
	st.subheader('By Leon')
	st.markdown('---')
	if st.button('Clique aqui para realizar a previsão do IPCA para os próximos 12 meses (demora cerca de 2 mim)'):
		#Importando dados
		ipca = sgs.get(('IPCA', 433), start="1995-01-01")
		#ipca.index = pd.DatetimeIndex(ipca.index ,freq='MS')
		## AUTOARIMA
		modelo = auto_arima(
    ipca, 
   trace = True, 
    stepwise = False, 
    seasonal=True, 
    start_p=0, 
    start_d=0, 
    start_q=0, 
    start_P=0,
    start_D=0, 
    start_Q=0, 
    max_p=2,
    max_d=2, 
    max_q=2,
    max_P=2,
    max_D=2, 
    max_Q=2,
    m=12)

		sns.set(style="darkgrid")
		fig, ax = plt.subplots(figsize = (18,9))
		plt.title('Previsão IPCA', fontsize=18, fontweight='bold')
		plt.plot(modelo.predict(12), marker = 'o')
		plt.show()
		st.pyplot(fig)
		
		#subm = pd.DataFrame()
		#subm["previsão IPCA"] = modelo.predict(12)
		#st.dataframe(ipca)
		pred=np.round(modelo.predict(12),decimals = 2)
		df = pd.DataFrame()
		df["previsão IPCA (considere duas casas decimais"] = pred
		st.table(df)
		#st.table(modelo.predict(12))

	csv = df.to_csv(index=True)
	b64 = base64.b64encode(csv.encode()).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="Arquivo.csv">Clique para salvar a previsão em csv</a>'
	st.markdown(href, unsafe_allow_html=True)