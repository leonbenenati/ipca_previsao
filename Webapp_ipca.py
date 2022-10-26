import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import SARIMAX
from bcb import sgs
import pickle

paginas = ['Previsão']
pagina = st.sidebar.radio('Selecione uma página', paginas)



if pagina == 'Previsão':
	st.title("Previsão IPCA")
	st.subheader('By: Leon Emiliano Benenati')
	
	if st.button('CLIQUE Aqui para realizar a previsão'):
		#Importando dados
		ipca = sgs.get(('IPCA', 433), start="1995-01-01")
		#ipca.index = pd.DatetimeIndex(ipca.index ,freq='MS')

		#Modelo Sarima
		sarima= SARIMAX(ipca, order=(1, 1, 2), seasonal_order=(1,0,1,12),trend="c").fit()
		#sarima.forecast(12)

		sns.set(style="darkgrid")
		fig, ax = plt.subplots(figsize = (18,9))
		plt.title('Previsão IPCA', fontsize=18, fontweight='bold')
		plt.plot(sarima.forecast(12), marker = 'o')
		plt.show()
		st.pyplot(fig)
		
		subm = pd.DataFrame()
		subm["previsão IPCA"] = sarima.forecast(12)
		#st.dataframe(ipca)
		#st.dataframe(subm)
		st.table(sarima.forecast(12))