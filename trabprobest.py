import pandas as pd
import numpy as np
import math
from scipy.stats import gamma, norm, probplot, pearsonr, power_divergence
import matplotlib.pyplot as plt


reading_chromecast = pd.read_csv('dataset_chromecast.csv', sep = ',')

reading_chromecast['bytes_up_log'] = np.log10(reading_chromecast['bytes_up'])
reading_chromecast['bytes_down_log'] = np.log10(reading_chromecast['bytes_down'])

reading_chromecast.bytes_up_log.loc[np.isneginf(reading_chromecast.bytes_up_log)]=0
reading_chromecast.bytes_down_log.loc[np.isneginf(reading_chromecast.bytes_down_log)]=0

reading_chromecast.to_csv('dataset_chromecast_log10.csv', index=False, sep = ',')



reading_smarttv = pd.read_csv('dataset_smart-tv.csv', sep = ',')

reading_smarttv['bytes_up_log'] = np.log10(reading_smarttv['bytes_up'])
reading_smarttv['bytes_down_log'] = np.log10(reading_smarttv['bytes_down'])

reading_smarttv.bytes_up_log.loc[np.isneginf(reading_smarttv.bytes_up_log)]=0
reading_smarttv.bytes_down_log.loc[np.isneginf(reading_smarttv.bytes_down_log)]=0

reading_smarttv.to_csv('dataset_smarttv_log10.csv', index=False, sep = ',')


chromecast_log10 = pd.read_csv('dataset_chromecast_log10.csv', sep = ',')
smarttv_log10 =  pd.read_csv('dataset_smarttv_log10.csv', sep = ',')

#DEFININDO A FUNCAO DO METODO DE STURGES
def sturges(dataframe):
    n = len(dataframe.index)
    k = 1+3.3322*np.log10(n)
    return round(k)

#HISTOGRAMA
plt.figure(figsize=(10,6))
plt.hist(chromecast_log10.bytes_down_log, bins = sturges(chromecast_log10,'bytes_down_log'), edgecolor="black")
plt.title("Bytes Down Chromecast")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(chromecast_log10.bytes_up_log, bins = sturges(chromecast_log10,'bytes_up_log'), edgecolor="black")
plt.title("Bytes Up Chromecast")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(smarttv_log10.bytes_down_log, bins = sturges(smarttv_log10,'bytes_down_log'), edgecolor="black")
plt.title("Bytes Down Smart TV")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(smarttv_log10.bytes_up_log, bins = sturges(smarttv_log10,'bytes_up_log'), edgecolor="black")
plt.title("Bytes Up Smart TV")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()



#FUNCAO DESTRIBUICAO EMPIRICA
plt.figure(figsize=(10,6))
sorting_values = chromecast_log10["bytes_down_log"].sort_values()
linspace = np.linspace(0, 1, len(chromecast_log10['bytes_down_log']))
plt.plot(sorting_values, linspace)
plt.title("Distribuição Empírica Bytes Down Chromecast")
plt.xlabel("BPS (LOG10)")
plt.ylabel("f(x)")
plt.show()

plt.figure(figsize=(10,6))
sorting_values = chromecast_log10["bytes_up_log"].sort_values()
linspace = np.linspace(0, 1, len(chromecast_log10['bytes_up_log']))
plt.plot(sorting_values, linspace)
plt.title("Distribuição Empírica Bytes Up Chromecast")
plt.xlabel("BPS (LOG10)")
plt.ylabel("f(x)")
plt.show()

plt.figure(figsize=(10,6))
sorting_values = smarttv_log10["bytes_down_log"].sort_values()
linspace = np.linspace(0, 1, len(smarttv_log10['bytes_down_log']))
plt.plot(sorting_values, linspace)
plt.title("Distribuição Empírica Bytes Down Smart TV")
plt.xlabel("BPS (LOG10)")
plt.ylabel("f(x)")
plt.show()

plt.figure(figsize=(10,6))
sorting_values = smarttv_log10["bytes_up_log"].sort_values()
linspace = np.linspace(0, 1, len(smarttv_log10['bytes_up_log']))
plt.plot(sorting_values, linspace)
plt.title("Distribuição Empírica Bytes Up Smart TV")
plt.xlabel("BPS (LOG10)")
plt.ylabel("f(x)")
plt.show()



#BOXPLOT
plt.figure(figsize=(10,6))
bytes_down_chromecast, bytes_up_chromecast = chromecast_log10["bytes_down_log"], chromecast_log10["bytes_up_log"]
bytes_down_smarttv, bytes_up_smarttv = smarttv_log10["bytes_down_log"], smarttv_log10["bytes_up_log"]
labels=["Bytes Down Chromecast", "Bytes Up Chromecast", "Bytes Down Smart TV", "Bytes Up Smart TV "]
plt.boxplot([bytes_down_chromecast, bytes_up_chromecast, bytes_down_smarttv, bytes_up_smarttv], labels=labels)
plt.ylabel("BPS (LOG10)")
plt.title("BOXPLOT Chromecast ; SmartTV")
plt.show()


#MEDIA VARIANCIA E DESVIO PADRAO
print('Chromecast Bytes Down')
mean, var, std = chromecast_log10['bytes_down_log'].mean(), chromecast_log10['bytes_down_log'].var(), chromecast_log10['bytes_down_log'].std()
print(f'Média: {mean} ; Variância: {var} ; Desvio Padrão: {std}')
print()
print('Chromecast Bytes Up')
mean, var, std = chromecast_log10['bytes_up_log'].mean(), chromecast_log10['bytes_up_log'].var(), chromecast_log10['bytes_up_log'].std()
print(f'Média: {mean} ; Variância: {var} ; Desvio Padrão: {std}')

print()

print('Smart TV Bytes Down')
mean, var, std = smarttv_log10['bytes_down_log'].mean(), smarttv_log10['bytes_down_log'].var(), smarttv_log10['bytes_down_log'].std()
print(f'Média: {mean} ; Variância: {var} ; Desvio Padrão: {std}')
print()
print('Smart TV Bytes Up')
mean, var, std = smarttv_log10['bytes_up_log'].mean(), smarttv_log10['bytes_up_log'].var(), smarttv_log10['bytes_up_log'].std()
print(f'Média: {mean} ; Variância: {var} ; Desvio Padrão: {std}')



#TRATAMENTO PARA PEGAR A HORA DAS COLETAS
chromecast_log10['hora'] = chromecast_log10['date_hour'].apply(lambda x: int(x.split(" ")[1].split(":")[0]))
smarttv_log10['hora'] = smarttv_log10['date_hour'].apply(lambda x: int(x.split(" ")[1].split(":")[0]))

chromecast_log10.to_csv('dataset_chromecast_log10_hora.csv', index=False, sep = ',')
smarttv_log10.to_csv('dataset_smarttv_log10_hora.csv', index=False, sep = ',')

chromecast_log10_hora = pd.read_csv('dataset_chromecast_log10_hora.csv', sep = ',')
smarttv_log10_hora = pd.read_csv('dataset_smarttv_log10_hora.csv', sep = ',')


#BOXPLOT Estatistica por horario
for hora in range(0, 24):
    plt.figure(figsize=(10,6))
    hora_bytesdown_chromecast = chromecast_log10_hora[chromecast_log10_hora["hora"] == hora]["bytes_down_log"]
    hora_bytesup_chromecast   = chromecast_log10_hora[chromecast_log10_hora["hora"] == hora]["bytes_up_log"]
    hora_bytesdown_smarttv    = smarttv_log10_hora[smarttv_log10_hora["hora"] == hora]["bytes_down_log"]
    hora_bytesup_smarttv      = smarttv_log10_hora[smarttv_log10_hora["hora"] == hora]["bytes_up_log"]
    labels                    = ["Bytes Down Chromecast", "Bytes Up Chromecast","Bytes Down Smart TV", "Bytes Up Smart TV"]
    plt.boxplot([hora_bytesdown_chromecast, hora_bytesup_chromecast, hora_bytesdown_smarttv, hora_bytesup_smarttv], labels=labels)
    plt.ylabel("BPS (LOG10)")
    plt.title(f"BOXPLOT Chromecast ; SmartTV ; hora = {hora}")
    plt.show()


#MEDIA VARIANCIA E DESVIO PADRAO PARA AS HORAS CHROMECAST
plt.figure(figsize=(10,6))
mean = chromecast_log10_hora.groupby("hora")["bytes_down_log"].mean()
var  = chromecast_log10_hora.groupby("hora")["bytes_down_log"].var()
std  = chromecast_log10_hora.groupby("hora")["bytes_down_log"].std()
plt.plot(mean)
plt.plot(var)
plt.plot(std)
plt.xlabel("Hora")
plt.ylabel("BPS (LOG10)")
plt.legend(["Média", "Variância", "Desvio Padrão"])
plt.title('Bytes Down Chromecast: Média, Variância e Desvio Padrão por Hora')
plt.show()

plt.figure(figsize=(10,6))
mean = chromecast_log10_hora.groupby("hora")["bytes_up_log"].mean()
var  = chromecast_log10_hora.groupby("hora")["bytes_up_log"].var()
std  = chromecast_log10_hora.groupby("hora")["bytes_up_log"].std()
plt.plot(mean)
plt.plot(var)
plt.plot(std)
plt.xlabel("Hora")
plt.ylabel("BPS (LOG10)")
plt.legend(["Média", "Variância", "Desvio Padrão"])
plt.title('Bytes Up Chromecast: Média, Variância e Desvio Padrão por Hora')
plt.show()


#MEDIA VARIANCIA E DESVIO PADRAO PARA AS HORAS SMART TV
plt.figure(figsize=(10,6))
mean = smarttv_log10_hora.groupby("hora")["bytes_down_log"].mean()
var  = smarttv_log10_hora.groupby("hora")["bytes_down_log"].var()
std  = smarttv_log10_hora.groupby("hora")["bytes_down_log"].std()
plt.plot(mean)
plt.plot(var)
plt.plot(std)
plt.xlabel("Hora")
plt.ylabel("BPS (LOG10)")
plt.legend(["Média", "Variância", "Desvio Padrão"])
plt.title('Bytes Down Smart TV: Média, Variância e Desvio Padrão por Hora')
plt.show()

plt.figure(figsize=(10,6))
mean = smarttv_log10_hora.groupby("hora")["bytes_up_log"].mean()
var  = smarttv_log10_hora.groupby("hora")["bytes_up_log"].var()
std  = smarttv_log10_hora.groupby("hora")["bytes_up_log"].std()
plt.plot(mean)
plt.plot(var)
plt.plot(std)
plt.xlabel("Hora")
plt.ylabel("BPS (LOG10)")
plt.legend(["Média", "Variância", "Desvio Padrão"])
plt.title('Bytes Up Smart TV: Média, Variância e Desvio Padrão por Hora')
plt.show()



#DEFININDO OS DATASETS 
dataset1_hora = smarttv_log10_hora.groupby("hora")["bytes_up_log"].median().idxmax()
dataset1 = smarttv_log10_hora.where(smarttv_log10_hora.hora == dataset1_hora).dropna()

dataset2_hora = smarttv_log10_hora.groupby("hora")["bytes_up_log"].mean().idxmax()
dataset2 = smarttv_log10_hora.where(smarttv_log10_hora.hora == dataset2_hora).dropna()

dataset3_hora = smarttv_log10_hora.groupby("hora")["bytes_down_log"].median().idxmax()
dataset3 = smarttv_log10_hora.where(smarttv_log10_hora.hora == dataset3_hora).dropna()

dataset4_hora = smarttv_log10_hora.groupby("hora")["bytes_down_log"].mean().idxmax()
dataset4 = smarttv_log10_hora.where(smarttv_log10_hora.hora == dataset4_hora).dropna()

dataset5_hora = chromecast_log10_hora.groupby("hora")["bytes_up_log"].median().idxmax()
dataset5 = chromecast_log10_hora.where(chromecast_log10_hora.hora == dataset5_hora).dropna()

dataset6_hora = chromecast_log10_hora.groupby("hora")["bytes_up_log"].mean().idxmax()
dataset6 = chromecast_log10_hora.where(chromecast_log10_hora.hora == dataset6_hora).dropna()

dataset7_hora = chromecast_log10_hora.groupby("hora")["bytes_down_log"].median().idxmax()
dataset7 = chromecast_log10_hora.where(chromecast_log10_hora.hora == dataset7_hora).dropna()

dataset8_hora = chromecast_log10_hora.groupby("hora")["bytes_down_log"].mean().idxmax()
dataset8 = chromecast_log10_hora.where(chromecast_log10_hora.hora == dataset8_hora).dropna()

#HISTOGRAMAS DOS DATASETS
plt.figure(figsize=(10,6))
plt.hist(dataset1.bytes_up_log, bins = sturges(dataset1,'bytes_up_log'), edgecolor="black")
plt.title(f"Bytes Up Smart TV Maior Mediana - {dataset1_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(dataset2.bytes_up_log, bins = sturges(dataset2,'bytes_up_log'), edgecolor="black")
plt.title(f"Bytes Up Smart TV Maior Média - {dataset2_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(dataset3.bytes_down_log, bins = sturges(dataset3,'bytes_down_log'), edgecolor="black")
plt.title(f"Bytes Down Smart TV Maior Mediana - {dataset3_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(dataset4.bytes_down_log, bins = sturges(dataset4,'bytes_down_log'), edgecolor="black")
plt.title(f"Bytes Down Smart TV Maior Média - {dataset4_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(dataset5.bytes_up_log, bins = sturges(dataset5,'bytes_up_log'), edgecolor="black")
plt.title(f"Bytes Up Chromecast Maior Mediana - {dataset5_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(dataset6.bytes_up_log, bins = sturges(dataset6,'bytes_up_log'), edgecolor="black")
plt.title(f"Bytes Up Chromecast Maior Média - {dataset6_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(dataset7.bytes_down_log, bins = sturges(dataset7,'bytes_down_log'), edgecolor="black")
plt.title(f"Bytes Down Chromecast Maior Mediana - {dataset7_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()

plt.figure(figsize=(10,6))
plt.hist(dataset8.bytes_down_log, bins = sturges(dataset8,'bytes_down_log'), edgecolor="black")
plt.title(f"Bytes Down Chromecast Maior Média - {dataset8_hora} horas")
plt.xlabel("BPS (LOG10)")
plt.ylabel("Frequência")
plt.show()


dataset1_mean = dataset1['bytes_up_log'].mean()
dataset2_mean = dataset2['bytes_up_log'].mean()
dataset3_mean = dataset3['bytes_down_log'].mean()
dataset4_mean = dataset4['bytes_down_log'].mean()
dataset5_mean = dataset5['bytes_up_log'].mean()
dataset6_mean = dataset6['bytes_up_log'].mean()
dataset7_mean = dataset7['bytes_down_log'].mean()
dataset8_mean = dataset8['bytes_down_log'].mean()

dataset1_std= dataset1['bytes_up_log'].std()
dataset2_std = dataset2['bytes_up_log'].std()
dataset3_std = dataset3['bytes_down_log'].std()
dataset4_std = dataset4['bytes_down_log'].std()
dataset5_std = dataset5['bytes_up_log'].std()
dataset6_std = dataset6['bytes_up_log'].std()
dataset7_std = dataset7['bytes_down_log'].std()
dataset8_std = dataset8['bytes_down_log'].std()


print(f'Mean: {dataset1_mean} ; Standard Deviation: {dataset1_std}')

print(f'Mean: {dataset2_mean} ; Standard Deviation: {dataset2_std}')

print(f'Mean: {dataset3_mean} ; Standard Deviation: {dataset3_std}')

print(f'Mean: {dataset4_mean} ; Standard Deviation: {dataset4_std}')

print(f'Mean: {dataset5_mean} ; Standard Deviation: {dataset5_std}')

print(f'Mean: {dataset6_mean} ; Standard Deviation: {dataset6_std}')

print(f'Mean: {dataset7_mean} ; Standard Deviation: {dataset7_std}')

print(f'Mean: {dataset8_mean} ; Standard Deviation: {dataset8_std}')



#OBTENDO SHAPE OFFSET E SCALE PARA O CALCULO DA GAMMA
shape_dataset1, offset_dataset1, scale_dataset1 = gamma.fit(dataset1["bytes_up_log"])
shape_dataset2, offset_dataset2, scale_dataset2 = gamma.fit(dataset2["bytes_up_log"])
shape_dataset3, offset_dataset3, scale_dataset3 = gamma.fit(dataset3["bytes_down_log"])
shape_dataset4, offset_dataset4, scale_dataset4 = gamma.fit(dataset4["bytes_down_log"])
shape_dataset5, offset_dataset5, scale_dataset5 = gamma.fit(dataset5["bytes_up_log"])
shape_dataset6, offset_dataset6, scale_dataset6 = gamma.fit(dataset6["bytes_up_log"])
shape_dataset7, offset_dataset7, scale_dataset7 = gamma.fit(dataset7["bytes_down_log"])
shape_dataset8, offset_dataset8, scale_dataset8 = gamma.fit(dataset8["bytes_down_log"])

print(f'Shape: {shape_dataset1} ; Offset: {offset_dataset1} ; Scale: {scale_dataset1}')
print(f'Shape: {shape_dataset2} ; Offset: {offset_dataset2} ; Scale: {scale_dataset2}')
print(f'Shape: {shape_dataset3} ; Offset: {offset_dataset3} ; Scale: {scale_dataset3}')
print(f'Shape: {shape_dataset4} ; Offset: {offset_dataset4} ; Scale: {scale_dataset4}')
print(f'Shape: {shape_dataset5} ; Offset: {offset_dataset5} ; Scale: {scale_dataset5}')
print(f'Shape: {shape_dataset6} ; Offset: {offset_dataset6} ; Scale: {scale_dataset6}')
print(f'Shape: {shape_dataset7} ; Offset: {offset_dataset7} ; Scale: {scale_dataset7}')
print(f'Shape: {shape_dataset8} ; Offset: {offset_dataset8} ; Scale: {scale_dataset8}')



#REALIZANDO OS GRAFICOS MLE
plt.figure(figsize=(10,6))
plt.title(f'Byes Up SmartTV Maior Mediana - Dataset 1 - {dataset1_hora} horas')
data = dataset1['bytes_up_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset1_mean, dataset1_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset1, offset_dataset1, scale_dataset1)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset1,'bytes_up_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()

plt.figure(figsize=(10,6))
plt.title(f'Byes Up SmartTV Maior Média - Dataset 2 - {dataset2_hora} horas')
data = dataset2['bytes_up_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset2_mean, dataset2_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset2, offset_dataset2, scale_dataset2)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset2,'bytes_up_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()

plt.figure(figsize=(10,6))
plt.title(f'Byes Down SmartTV Maior Mediana - Dataset 3 - {dataset3_hora} horas')
data = dataset3['bytes_down_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset3_mean, dataset3_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset3, offset_dataset3, scale_dataset3)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset3,'bytes_down_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()

plt.figure(figsize=(10,6))
plt.title(f'Byes Down SmartTV Maior Média - Dataset 4 - {dataset4_hora} horas')
data = dataset4['bytes_down_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset4_mean, dataset4_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset4, offset_dataset4, scale_dataset4)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset4,'bytes_down_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()

plt.figure(figsize=(10,6))
plt.title(f'Byes Up Chromecast Maior Mediana - Dataset 5 - {dataset5_hora} horas')
data = dataset5['bytes_up_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset5_mean, dataset5_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset5, offset_dataset5, scale_dataset5)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset5,'bytes_up_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()

plt.figure(figsize=(10,6))
plt.title(f'Byes Up Chromecast Maior Média - Dataset 6 - {dataset6_hora} horas')
data = dataset6['bytes_up_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset6_mean, dataset6_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset6, offset_dataset6, scale_dataset6)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset6,'bytes_up_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()

plt.figure(figsize=(10,6))
plt.title(f'Byes Down Chromecast Maior Mediana - Dataset 7 - {dataset7_hora} horas')
data = dataset7['bytes_down_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset7_mean, dataset7_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset7, offset_dataset7, scale_dataset7)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset7,'bytes_down_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()

plt.figure(figsize=(10,6))
plt.title(f'Byes Down Chromecast Maior Média - Dataset 8 - {dataset8_hora} horas')
data = dataset8['bytes_down_log']
x = np.linspace(data.min(),data.max(),len(data))
y = norm.pdf(x, dataset8_mean, dataset8_std)
plt.plot(x,y)
x = np.linspace(0, data.max(), 100)
y = gamma.pdf(x, shape_dataset8, offset_dataset8, scale_dataset8)
plt.plot(x, y,label='Gamma')
plt.xlabel('BPS (LOG10)')
plt.ylabel('Densidade')
plt.hist(data, bins = sturges(dataset8,'bytes_down_log'), edgecolor="black",density=True)
plt.legend(["Gaussiana", "Gamma", "Histograma"])
plt.show()


#GRAFICOS DE PROBABILIDADE
fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Up SmartTV Maior Mediana - Dataset 1 - {dataset1_hora} horas')
x = dataset1.bytes_up_log
probplot(x, dist=gamma, sparams=(shape_dataset1,offset_dataset1,scale_dataset1), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset1_mean,dataset1_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Up SmartTV Maior Média - Dataset 2 - {dataset2_hora} horas')
x = dataset2.bytes_up_log
probplot(x, dist=gamma, sparams=(shape_dataset2,offset_dataset2,scale_dataset2), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset2_mean,dataset2_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Down SmartTV Maior Mediana - Dataset 3 - {dataset3_hora} horas')
x = dataset3.bytes_down_log
probplot(x, dist=gamma, sparams=(shape_dataset3,offset_dataset3,scale_dataset3), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset3_mean,dataset3_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Down SmartTV Maior Média - Dataset 4 - {dataset4_hora} horas')
x = dataset4.bytes_down_log
probplot(x, dist=gamma, sparams=(shape_dataset4,offset_dataset4,scale_dataset4), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset4_mean,dataset4_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Up Chromecast Maior Mediana - Dataset 5 - {dataset5_hora} horas')
x = dataset5.bytes_up_log
probplot(x, dist=gamma, sparams=(shape_dataset5,offset_dataset5,scale_dataset5), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset5_mean,dataset5_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Up Chromecast Maior Média - Dataset 6 - {dataset6_hora} horas')
x = dataset6.bytes_up_log
probplot(x, dist=gamma, sparams=(shape_dataset6,offset_dataset6,scale_dataset6), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset6_mean,dataset6_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Down Chromecast Maior Mediana - Dataset 7 - {dataset7_hora} horas')
x = dataset7.bytes_down_log
probplot(x, dist=gamma, sparams=(shape_dataset7,offset_dataset7,scale_dataset7), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset7_mean,dataset7_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle(f'Bytes Down Chromecast Maior Média - Dataset 8 - {dataset8_hora} horas')
x = dataset8.bytes_down_log
probplot(x, dist=gamma, sparams=(shape_dataset8,offset_dataset8,scale_dataset8), plot=axes[0])
axes[0].set_title('Gamma')
axes[0].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
probplot(x, dist=norm, sparams=(dataset8_mean,dataset8_std), plot=axes[1])
axes[1].set_title('Gaussiana')
axes[1].set(xlabel = 'Quantis', ylabel = 'Valores Ordenados')
plt.show()



#DETERMINANDO A CORRELACAO ENTRE OS DATASETS
pearson_correlation_d1_d3 = pearsonr(dataset1["bytes_up_log"], dataset3["bytes_down_log"])
pearson_correlation_d2_d4 = pearsonr(dataset2["bytes_up_log"], dataset4["bytes_down_log"])

if dataset5.shape[0] != dataset7.shape[0]:
    if dataset5.shape[0] > dataset7.shape[0]:
        dataset5 = dataset5.sample(n=dataset7.shape[0])
    else:
        dataset7 = dataset7.sample(n=dataset5.shape[0])

if dataset6.shape[0] != dataset8.shape[0]:
    if dataset6.shape[0] > dataset8.shape[0]:
        dataset6 = dataset6.sample(n=dataset8.shape[0])
    else:
        dataset8 = dataset8.sample(n=dataset6.shape[0])

pearson_correlation_d5_d7 = pearsonr(dataset7["bytes_up_log"], dataset7["bytes_down_log"])
pearson_correlation_d6_d8 = pearsonr(dataset8["bytes_up_log"], dataset8["bytes_down_log"])

print(f'Pearsons Correlation Coefficient dataset1 and dataset3: {pearson_correlation_d1_d3[0]}')
print(f'Pearsons Correlation Coefficient dataset2 and dataset4: {pearson_correlation_d2_d4[0]}')
print(f'Pearsons Correlation Coefficient dataset5 and dataset7: {pearson_correlation_d5_d7[0]}')
print(f'Pearsons Correlation Coefficient dataset6 and dataset8: {pearson_correlation_d6_d8[0]}')

plt.figure(figsize=(10,6))
plt.scatter(dataset1["bytes_up_log"], dataset3["bytes_down_log"])
plt.xlabel('Bytes Up BPS (LOG10)')
plt.ylabel('Bytes Down BPS (LOG10)')
plt.title("Dispersion Graph datasets 1 and 3")
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(dataset2["bytes_up_log"], dataset4["bytes_down_log"])
plt.xlabel('Bytes Up BPS (LOG10)')
plt.ylabel('Bytes Down BPS (LOG10)')
plt.title("Dispersion Graph datasets 2 and 4")
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(dataset7["bytes_up_log"], dataset7["bytes_down_log"])
plt.xlabel('Bytes Up BPS (LOG10)')
plt.ylabel('Bytes Down BPS (LOG10)')
plt.title("Dispersion Graph datasets 5 and 7")
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(dataset8["bytes_up_log"], dataset8["bytes_down_log"])
plt.xlabel('Bytes Up BPS (LOG10)')
plt.ylabel('Bytes Down BPS (LOG10)')
plt.title("Dispersion Graph datasets 6 and 8")
plt.show()



#REALIZANDO O TESTE G PARA OS DATASETS
def classes(dataframe, column):
    n_bins = sturges(dataframe)
    amplitude = (dataframe[column].max() - dataframe[column].min())/n_bins
    classes = [dataframe[column].min()]
    for BIN in range(n_bins):
        classes.append(classes[-1] + amplitude)
    return classes

def add_bins(biggest_dataset, smallest_dataset, column):
    bins = classes(smallest_dataset,column)
    legend = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    series_classes = pd.cut(x=biggest_dataset[column], bins = bins, labels = legend, include_lowest = True, right=True)
    biggest_dataset['Classes'] = series_classes
    series_classes = pd.cut(x=smallest_dataset[column], bins = bins, labels = legend, include_lowest = True, right=True)
    smallest_dataset['Classes'] = series_classes

add_bins(dataset1,dataset5, 'bytes_up_log')
add_bins(dataset2,dataset6, 'bytes_up_log')
add_bins(dataset3,dataset7, 'bytes_down_log')
add_bins(dataset4,dataset8, 'bytes_down_log')

dataset1_obs = dataset1['Classes'].value_counts().sort_index()
dataset5_obs = dataset5['Classes'].value_counts().sort_index()
dataset1_obs = dataset1_obs / dataset1_obs.sum()
dataset5_obs = dataset5_obs / dataset5_obs.sum()
g1 = power_divergence(dataset1_obs, dataset5_obs, lambda_='log-likelihood')

dataset2_obs = dataset2['Classes'].value_counts().sort_index()
dataset6_obs = dataset6['Classes'].value_counts().sort_index()
dataset2_obs = dataset2_obs / dataset2_obs.sum()
dataset6_obs = dataset6_obs / dataset6_obs.sum()
g2 = power_divergence(dataset2_obs, dataset6_obs, lambda_='log-likelihood')

dataset3_obs = dataset3['Classes'].value_counts().sort_index()
dataset7_obs = dataset7['Classes'].value_counts().sort_index()
dataset3_obs = dataset3_obs / dataset3_obs.sum()
dataset7_obs = dataset7_obs / dataset7_obs.sum()
g3 = power_divergence(dataset3_obs, dataset7_obs, lambda_='log-likelihood')

dataset4_obs = dataset4['Classes'].value_counts().sort_index()
dataset8_obs = dataset8['Classes'].value_counts().sort_index()
dataset4_obs = dataset4_obs / dataset4_obs.sum()
dataset8_obs = dataset8_obs / dataset8_obs.sum()
g4 = power_divergence(dataset4_obs, dataset8_obs, lambda_='log-likelihood')

print(f'G-Test datasets 1 and 5: Power Divergence = {g1[0]} ; P-Value = {g1[1]}')
print(f'G-Test datasets 2 and 6: Power Divergence = {g2[0]} ; P-Value = {g2[1]}')
print(f'G-Test datasets 3 and 7: Power Divergence = {g3[0]} ; P-Value = {g3[1]}')
print(f'G-Test datasets 4 and 8: Power Divergence = {g4[0]} ; P-Value = {g4[1]}')