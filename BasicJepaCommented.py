# Entonces quiero construir un tokenizer y dos encoders to feed into a small transformer. 
#- encoder1 mira solo los visibles y crea los embedding de contexto que veria el modelo en una situacion rea
#- encoder2 mira todos los tokens y crea embeddings correctos para los target tokens, quedandonos solo con los embeddings
#para los tokens escondidos (van a funcionar como targets para el preedictor)
#-el transformer tiene una arquitectura lightweight, ya que la mayoria del trabajo se deberia hacer en el encoder de contexto. 


import argparse   # ni idea la verdad, lo tengo que mirar 
import copy       # imagino que para copiar el databse de incio 
import math       # funciones, operaciones y constantes 
import random     # para generar combinaciones aleatorios de tokens en los batches 
                  # integer in [a, b] INCLUDING both ends
from pathlib import Path  # me deja poner las cosas en ciertos folders 

import torch                      # duh
import torch.nn as nn             # tengo que mirarlo tambien 
import torch.nn.functional as F   # el funcional, ya que me permite hacer operaciones y parse sobre las funiones de cada attention 
#head 
import torchvision                # datasets (CIFAR-10) and image transforms
import torchvision.transforms as T
from PIL import Image             # Pillow: reads image files into PIL image objects
from torch.utils.data import DataLoader, Dataset # batching and data-loading 

#constantes 
IMG=32 
PATCH=4
GRID=IMG//PATCH 
N_PATCH=GRID*GRID 


#Una clase es un molde para crear objetos a lo largo del script. Cada objeto 
#reciibe datos, los manipula y los guarda pudiendo manipularlos de nuevo. 
#es como un state machine lowkey, me imagino que el analogo seria como crear un 
#una turing machine dentro de python aunque no se si la analogia es mierda 

#Primero voy a crear la clase para cada bloque de transformadores con atencion
#nn.Module es la base para todos los neural networls y layers dentro de PyTorch 

class Block(nn.Module): 

    def __init__(self, dim, heads):
        #Esto me da el estado inicial para cada layer que llame 
        #Self lo tienen todas las clases, no che que es 
        #Self apunta hacia los datos internos de cada objeto para que no se confundan los 
        #datos entre bloques de transformadores 

        #dim es el numero de dimensiones en las que trabaja el espacio de embedding 
        #heads es el numero de attention heads, las dimensiones tienen que ser divisibles entre 
        # el numero de heads, me imagino que para poder dividir dimensiones de manera completa 
        
        #inicializa el nn.Module lo que me da acceso a trackear parametros 
        #me deja mover todo a la gpu, y cambiar entre modo de entreno y de evaluacion 

        super().__init__() 

        self.n1= nn.LayerNorm(dim) 

        #LayerNorm es una funcion que normaliza vectores usando su media y varianza, 
        #por lo que vectores con el mismo spread relative spread (1,2,5) y (100,200,500)
        #terminan siendo el mismo vector. Para que sirve actually, ni puta idea

        #Vale entonces al parecer, evita que los numeros se vayan de control, guardando el 
        #spread (resta la media y divide por la desviacion estandar) dejando 
        #de media cero y varianza 1, se usa el propio vector para evitar ruido cruzado 

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        #Aqui es donde pasa el attention, donde se mira el token junto al resto para 
        # que el modelo "entienda" el contexto. Divide el vector en varias cabezas que se fijan cada una en una cosa
        # de aqui es de donde apracen las dimensiones de los embeddings me imagino, para eso es 
        # que el numero de dimensiones tenia que ser divisible entre heads

        self.n2 = nn.LayerNorm(dim)

        #Una segunda capa de normalizacion para pasarla antes del mlp, me imagino que esto 
        #normalizara otra vez por lo mismo de antes, para que no se salgan de control los numerinos
        

        self.mlp = nn.Sequential(nn.Linear(dim,4*dim), nn.GELU(), nn.Linear(4*dim,dim)) 

        #El mlp es un multi-layer perceptron. Creo que hace falt entender primero 
        #las funciones y despues lo que hace
        # - nn.Sequential(a,b,c) deja claro el orden de aplicacion de funciones aplicando a,b y c de manera sequenciaol 
        # - nn.Linear expande las dimesiones (los evalua contra cada neurona del perceptron)
        # - nn.GELU toma los detrctores con valores negativos (en los que la neurona activada no coincide con el token 
        # introducido) y los manda a cero
        # - El linear de vuelta usa los weights para multiplicar el vector y por lo tanto le inyecta el contexto
        # asociado (token de salida)

        #entonces despues del Attention que procesa tokens juntos, el mlp procesa los tokens 
        #de manera separada para dar el siguiente token con xbloque = x + xattn + xmlp
        #Ahora pasa para el siguiente bloque hasta que llega al ultimo que usa 
        # el diccionario para transformar el token en probabilidades 

        def forward(self,x): #Donde x es (B,N,D)

            h = self.n1(x) #Habiamos definido antes que .n1() es la layernorm

            attn_out = self.atn(h,h,h,need_weights=False)[0]

            # esto simplemente realiza la self-attention, ya que el mismo tensor 
            # es el query, key and value que devuelve (attn_output, attn-weights)
            # los weights que pedimos que no calcule son los attention weights 
            # estos nos entregan la "relacion gramatical" entre palabras, es mas bien 
            # como informacion de relacion que se usa pero no queremos que devuelva

            x = x + attn_out 
            x = x + self.mlp(self.n2(x)) 
            return x
        # Esta funcion es el transformador en si, realizando la normalizacion 
        # y el paso por las dos redes devolviendo el token con la informacion propia 
        # y de su contexto, lista para pasar por el siguiente bloque 

class Encoder(nn.Module)


    