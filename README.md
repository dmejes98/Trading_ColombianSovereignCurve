# Deep Reinforcement Learning Applied in Fixed Income Trading Strategies 
### Aprendizaje Reforzado Profundo para la Administración de Portafolios de Renta Fija
Trabajo de grado para Maestría en Ciencias de los Datos y Analítica
Universidad EAFIT
Estudiante: David Mejía Estrada
Directora: Paula María Almonacid Hurtado
Medellín, 2023

##### Url del repositorio
En [este repositorio](https://github.com/dmejes98/Trading_ColombianSovereignCurve) puede encontrar el código completo utilizado para la construcción del trabajo de grado referenciado. Todo el código está escrito en lenguaje de programación "python". 
En el repositorio se puede encontrar "requirements.txt", que contiene las librerías necesarias para ejecutar los modelos. Se debe tener en cuenta que la versión de python a utilizar debe ser 3,7 o inferior.

##### Flujo de informacion - Ingenieria de Datos 

Descripcion del Flujo. Dentro de la carpeta "Data" se encuentran los siguientes archivos de texto: 
1. Utilizando "consolidar.py" se consolidan archivos provenientes del [SEN de BanRep](https://www.banrep.gov.co/es/sistemas-pago/estadisticas-sen), y se realiza limpieza de datos.
2. Utilizando "tasas_ponderadas.py" se encuentra tasa ponderada de negociación de cada título para cada día bursátil, ponderando por valor de giro por operación. 
3. Utilizando "valoracion.py" se valora cada uno de los títulos segun sus características faciales, fecha de negociación y tasa ponderada del día. No solo se obtiene precio limpio y precio sucio, sino, además, duración, duración modificada, convexidad y DV01. 
4. Utilizando "imputacion.py" se introducen datos sintéticos de los títulos que no negociaron en determinado día, para así evitar tener faltantes. La justificación a esto es que, si bien no se negociaron estos títulos, no quiere decir que no se hubiesen podido negociar, es decir, el algoritmo de DRL podría igualmente comprar y vender estos títulos en estos dias, convirtiéndose en creador de mercado. Posteriormente estos sintéticos se valoraron utilizando "valoracion.py"

En esta carpeta se deben ubicar los archivos de excel con los datos provenientes del Banco de la República.

##### Modelos de DRL

Dentro de la carpeta "Model" se encuentran los siguientes archivos de texto:
* "EnvTES_train.py": Este es el entorno con el que los diferentes modelos de DRL interactúan durante su etapa de entrenamiento.
* "EnvTES_val.py": Este es el entorno con el que los diferentes modelos de DRL interactúan durante su etapa de validación.
* "EnvTES_trade.py": Este es el entorno con el que los diferentes modelos de DRL interactúan durante su etapa de negociación con datos no antes vistos.
* "model.py": En este se encuentra la ejecución central de los modelos, incluyendo el modelo de ensamble. 

En esta carpeta se deben ubicar los archivos de excel "consolidado_total.xlsx" y "data.xlsx", con la información lista tras el proceso de ingeniería de datos y de tablas maestras necesarias para la ejecución.

La ejecución de los modelos arroja el excel "Resultados.xlsx" en esta misma carpeta. También, llena las carpetas "csv", "images" y "Working", con los resultados de los modelos discriminados por etapas de entrenamiento, validación y testeo (negociación).

##### .gitignore

No se suben al repositorio archivos de excel ni algunas carpetas. Dentro de los archivos ignorados se encuentra el libro de excel "uso modelos.xlsx", en donde se encuentra un resumen completo de los resultados de la ejecución de los modelos.

Toda esta información que no es cargada al repositorio se adjuta al Comité de la Maestría, a través del [formulario para la entrega del proyecto](https://forms.office.com/r/Qd9SSGCPfr) en la sección de anexos.

### Disclaimer

Este trabajo de grado tiene fines únicamente académicos. Si bien el objetivo es realizar aportes científicos sobre aplicaciones de machine learning al mercado de valores, en ningún momento lo dicho en este trabajo debe entenderse explícita o implícitamente como una recomendación de inversión.
La inversión en los mercados financieros puede derivar en una pérdida parcial o total del patrimonio invertido, y, por tanto, debe estar debidamente asesorada por un profesional del mercado de valores certificado y vigilado por el **Autorregulador del Mercado de Valores (AMV)** y la **Superintendencia Financiera de Colombia (SFC)**.
 
