# Trading_ColombianSovereignCurve
Deep Reinforcement Learning Applied in Fixed Income Trading Strategies


### Flujo de informacion - Ingenieria de Datos

Descripcion del Flujo:
"1. Utilizando "consolidar.py" se consolidan archivos provenientes del SEN de BanRep "https://www.banrep.gov.co/es/sistemas-pago/estadisticas-sen", y se eliminan datos de negociacion de otras ruedas diferentes a CONH, titulos denominados en UVR y TES verdes.
"2. Utilizando "tasas_ponderadas.py" se encuentra tasa ponderada de negociacion de cada titulo para cada dia bursatil, ponderando por valor de giro por operacion.
"3. Utilizando "valoracion.py" se valora cada uno de los titulos segun sus caracteristicas faciales, fecha de negociacion y tasa ponderada del dia. No solo se obtiene precio limpio y precio sucio, sino, ademas, duracion, duracion modificada, convezidad y DV01.
"4. Utilizando "imputacion.py" se introducen datos sinteticos de los titulos que no negociaron en determinado dia, para asi evitar tener faltantes. La justificacion a esto es que, si bien no se negociaron estos titulos, no quiere decir que no se hubiesen podido negociar, es decir, el algoritmo de DRL podria igualmente comprar y vender estos titulos en estos dias, convirtiendose en creador de mercado. Posteriormente estos sinteticos se valoraron utilizando "valoracion.py"
