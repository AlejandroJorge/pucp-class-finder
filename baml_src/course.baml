class Course {
  code string @description(#"6 characters like 1INF23"#)
  name string
  credits float
  faculty string @description(#"Full name like Estudios Generales Ciencias or Facultad de Ciencias e Ingenieria"#)
  year int
  term 0 | 1 | 2
  
  summary string
  learningOutcomes string[] @description(#"Usually known as objetivos de aprendizaje"#)
  syllabus Unit[] @description(#"Usually known as UNIDAD under PROGRAMA ANALITICO"#)

  programs Program[]
  professors string[]

  lectureHours int?
  lectureFrequency "weekly" | "biweekly"?
  practiceHours int?
  practiceFrequency "weekly" | "biweekly"?
  laboratoryHours int?
  laboratoryFrequency "weekly" | "biweekly"?

  evaluationFormula string @description(#"In format 0.4Pa + 0.3Pb + 0.3Ex"#)

  bibliography string[] @description(#"Name and editorial list"#)

  extractedAt string @description(#"
    Timestamp in format YYYY-MM-DDTHH:MM:SSZ
  "#)
}

class Unit {
  title string
  hours int
  topics_covered string[]
}

class CoursePrerequisite {
  code string
  name string
}

class CreditsPrerequisite {
  value int
}

class Program {
  name string
  isElective bool
  prerequisites (CoursePrerequisite | CreditsPrerequisite)[]
}

function ExtractCourse(raw_text: string) -> Course {
  client CustomGemini

  prompt #"
    Extract from this content:
    {{ raw_text }}

    {{ ctx.output_format }}
  "#
}

test example_course {
  functions [ExtractCourse]
  args {
    raw_text #"
      
FACULTAD DE
CIENCIAS E
INGENIERÍA
PROBABILIDAD Y
ESTADÍSTICA
2025-1
I.
INFORMACIÓN GENERAL
CURSO
PROBABILIDAD Y ESTADÍSTICA
CLAVE
1EST22
CRÉDITOS
3.5
HORAS DE DICTADO
CLASE: 3 Semanal
PRACTICA: 2 Quincenal
EXAMEN:
HORARIO
TODOS
PROFESORES
CRISTIAN LUIS BAYES RODRIGUEZ
ALEX DE LA CRUZ HUAYANAY
II.
PLANES CURRICULARES DONDE SE DICTA EL CURSO
ESPECIALIDAD
ETAPA
NIVEL
CARÁCTER
REQUISITOS
INGENIERÍA
MECATRÓNICA
PREGRADO EN
FACULTAD
5
OBLIGATORIO  1MAT09 CÁLCULO
APLICADO [07]
INGENIERÍA INFORMÁTICA PREGRADO EN
FACULTAD
6
OBLIGATORIO  1MAT09 CÁLCULO
APLICADO [07]
Tipos de requisito
�� 04 = Haber cursado o cursar simultáneamente
�� 05 = Haber aprobado o cursar simultáneamente
�� 06 = Promedio de notas no menor de 08
�� 07 = Haber aprobado el curso
III.
DESCRIPCIÓN DEL CURSO
Este curso aporta a las siguientes competencias de la carrera de Ingeniería informática:
C1. Resolución de problemas: Caracteriza, analiza y modela los problemas u oportunidades de la
organización y sociedad a través del enfoque de procesos, riesgos y mejora continua para determinar
necesidades de automatización de datos e información y la generación de conocimientos mediante
tecnologías informáticas que apoyen a la toma de decisiones.
C6. Trabajo experimental
Diseña, conduce y analiza experimentos de un tema o problema relevante de su interés sustentado
en literatura académica, analizando e interpretando datos mediante métodos
adecuados y coherentes con el paradigma, elaborando conclusiones y recomendaciones basadas en
el conocimiento y comprensión de hechos esenciales, conceptos, principios y teorías relativas a la
informática.
IV.
SUMILLA
Es un curso teórico-práctico que brinda una introducción a las principales técnicas estadísticas, con
el objetivo de obtener información que ayude a la toma de decisiones, a partir del análisis de datos.
Se presentan técnicas estadísticas univariadas y bivariadas. Se brinda una introducción a la teoría de
probabilidad y los principales modelos probabilísticos. Por otro lado, se da una introducción a la
estadística inferencial, incluyendo su aplicación a modelos de regresión lineal simple y múltiple.
V.
OBJETIVOS
FACULTAD DE CIENCIAS E INGENIERÍA
1EST22 - PROBABILIDAD Y ESTADÍSTICA
Una vez aprobado el curso, el alumno habrá alcanzado los siguientes objetivos:
- Adquirirá los conocimientos de probabilidad e inferencia estadística, que son utilizados al abordar el
análisis de datos.
- Podrá hacer aplicaciones básicas de la estadística inferencial empleando técnicas de estimación
puntual,
de intervalo y prueba de hipótesis.
- Podrá utilizar modelos probabilísticos en situaciones de carácter aleatorio, producto de fenómenos
naturales y de la intervención del hombre.
- Estará en capacidad de elaborar, analizar, validar e interpretar modelos de regresión lineal simple y
múltiple en situaciones reales.
- Aprenderá el uso básico de un software para aplicaciones estadísticas, así como a interpretar
resultados
proporcionados por este.
VI.
PROGRAMA ANALÍTICO
CAPÍTULO 1
PROBABILIDAD (6 horas)
Conceptos básicos: experimento aleatorio, espacio muestral, eventos, álgebra de eventos.
Probabilidad:
definición y propiedades. Cálculo de probabilidad. Probabilidad condicional. Eventos independientes.
Teoremas de probabilidad total y de Bayes. Independencia.
CAPÍTULO 2
VARIABLE ALEATORIA (9 horas)
Variables aleatorias discretas y continuas. Función de distribución. Cuantiles. Esperanza y desviación
estándar. Simulación de variables aleatórias y aplicaciones con R. Modelos Discretos: Bernoulli,
Binomial y
Poisson. Modelos continuos: Exponencial, Gamma y Normal. Exploración de datos: histograma,
gráfico de
cuantiles.
CAPÍTULO 3
VECTOR ALEATORIO (6 horas)
Distribución conjunta, marginal y condicional para dos variables aleatorias discretas o continuas.
Independencia. Modelo Multinomial. Modelo Normal Multivariado.
CAPÍTULO 4
INFERENCIA ESTADÍSTICA (9 horas)
Muestra aleatoria. Convergencia en distribución y en probabilidad. Estimador puntual: propiedades.
Estimación por máxima verosimilitud. Criterios de comparación de modelos. Estimación por intervalo
para la media, proporción y varianza. Estimación del tamaño de muestra. Prueba de hipótesis
relativas a la media, proporción y varianza.
CAPÍTULO 5
MODELO DE REGRESIÓN LINEAL (12 horas)
El modelo de regresión lineal múltiple: supuestos del modelo. Ajuste del modelo por el método de
mínimos
cuadrados. Análisis de varianza. Idoneidad del modelo. Intervalos de confianza y pruebas de
hipótesis sobre
los parámetros del modelo. Intervalos de confianza para la media y de predicción. Prueba de
hipótesis lineal
general. Verificación de supuestos usando residuales. Variables indicadoras.
VII. METODOLOGÍA
Las clases son expositivas. En estas, se estudian las aplicaciones relacionadas con el campo de la
ingeniería.
Asimismo, se promueve el trabajo en equipo y se incentiva el uso del programa estadístico R.
Consideraciones a tener en cuenta sobre la evaluación:
La evaluación continua y/o los exámenes pueden ser en aula o en casa, individuales o grupales.
Oportunamente se indicará como se realizarán, siempre con una anticipación no menor de dos
semanas de la fecha de entrega, evaluación en línea o realización en algún ambiente de la
Universidad.
VIII. EVALUACIÓN
FACULTAD DE CIENCIAS E INGENIERÍA
1EST22 - PROBABILIDAD Y ESTADÍSTICA
Sistema de evaluación
N° Codigo
Tipo de
Evaluación
Cant.
Eval.
Forma de
aplicar los
pesos
Pesos
Cant. Eval.
Eliminables
Consideracion
es adicionales
Observaciones
1
Pa
Práctica tipo
A
4
Por Promedio Pa=3
1
2
Ex
Examen
2
Por
Evaluación
Ex1=3
Ex2=4
Modalidad de evaluación: 2
Fórmula para el cálculo de la nota final
 ( 3Pa + 3Ex1 + 4Ex2 ) / 10
Aproximación de los promedios parciales No definido
Aproximación de la nota final No definido
IX.
BIBLIOGRAFÍA
Referencia obligatoria
X.
POLÍTICA CONTRA EL PLAGIO
Para la corrección y evaluación de todos los trabajos del curso se va a tomar en cuenta el debido
respeto a los derechos de autor, castigando severamente cualquier indicio de plagio con la nota
CERO (00). Estas medidas serán independientes del proceso administrativo de sanción que la
facultad estime conveniente de acuerdo a cada caso en particular. Para obtener más información,
referirse a los siguientes sitios en internet
www.pucp.edu.pe/documento/pucp/plagio.pdf
    "#
  }
}