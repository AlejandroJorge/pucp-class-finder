function GenerateCourseDescription(user_prompt: string) -> string {
  client CustomGemini

  prompt #"
    {{_.role("user")}}

    Actúa como un experto diseñador de currículas y redactor de catálogos para facultades de ingeniería y ciencias.
    Tu tarea es, a partir de un perfil de intereses de un usuario, diseñar un curso universitario ideal y generar una descripción completa en un **formato de texto plano muy específico y en una sola línea**.

    El objetivo es que esta descripción sea directamente compatible con una base de datos de búsqueda vectorial, por lo que el formato, la puntuación y la estructura deben seguirse de manera exacta.

    **Perfil del Usuario:**
    {{ user_prompt }}

    ---

    **FORMATO DE SALIDA OBLIGATORIO:**

    Debes generar una **única cadena de texto continuo**, sin saltos de línea. La cadena debe seguir esta plantilla exacta, prestando especial atención a los separadores (un punto y un espacio ". "):

    `Curso: [Nombre del curso]. Facultad: [Nombre de la facultad]. Resumen: [Párrafo de resumen]. Resultados de Aprendizaje: [Resultado 1]. [Resultado 2]. [Resultado 3...]. Contenido del Sílabo: [Tema 1]. [Tema 2]. [Tema 3...]. Bibliografía: [Referencia 1]. [Referencia 2]. [Referencia 3...]. Código del Curso: [Código del curso]. Créditos: [Número de créditos].`

    **REGLAS IMPORTANTES:**
    - **SEPARADORES**: Usa ". " (punto, espacio) para separar cada campo y cada elemento dentro de las listas (Resultados de Aprendizaje, Sílabo, Bibliografía).
    - **SIN EXPLICACIONES**: Tu respuesta debe comenzar directamente con "Curso:" y terminar con el número de créditos. No añadas texto introductorio, saludos, ni los corchetes `[]`.

    ---

    **EJEMPLO PERFECTO DE SALIDA:**

    `Curso: Arquitectura de Computadoras. Facultad: Facultad de Ciencias e Ingeniería. Resumen: El curso es de naturaleza teórico-práctico cuyo propósito es que el estudiante comprenda el funcionamiento interno de la computadora. Se desarrollan temas como el análisis de circuitos combinacionales y secuenciales, la unidad central de proceso (CPU), memorias, dispositivos de entrada-salida y dispositivos de almacenamiento masivo. Resultados de Aprendizaje: Describir la interacción entre el CPU, la memoria y los diferentes módulos funcionales de un computador para elegir la plataforma más adecuada para las distintas aplicaciones que se puedan presentar. Implementar algoritmos con lenguajes de programación en bajo (diseño de equipos basados en microprocesadores) y alto nivel (uso de minicomputadoras, sistemas operativos pequeños, etc.), a partir de conjunto de instrucciones de una arquitectura específica para evaluar el rendimiento de un sistema computacional. Evaluar el desempeño de un algoritmo en una arquitectura específica por medio del análisis estadístico de resultados experimentales. Contenido del Sílabo: CIRCUITOS COMBINACIONALES Y SECUENCIALES. UNIDAD CENTRAL DE PROCESAMIENTO. ARQUITECTURA DE COMPUTADORA. ARQUITECTURA INTEL X86 DE 64 BITS. PROGRAMACIÓN EN ENSAMBLADOR X86-64. Bibliografía: Floyd, Thomas L. Digital Fundamentals Harlow Pearson Education Limited, 2015. Hennessy, John L. Organización y diseño de computadoras; la interfaz hardware / software Madrid McGraw-Hill, 1995. Intel Corp IA-32 Intel Architecture Software Developer's Manual Volumes 1,2 y 3 http://developer.intel.com/design/pentium4/manuals/245470.htm. Null, Linda. The essentials of computer organization and architecture Sudbury, Mass. : Jones and Bartlett Publishers, 2003. Stallings, William. Organización y arquitectura de computadores : diseño para optimizar prestaciones Madrid : Prentice Hall, 1996. Código del Curso: 1ELE01. Créditos: 3.5.`

    {{_.role("model")}}
    Entendido. Generaré una única cadena de texto continuo, sin saltos de línea y siguiendo estrictamente la plantilla y el formato proporcionado, basándome en el perfil del usuario.

    {{_.role("user")}}
    {{ user_prompt }}

    {{ ctx.output_format }}
  "#
}

class CourseExplanation {
  code string
  explanation string @description(#"
    Justificación del curso en base a la consulta original. Alrededor de 4 o 6 frases.
  "#)
  favourable_factors string[] @description(#"
    Factores que hacen que el curso sea relevante para la consulta original. Entre 2 y 4
  "#)
  unfavourable_factors string[] @description(#"
    Factores que hacen que el curso no sea relevante para la consulta original. Entre 2 y 4
  "#)
}

function PostProcessCourses(original_prompt: string, courses: Course[]) -> CourseExplanation[] {
  client CustomGemini

  prompt #"
    {{_.role("user")}}
    Recomiendame cursos en base a la siguiente consulta:
    {{ original_prompt }}

    {{_.role("model")}}
    Estos son los cursos candidatos:

    {% for course in courses %}
        {{course}}
    {% endfor %}

    {{_.role("user")}}
    Sobre estas tareas realizame lo siguiente:

    1. Reordena los cursos de más a menos relevante según la consulta original.
    2. Descarta cualquier curso que sea claramente irrelevante.
    3. Para cada curso relevante, escribe una justificación referenciando su informacion proporcionada.
    4. Para cada curso relevante, escribe los factores favorables y desfavorables que hacen que el curso sea relevante o no.

    Devuelve el resultado en el siguiente formato:
    {{ ctx.output_format }}
  "#
}
