function GenerateCourseDescription(user_prompt: string) -> string {
  client CustomGemini

  prompt #"
    {{_.role("user")}}

    Actúa como un experto redactor de catálogos universitarios de ingeniería y ciencias de la computación.
    Dado el siguiente perfil de intereses y metas de un usuario, genera un párrafo conciso y descriptivo que resuma un curso universitario ideal para esta persona.
    Utiliza terminología técnica precisa y conceptos clave del dominio de la informática. No uses un tono conversacional.
    Enfócate en temas, tecnologías y habilidades.

    {{_.role("model")}}
    Entendido. Por favor, proporcióname el perfil del usuario.

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
