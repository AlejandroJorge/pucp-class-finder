import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

import firebase_admin
from firebase_admin import credentials, firestore
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from baml_client.sync_client import b
from baml_client.types import Course
from baml_client.config import set_log_level

set_log_level("OFF")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "courses"
MODEL_NAME = 'all-mpnet-base-v2'

try:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
    logging.info("Firebase connection done.")
except ValueError:
    logging.warning("Firebase application already initialized. Skipping.")

db = firestore.client()


try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    encoder = SentenceTransformer(MODEL_NAME)
    logging.info("Clientes de Qdrant y SentenceTransformer inicializados.")
except Exception as e:
    logging.error(f"Error durante la inicialización de clientes: {e}")
    exit(1)


def expand_query_with_gemini(user_prompt: str) -> str:
    logging.info("Llamando a la API de Gemini para refinar la consulta...")
    
    try:
        response = b.GenerateCourseDescription(user_prompt)
        refined_query = response.strip()
        
        logging.info("Consulta refinada por Gemini obtenida exitosamente.")
        return refined_query
    except Exception as e:
        logging.error(f"Error en la llamada a la API de Gemini: {e}")
        return ""

def search_courses(query_text: str, top_k: int = 6) -> List[Course]:
    if not query_text:
        logging.error("La consulta refinada está vacía. Abortando búsqueda.")
        return []
        
    logging.info("Vectorizando la consulta refinada...")
    query_vector = encoder.encode(query_text).tolist()

    logging.info(f"Buscando los {top_k} cursos más similares en Qdrant...")
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    
    if not search_results:
        logging.warning("No se encontraron resultados en Qdrant.")
        return []

    logging.info(f"Cursos extraidos de Qdrant: {[search_result.payload.get('code') if search_result.payload else None for search_result in search_results]}")

    course_codes = [
        hit.payload['code'] for hit in search_results if hit.payload and 'code' in hit.payload
    ]

    if not course_codes:
        logging.warning("Los resultados de Qdrant no contenían códigos de curso en el payload.")
        return []

    logging.info(f"Recuperando información completa de {len(course_codes)} cursos desde Firestore...")
    docs_stream = db.collection("courses").where("code", "in", course_codes).stream()

    firestore_results_map = {doc.to_dict()['code']: doc.to_dict() for doc in docs_stream}

    final_courses = [firestore_results_map.get(code) for code in course_codes if firestore_results_map.get(code)]
    
    return [course for course in final_courses if course is not None]

def post_process_courses(user_prompt: str, courses: List[Course]) -> List[Dict[str, Any]]:
    if not courses:
        return []

    logging.info(f"Llamando a Gemini para re-ordenar y explicar {len(courses)} cursos...")

    try:
        response = b.PostProcessCourses(user_prompt, courses)

        logging.info(f"Re-ranking completado. {len(response)} cursos finales.")

        final_courses = []
        for course_explanation in response:
            c = next(
                (
                    c for c in courses
                    if (
                        (isinstance(c, dict) and c.get('code') == course_explanation.code)
                        or
                        (hasattr(c, 'code') and c.code == course_explanation.code)
                    )
                ),
                None
            )
            if c is None:
                continue
            course_dict = c.model_dump() if hasattr(c, 'model_dump') else dict(c)
            course_dict['explanation'] = course_explanation.explanation
            course_dict['favourable_factors'] = course_explanation.favourable_factors
            course_dict['unfavourable_factors'] = course_explanation.unfavourable_factors
            final_courses.append(course_dict)

        return final_courses

    except Exception as e:
        logging.error(f"Error en el proceso de re-ranking con Gemini: {e}")
        return []

if __name__ == "__main__":
    prompt = "Quiero ser gerente general en algun futuro de una importante empresa peruana como un banco o corporacion"

    refined_query = expand_query_with_gemini(prompt)
    print("==================================================")
    print("Refined query: ", refined_query)

    initial_recommendations = search_courses(refined_query)
    print("==================================================")
    print("Initial recommendations: ", len(initial_recommendations))

    final_recommendations = post_process_courses(prompt, initial_recommendations)
    print("==================================================")
    print("Final recommendations: ", len(final_recommendations))

    print("==================================================")
    for i, course in enumerate(final_recommendations, 1):
        print(f"{i}. {course.get('name', '')} (Código: {course.get('code', '')})")
        print(f"   Recomendación: {course.get('explanation', '')}")
        print(f"   Factores favorables: {course.get('favourable_factors', [])}")
        print(f"   Factores desfavorables: {course.get('unfavourable_factors', [])}")
        syllabus = course.get('syllabus', [])
        if isinstance(syllabus, list):
            topics = [unit.get('title', '') for unit in syllabus if isinstance(unit, dict)]
            print(f"   Temas cubiertos: {topics}")
        print()

    print("==================================================")
    print()
