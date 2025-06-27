import os
import uuid
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import fitz
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import firestore
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from baml_client.sync_client import b
from baml_client.config import set_log_level
from baml_client.types import Course

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
set_log_level("OFF")
load_dotenv()

# --- Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDFS_DIR_PATH = os.path.join(BASE_DIR, "raw-pdfs")
ARTIFACTS_DIR_PATH = os.path.join(BASE_DIR, "artifacts")

# --- Parámetros de Ingesta
BATCH_SIZE = 32
MAX_WORKERS = 4 # Número de hilos para procesar PDFs

# --- Configuración de Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "courses"
VECTOR_SIZE = 384  # Dimensión del modelo 'all-MiniLM-L6-v2'

UUID_NAMESPACE = uuid.UUID("a55a2530-9223-4477-a378-b17173e3a473")

# --- Inicialización de Clientes y Modelos
try:
    # Inicializa Firebase (una sola vez)
    cred = firebase_admin.credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
    logging.info("Firebase connection done.")
except ValueError:
    logging.warning("Firebase application already initialized. Skipping.")

db = firestore.client()
encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ==============================================================================
# FUNCIONES
# ==============================================================================

def process_pdf_to_course(filename: str) -> tuple[Course, bool]:
    """
    Función de trabajo para los hilos. Procesa un solo PDF a un objeto Course.
    Devuelve el objeto Course y un booleano indicando si se hizo una llamada al LLM.
    """
    # 1. Extracción de PDF a TXT (con caché)
    txt_artifact_path = os.path.join(ARTIFACTS_DIR_PATH, f"{filename}.txt")
    if os.path.exists(txt_artifact_path):
        with open(txt_artifact_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
    else:
        pdf_path = os.path.join(PDFS_DIR_PATH, f"{filename}.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Couldn't find {filename}.pdf")
        with fitz.open(pdf_path) as doc:
            txt_content = "".join(page.get_textpage().extractText() for page in doc)
        with open(txt_artifact_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
    logging.info(f"[{filename}] TXT extraído.")

    # 2. Extracción de TXT a JSON (con caché y LLM)
    json_artifact_path = os.path.join(ARTIFACTS_DIR_PATH, f"{filename}.json")
    made_llm_call = False
    if os.path.exists(json_artifact_path):
        with open(json_artifact_path, 'r', encoding='utf-8') as f:
            structured_content = Course.model_validate_json(f.read())
    else:
        structured_content = b.ExtractCourse(txt_content, datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
        made_llm_call = True
        with open(json_artifact_path, 'w+', encoding='utf-8') as f:
            # Usamos model_dump_json para un formato JSON limpio
            f.write(structured_content.model_dump_json(indent=4))
    logging.info(f"[{filename}] JSON extraído.")
    
    return structured_content, made_llm_call

def upload_batch(batch: List[Course]):
    """
    Toma un lote de objetos Course, los vectoriza y los sube a Qdrant y Firestore.
    """
    if not batch:
        return
        
    logging.info(f"Procesando lote de {len(batch)} cursos para carga.")

    # --- 1. Vectorización por lotes ---
    texts_to_encode = [
        (f"Curso: {c.name}. Facultad: {c.faculty}. Resumen: {c.summary}. "
         f"Resultados de Aprendizaje: {'. '.join(c.learningOutcomes)}. "
         f"Contenido del Sílabo: {'. '.join(topic.title for topic in c.syllabus)}. "
         f"Bibliografía: {'. '.join(c.bibliography)}. "
         f"Código del Curso: {c.code}. "
         f"Créditos: {c.credits}.")
        for c in batch
    ]
    vectors = encoder.encode(texts_to_encode, show_progress_bar=False).tolist()

    # --- 2. Carga por lotes a Qdrant ---
    qdrant_points = [
        models.PointStruct(
            id=str(uuid.uuid5(UUID_NAMESPACE, course.code)),
            vector=vector,
            payload={'code': course.code}
        )
        for course, vector in zip(batch, vectors)
    ]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points, wait=False)
    logging.info(f"Lote enviado a Qdrant.")

    # --- 3. Carga por lotes a Firestore ---
    firestore_batch = db.batch()
    for course in batch:
        doc_ref = db.collection("courses").document(course.code)
        firestore_batch.set(doc_ref, course.model_dump())
    firestore_batch.commit()
    logging.info(f"Lote enviado a Firestore.")


def main():
    """
    Orquesta el proceso de extracción paralela y carga por lotes.
    """
    if not os.path.exists(PDFS_DIR_PATH):
        raise Exception("Couldn't find raw pdfs directory")
    if not os.path.exists(ARTIFACTS_DIR_PATH):
        os.mkdir(ARTIFACTS_DIR_PATH)

    # Asegura que la colección de Qdrant exista
    try:
        collection_exists = qdrant_client.collection_exists(collection_name=COLLECTION_NAME)
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            logging.info(f"Colección '{COLLECTION_NAME}' en Qdrant creada.")

        qdrant_client.create_payload_index(collection_name=COLLECTION_NAME, field_name='code', field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception:
        logging.warning(f"No se pudo crear la colección '{COLLECTION_NAME}'")

    pdf_filenames = [fn[:-4] for fn in os.listdir(PDFS_DIR_PATH) if fn.lower().endswith('.pdf')]
    courses_batch = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_filename = {executor.submit(process_pdf_to_course, fn): fn for fn in pdf_filenames}

        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                # Recolecta el resultado del hilo
                course, made_llm_call = future.result()
                courses_batch.append(course)
                
                logging.info(f"[{filename}] Procesado y listo para el lote.")

                # Si el lote está lleno, súbelo y vacíalo
                if len(courses_batch) >= BATCH_SIZE:
                    upload_batch(courses_batch)
                    courses_batch = [] # Reinicia el lote

                # Mantiene la pausa si se usó el LLM
                if made_llm_call:
                    time.sleep(random.uniform(5, 10))

            except Exception as e:
                logging.error(f"Error procesando {filename}: {e}")

    # Sube cualquier curso restante en el último lote parcial
    if courses_batch:
        upload_batch(courses_batch)

    logging.info("--- Proceso completado ---")


if __name__ == "__main__":
    main()