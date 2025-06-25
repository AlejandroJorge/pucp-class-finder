import fitz
import os
from dotenv import load_dotenv
import getpass
from google import genai
import threading
import json
import time
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    GEMINI_API_KEY = getpass.getpass("Please insert your gemini api key to continue: ")

GEMINI_MODEL=os.getenv("GEMINI_MODEL")
if GEMINI_MODEL is None:
    GEMINI_MODEL = "gemini-2.0-flash-lite"
    assert GEMINI_MODEL is not None

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

pdfs_dir_path = os.path.join(os.path.curdir, "raw-pdfs")
json_dir_path = os.path.join(os.path.curdir, "json-output")

def initialize_firestore():
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
        print("Firebase connection done.")
    except ValueError:
        print("Firebase application already initialized. Skipping")
    except Exception as ex:
        print(f"Firebase application error: {ex}")
        exit(1)

def transform_pdf_to_text(filename: str) -> str:
    curr_pdf_path = os.path.join(pdfs_dir_path, f"{filename}.pdf")
    text_output = ""

    print(f"{threading.get_ident()}: Parsing {filename} to text")
    with fitz.open(curr_pdf_path) as doc:
        print(f"{threading.get_ident()}: Reading from {curr_pdf_path}")
        for page in doc:
            text_output += page.get_textpage().extractText()
    print(f"{threading.get_ident()}: Parsed {filename} to text")

    return text_output

def transform_text_to_json(text: str):
    prompt = f"""
        The following is unstructured text extracted from a pdf of a sillabus, I need you to transform it to a json. Answer ONLY with json, not a single introduction word since this output will be parsed.

        Unstructured data:
        {text}

        Json expected fields:
        - codigo
        - nombre
        - facultad
        - cantidad_creditos
        - horas_clase:
            - tipo
            - cantidad
        - horas_practica:
            - tipo
            - cantidad
        - horas_laboratorio:
            - tipo
            - cantidad
        - profesores[]
        - planes_donde_se_dicta[]:
            - especialidad
            - etapa
        - cursos_requisito[]:
            - codigo
            - nombre
            - tipo_requisito
        - descripcion
        - objetivos[]
        - contenidos[]:
            - nro_capitulo
            - titulo
            - cantidad_estimada_horas
            - subcapitulos[]
    """
    print(f"{threading.get_ident()}: Making LLM request")
    raw_text_response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    ).text
    print(f"{threading.get_ident()}: Got LLM response")
    if raw_text_response is None:
        raise Exception("No response from LLM")
    json_text = raw_text_response[7:-3]


    return json_text

def sync_with_firestore(data: dict):
    if not firebase_admin._apps:
        initialize_app()

    db = firestore.client()

    codigo = data.get('codigo')
    if codigo is None:
        raise Exception("Couldn't get 'codigo' field from parsed json")

    try:
        print(f"Writing {codigo} to firestore")
        doc_ref = db.collection("classes").document(codigo)
        doc_ref.set(data)
        print(f"Written {codigo} to firestore successfully")
    except Exception as ex:
        print(f"Couldn't sync class with 'codigo': {codigo} because: {ex}")

def pipeline(filename: str):
    try:
        text_content = transform_pdf_to_text(filename)
        unparsed_json = transform_text_to_json(text_content)
        parsed_json = json.loads(unparsed_json)
        sync_with_firestore(parsed_json)
        time.sleep(50) # Gemini free tier
    except Exception as ex:
        print(f"{threading.get_ident()}: Error while processing {filename}: {ex}")

def chained_pipeline(*filenames: str):
    for filename in filenames:
        pipeline(filename)

def main():
    if not os.path.exists(pdfs_dir_path):
        raise Exception("Couldn't find raw pdfs directory")

    if not os.path.exists(json_dir_path):
        os.mkdir(json_dir_path)

    for preprocessing_file in os.listdir(pdfs_dir_path):
        if preprocessing_file[-3:].lower() == 'pdf':
            curr_path = os.path.join(pdfs_dir_path, preprocessing_file)
            os.rename(curr_path, curr_path.lower())

    filenames = [filename[:-4] for filename in os.listdir(pdfs_dir_path) if filename[-3:] == 'pdf']

    n = 20
    filenames_chunks = [filenames[i * (len(filenames) // n) + min(i, len(filenames) % n):(i + 1) * (len(filenames) // n) + min(i + 1, len(filenames) % n)] for i in range(n)]

    threads: list[threading.Thread] = []
    for i in range(n):
        t = threading.Thread(target=chained_pipeline, args=filenames_chunks[i])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()

