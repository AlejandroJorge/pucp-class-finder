import fitz
import os
import json
import logging
from dotenv import load_dotenv
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import firebase_admin
from firebase_admin import firestore

from baml_client.sync_client import b
from baml_client.config import set_log_level
from baml_client.types import Course

set_log_level("OFF")
logging.basicConfig(
    level=logging.INFO
)


load_dotenv()

pdfs_dir_path = os.path.join(os.path.curdir, "raw-pdfs")
artifacts_dir_path = os.path.join(os.path.curdir, "artifacts")

def initialize_firestore():
    try:
        cred = firebase_admin.credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
        logging.info("Firebase connection done.")
    except ValueError:
        logging.warning("Firebase application already initialized. Skipping")
    except Exception as ex:
        logging.error(f"Firebase application error: {ex}")
        exit(1)

def process_and_upload(filename: str):
    made_llm_call = False

    txt_artifact_path = os.path.join(artifacts_dir_path, f"{filename}.txt")

    logging.info(f"Transforming {filename}.pdf to {filename}.txt")

    txt_content = ""
    if os.path.exists(txt_artifact_path):
        with open(txt_artifact_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
    else:
        if os.path.exists(os.path.join(pdfs_dir_path, f"{filename}.pdf")):
            with fitz.open(os.path.join(pdfs_dir_path, f"{filename}.pdf")) as doc:
                curr_txt = ""
                for page in doc:
                    curr_txt += page.get_textpage().extractText()
                with open(txt_artifact_path, 'w+', encoding='utf-8') as f:
                    f.write(curr_txt)
                txt_content = curr_txt
        else:
            raise Exception(f"Couldn't find {filename}.pdf")

    logging.info(f"Extracted {filename}.txt successfully")

    logging.info(f"Extracting {filename}.txt to {filename}.json")

    json_artifact_path = os.path.join(artifacts_dir_path, f"{filename}.json")

    structured_content = ""
    if os.path.exists(json_artifact_path):
        with open(json_artifact_path, 'r', encoding='utf-8') as f:
            structured_content = Course.model_validate_json(f.read())
    else:
        made_llm_call = True
        structured_content = b.ExtractCourse(txt_content)
        with open(json_artifact_path, 'w+', encoding='utf-8') as f:
            json.dump(structured_content.model_dump(), f, indent=4)

    logging.info(f"Extracted {filename}.json successfully")

    logging.info(f"Uploading {filename} to firestore")

    if not firebase_admin._apps:
        firebase_admin.initialize_app()

    db = firestore.client()

    doc_ref = db.collection("courses").document(structured_content.code)
    if doc_ref.get().exists:
        logging.info(f"Document {structured_content.code} already exists")
        return

    doc_ref.set(structured_content.model_dump())

    logging.info(f"Processed {filename} successfully")

    if made_llm_call:
        time.sleep(random.uniform(10, 15))

def main():
    if not os.path.exists(pdfs_dir_path):
        raise Exception("Couldn't find raw pdfs directory")

    if not os.path.exists(artifacts_dir_path):
        os.mkdir(artifacts_dir_path)

    pdf_filenames = [filename[:-4] for filename in os.listdir(pdfs_dir_path) if filename[-3:].lower() == 'pdf']

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_filename = {executor.submit(process_and_upload, filename): filename for filename in pdf_filenames}
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                future.result()
                logging.info(f"Processed {filename} successfully")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()


