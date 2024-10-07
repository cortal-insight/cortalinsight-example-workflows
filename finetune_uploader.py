import os
import math
import requests
from openai import OpenAI

# Upload using Files API (for files ≤ 512 MB)
def upload_file(client, file_path):
    """Upload the prepared JSONL file using  Files API."""
    try:
        with open(file_path, "rb") as f:
            response = client.files.create(
                file=f,
                purpose="fine-tune"
            )
        print(f"File uploaded successfully: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error during file upload: {e}")
        return None

# Create session using the Uploads API (for files > 512 MB)
def create_upload(api_key, file_name, file_size):

    url = "https://api.openai.com/v1/uploads"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        "filename": file_name,
        "purpose": "fine-tune",
        "mime_type": "application/json",
        "bytes": file_size 
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Upload part 
def upload_part(api_key, upload_id, file_part, part_number, total_parts):

    url = f"https://api.openai.com/v1/uploads/{upload_id}/parts"

    files = {
        'data': ('part', file_part)
    }

    response = requests.post(url, headers={'Authorization': f'Bearer {api_key}'}, files=files)
    if response.status_code != 200:
        raise Exception(f"Failed to upload part: {response.json()}")
    
    part_id = response.json()["id"]

    return part_id

# Complete the upload 
def complete_upload(api_key, upload_id, part_ids):

    url = f"https://api.openai.com/v1/uploads/{upload_id}/complete"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        "part_ids": part_ids  # Ordered list of part IDs
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"Failed to complete upload: {response.json()}")
    
    file_id = response.json()["file"]["id"]

    return file_id

# Upload large files (> 512 MB) in parts using Uploads API
def upload_large_file(api_key, file_path):
    """Upload a large file in parts using Uploads API."""
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    upload_session = create_upload(api_key, file_name, file_size)

    if 'id' not in upload_session:
        print(f"Error creating upload session: {upload_session}")
        return None

    upload_id = upload_session['id']
    print(f"Created upload session: {upload_id}")

    part_size = 50 * 1024 * 1024  # 5 MB part size
    total_parts = math.ceil(file_size / part_size)

    part_ids = []
    part_number = 0
    with open(file_path, 'rb') as f:
        while True:
            file_part = f.read(part_size)
            if not file_part:
                break
            part_number += 1
            part_id = upload_part(api_key, upload_id, file_part, part_number, total_parts)
            print(f"\tUploaded part {part_number}/{total_parts}: {part_id}")
            part_ids.append(part_id)
            
    upload_id = complete_upload(api_key, upload_id, part_ids)
    print(f"Upload completed / File ID: {upload_id}")
    return upload_id

def upload_dataset(api_key, file_path):

    client = OpenAI(api_key=api_key)

    file_size = os.path.getsize(file_path)

    if file_size <= 512 * 1024 * 1024:  # ≤ 512 MB
        print(f"File size {file_size / (1024 * 1024):.2f} MB: Using Files API for upload.")
        return upload_file(client, file_path)
    else:  # > 512 MB
        print(f"File size {file_size / (1024 * 1024):.2f} MB: Using Uploads API for multipart upload.")
        return upload_large_file(api_key, file_path)

# Start fine-tuning job after file upload
def start_fine_tuning_job(api_key, uploaded_file_id, model="gpt-4o-2024-08-06"):

    client = OpenAI(api_key=api_key)
    
    try:
        fine_tuning_response = client.fine_tuning.jobs.create(
            training_file=uploaded_file_id,
            model=model
        )
        return fine_tuning_response
        
    except Exception as e:
        print(f"Error starting fine-tuning job: {e}")

