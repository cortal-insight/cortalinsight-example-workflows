import os
import json
import base64
from io import BytesIO
from PIL import Image
from finetune_uploader import upload_dataset, start_fine_tuning_job

# Accepted image formats & constraints
ACCEPTED_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.webp')
MAX_IMAGES_PER_EXAMPLE = 10
MAX_IMAGE_SIZE_MB = 10
MAX_EXAMPLES = 50000
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

class ImageProcessor:
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image object to a base64 encoded string."""
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return ""

    @staticmethod
    def process_image(image_path: str):
        try:

            img = Image.open(image_path)

            if not image_path.lower().endswith(ACCEPTED_EXTENSIONS):
                return None, "Unsupported file format"

            if os.path.getsize(image_path) > MAX_IMAGE_SIZE_BYTES:
                return None, f"Image exceeds {MAX_IMAGE_SIZE_MB} MB"

            if img.mode not in ('RGB', 'RGBA'):
                return None, "Image is not in RGB or RGBA mode"

            img_base64 = ImageProcessor.image_to_base64(img)

            return img_base64, None
        except Exception as e:
            return None, f"Error processing image: {e}"

class DatasetPreparer:

    def __init__(self, main_directory: str, output_file: str):
        self.main_directory = main_directory
        self.output_file = output_file
        self.class_stats = {}
        self.total_examples = 0


    def generate_defect_json(self, image_url: str, defect_class: str):
        return {
            "messages": [
                { 
                    "role": "system", 
                    "content": "You are an assistant that identifies uncommon defects on steel surfaces." 
                },
                { 
                    "role": "user", 
                    "content": "What kind of defect is this?" 
                },
                { 
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                },
                { 
                    "role": "assistant", 
                    "content": defect_class 
                }
            ]
        }


    def print_stats(self):
        """ Print stats """

        print(f"\nDataset preparation complete. {self.total_examples} examples written to {self.output_file}.\n")
        print('Total number of classes: ', len(self.class_stats.keys()))
        print('Classes: ', self.class_stats.keys())

        print("Processing Stats Per Class:")
        
        for class_name, stats in self.class_stats.items():
            print(f"\nClass '{class_name}':")
            print(f"  Processed: {stats['processed']} images")
            print(f"  Skipped: {stats['skipped']} images")
            for reason, count in stats["skipped_reasons"].items():
                print(f"    Skipped due to {reason}: {count} images")

    def prepare_dataset(self):
        """Process all images and prepare dataset"""

        try:
            with open(self.output_file, 'w') as f:

                for class_dir in os.listdir(self.main_directory):
                    class_path = os.path.join(self.main_directory, class_dir)

                    if not os.path.isdir(class_path):
                        continue

                    class_stats[class_dir] = {
                        "processed": 0,
                        "skipped": 0,
                        "skipped_reasons": {}
                    }

                    for image_file in os.listdir(class_path):
                        image_path = os.path.join(class_path, image_file)
                        img_base64, error = ImageProcessor.process_image(image_path)

                        if img_base64:
                            
                            defect_json = self.generate_defect_json(f"data:image/jpeg;base64,{img_base64}", class_dir)
                            f.write(json.dumps(defect_json) + '\n')
                            class_stats[class_dir]["processed"] += 1
                            total_examples += 1

                            if self.total_examples >= MAX_EXAMPLES:
                                print(f"Reached maximum example limit of {MAX_EXAMPLES}. Stopping.")
                                break
                        else:
                            self.class_stats[class_dir]["skipped"] += 1

                            if error not in self.class_stats[class_dir]["skipped_reasons"]:
                                self.class_stats[class_dir]["skipped_reasons"][error] = 0
                            self.class_stats[class_dir]["skipped_reasons"][error] += 1

                    if self.total_examples >= MAX_EXAMPLES:
                        print(f"Reached maximum example limit of {MAX_EXAMPLES}. Stopping.")
                        break

            self.print_stats()

        except Exception as e:
            print(f'Something went wrong: {e}')

class ModelTrainer:

    @staticmethod
    def upload_dataset_node(api_key, file_path):

        uploaded_file_id = upload_dataset(api_key, file_path)
        
        if uploaded_file_id:
            print(f"Dataset uploaded successfully. File ID: {uploaded_file_id}")
            return uploaded_file_id
        else:
            print("File upload failed.")
            return None

    @staticmethod
    def submit_finetuning_job(api_key, uploaded_file_id, model="gpt-4o-2024-08-06"):
        if uploaded_file_id:
            start_fine_tuning_job(api_key, uploaded_file_id, model)
        else:
            print("No uploaded file to fine-tune.") 

# Example usage
if __name__ == "__main__":
    main_directory = "/path/to/images"  # Replace with your main directory path
    output_file = "prepared_dataset_classification.jsonl"  # Output as JSONL format
    api_key = '{Insert API key here}'

    dataset_preparer = DatasetPreparer(main_directory, output_file)
    print('\n1) Preparing the dataset...')
    dataset_preparer.prepare_dataset()

    # Upload dataset
    print('\n2) Uploading dataset...')
    uploaded_file_id = ModelTrainer.upload_dataset_node(api_key, output_file)

    # Submit fine-tuning job
    print('\n3) Submit fine-tuning job.')
    if uploaded_file_id:
        ModelTrainer.submit_finetuning_job(api_key, uploaded_file_id)

    
