import base64
import os
from io import BytesIO
from typing import Optional
from PIL import Image
from openai import OpenAI

# Insert custom classes to test
classes = ['pitted_surface', 'inclusion', 'crazing', 'unclear']

class Config:
    def __init__(self, api_key: str, image_dir: str):
        self.api_key = api_key
        self.image_dir = image_dir

class ImageClassification:
    def __init__(self, config: Config):
        self.api_key = config.api_key
        self.client = OpenAI(api_key=self.api_key)
        self.image_dir = config.image_dir

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            image = Image.open(image_path)
            print(f"Image {image_path} loaded successfully.")
            return image
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error loading image: {e}")
        return None

    def image_to_base64(self, image: Image.Image) -> str:
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return ""

    def predict(self, INSTRUCTION_PROMPT: str, base64_image: str) -> str:
        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION_PROMPT  
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=payload,
                max_tokens=300,
            )

            result = response.choices[0].message.content
            return result

        except Exception as e:
            print(f"Error during API prediction: {e}")
            return "Prediction failed"

    def classify_single_image(self, image_path: str):

        instruction_prompt = (
            "You are an inspection assistant for a manufacturing plant. "
            "Analyze the provided image of steel surfaces and classify it based on the kind of defect. "
            f"There are four classes: {classes}. "
            "You must always return only one option from that list."
            "If you are not sure, choose 'unclear'."
        )

        # Load and classify a single image
        image = self.load_image(image_path)
        if image:
            base64_img = self.image_to_base64(image)
            if base64_img:
                result = self.predict(instruction_prompt, base64_img)
                print(f"Classification result for {image_path}: {result}")
            else:
                print("Failed to convert image to base64.")
        else:
            print("Failed to load image.")

if __name__ == "__main__":
    
    api_key = '{Insert API key here}'
    config = Config(api_key=api_key, image_dir="images")
    classifier = ImageClassification(config)

    # Specify the single image path to classify
    single_image_path = '/path/to/image.jpg'
    classifier.classify_single_image(single_image_path)
