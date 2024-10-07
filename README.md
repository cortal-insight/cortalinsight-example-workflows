# cortalinsight-example-workflows

## Fine-Tuning GPT-4o for Image Classification

This repo provides you with easy-to-follow scripts and workflows for performing image classification using GPT-4o, to handle vision-related tasks that are domain specific like defect detection, manufacturing visual inspection, quality control and more.

To view more information, please have a look at the blog:
Fine-Tuning GPT-4o for Image Classification: [click here](https://open.substack.com/pub/cortalinsight/p/fine-tune-gpt-4o-vision-models-for?r=36v0p6&utm_campaign=post&utm_medium=web)
This blog provides detailed insights and step-by-step instructions for using the scripts in this repository to fine-tune GPT-4o for image classification tasks.

🧠 Steps to fine-tune GPT-4o on your custom dataset.
- Prepare your image classification dataset for GPT-4o fine-tuning
- Upload datasets (large or small) using the OpenAI API for fine-tuning
- Submit a fine-tune GPT-4o job for your specific vision tasks

📂 Dataset Directory Structure

Let's first structure your dataset. The script expects the following directory format:

    root_directory/
    │
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    ├── class2/
    │   ├── image1.jpeg
    │   ├── image2.webp
    │   └── ...
    └── classN/
        ├── image1.jpg
        ├── image2.png
        └── ...


- Accepted image formats: .jpeg, .jpg, .png, .webp
- Ensure images are in RGB or RGBA mode

⚙️ How to Use the Scripts

For fine-tuning:-

Before running the script, you need to manually update the following fields in classification_data_workflow.py:

    main_directory = "/path/to/images"  # Replace with your main directory path
    output_file = "prepared_dataset_classification.jsonl"  # Output as JSONL format
    api_key = "{Insert API key here}"

    python3 classification_data_workflow.py


For testing:-

Before running the script, you need to manually update the following fields in test_image_classfication.py:

    api_key =  "{Insert API key here}"
    model = "gpt-4o" or "custom fine-tuned model"
    single_image_path = "/path/to/images.jpg"

    python3 test_image_classfication.py


