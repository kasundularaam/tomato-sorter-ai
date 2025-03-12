import os
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import shutil

# Constants
INPUT_DIR = 'tomatoes'
OUTPUT_DIR = 'processed'
MODEL_PATH = 'best_tomato_model.pth'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.7


def load_model(model_path, device):
    # Create a ResNet-50 model
    model = models.resnet50(weights=None)

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)  # 2 classes: ripe and damaged
    )

    # Load weights and prepare model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)

    return model


def process_images(input_dir, output_dir, model, device):
    # Define transformation for inference
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create output directories
    ripe_dir = os.path.join(output_dir, 'ripe')
    damaged_dir = os.path.join(output_dir, 'damaged')
    uncertain_dir = os.path.join(output_dir, 'uncertain')

    for dir_path in [ripe_dir, damaged_dir, uncertain_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and
        any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Initialize counters
    counts = {'ripe': 0, 'damaged': 0, 'uncertain': 0}

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load and transform the image
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = nn.Softmax(dim=1)(outputs)
                confidence, pred_idx = torch.max(probabilities, 1)
                confidence = confidence.item()

                # Map class index to class name
                pred_class = ['ripe', 'damaged'][pred_idx.item()]

            # Determine destination directory
            if confidence >= CONFIDENCE_THRESHOLD:
                dest_dir = ripe_dir if pred_class == 'ripe' else damaged_dir
                counts[pred_class] += 1
            else:
                dest_dir = uncertain_dir
                counts['uncertain'] += 1

            # Copy the image to the appropriate directory
            dest_path = os.path.join(dest_dir, image_file)
            shutil.copy2(image_path, dest_path)

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    # Print summary
    print("\nProcessing complete!")
    print(f"Classified as ripe: {counts['ripe']} images")
    print(f"Classified as damaged: {counts['damaged']} images")
    print(
        f"Uncertain (below confidence threshold): {counts['uncertain']} images")
    print(f"\nResults saved to {output_dir}")


def main():
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist")
        return

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} does not exist")
        return

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    print(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH, device)

    # Process images
    process_images(INPUT_DIR, OUTPUT_DIR, model, device)


if __name__ == '__main__':
    main()
