import torch
import torchvision.transforms as transforms

from data_model import ExampleDataset
from model import ExampleModel
from visualization import preprocess_image, visualize_predictions

data_dir = "./animal_dataset/train"
model_path = "model/animal_6.pth"
test_image_path = "./animal_dataset/test/gatto/1001.jpeg"
image_size = (128, 128)
num_classes = 10


def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


# Settings
transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = ExampleModel(num_classes=num_classes)
with torch.no_grad():
    model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# Dataset
dataset = ExampleDataset(data_dir)

# Inference
original_image, image_tensor = preprocess_image(test_image_path, transform)
probabilities = predict(model, image_tensor, device)

# Visualize
class_names = dataset.classes
visualize_predictions(original_image, probabilities, class_names)
