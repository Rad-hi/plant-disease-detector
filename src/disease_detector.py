import torch
import torchvision.transforms as transforms

from resnet_9 import ResNet9


class DiseaseDetector:
    '''
    Wrapper class for the ResNet9 Classifier model that's trained on the plant disease
    dataset. This wrapper aims at simplifying the interface with the ML model.
    '''

    CLASSES = [
        'Apple: Apple scab',
        'Apple: Black rot',
        'Apple: Cedar apple rust',
        'Apple: healthy',
        'Blueberry: healthy',
        'Cherry[including sour]: Powdery mildew',
        'Cherry[including sour]: healthy',
        'Corn[maize]: Gray leaf spot',
        'Corn[maize]: Common rust ',
        'Corn[maize]: Northern Leaf Blight',
        'Corn[maize]: healthy',
        'Grape: Black rot',
        'Grape: Esca (Black Measles)',
        'Grape: Leaf blight (Isariopsis Leaf Spot)',
        'Grape: healthy',
        'Orange: Haunglongbing (Citrus greening)',
        'Peach: Bacterial spot',
        'Peach: healthy',
        'Pepper[bell]: Bacterial spot',
        'Pepper[bell]: healthy',
        'Potato: Early blight',
        'Potato: Late blight',
        'Potato: healthy',
        'Raspberry: healthy',
        'Soybean: healthy',
        'Squash: Powdery mildew',
        'Strawberry: Leaf scorch',
        'Strawberry: healthy',
        'Tomato: Bacterial spot',
        'Tomato: Early blight',
        'Tomato: Late blight',
        'Tomato: Leaf Mold',
        'Tomato: Septoria leaf spot',
        'Tomato: Spider mites',
        'Tomato: Target Spot',
        'Tomato: Tomato Yellow Leaf Curl Virus',
        'Tomato: Tomato mosaic virus',
        'Tomato: healthy'
    ]
    NUM_CLASSES: int = len(CLASSES)
    NUM_INPUTS: int = 3

    def __init__(self, model_path: str) -> None:
        '''
        Read in the saved model, and prepare if for use
        '''
        self._model = ResNet9(self.NUM_INPUTS, self.NUM_CLASSES)
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()

        # Create a transformation function that would turn images into tensors
        self._transform = transforms.ToTensor()

    def predict(self, img):
        '''
        Use the model to predict the input image
        '''
        input_tensor = self.__preprocess(img)
        yb = self._model(input_tensor)
        _, pred = torch.max(yb, dim=1)
        res = self.__postprocess(pred[0].item())
        return res

    def __preprocess(self, img) -> torch.Tensor:
        '''
        Apply any transformations that are necessary to the input image in order to make it
        in the form that the model expects
        (in our case, all we need to do is make it into a Tensor, but we might want to check
        if it's the correct shape, and make it into the correct one if it's not ...)
        '''
        tensor_from_img = self._transform(img).unsqueeze(0)
        return tensor_from_img

    def __postprocess(self, pred: int) -> str:
        '''
        Use the output of the model and interpret in any way suits you, in our case,
        we simply return the corresponding label to the output index
        '''
        return self.CLASSES[pred]
