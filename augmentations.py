import torchvision.transforms as transforms

class TrainTransform(object):
    def __init__(self):

        self.transform= transforms.Compose(
            [   transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )
        pass

    def __call__(self, sample):
        return self.transform(sample)



