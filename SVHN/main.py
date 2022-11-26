from trainer import  Trainer

from models.data_module import SVHNData
from  models.net import  Lnet,MLP


def main(model_name = "MLP"):
    data = SVHNData(root="dataset")
    model = MLP()
    if model_name =="LNet":
        model= Lnet()
    if model_name =="MLP":
        model = MLP()

    trainer =Trainer(max_epochs=20)
    trainer.fit(model,data)
    trainer.save()

if __name__ =="__main__":
    model_name = "LNet"
    main(model_name)