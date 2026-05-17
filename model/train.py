import torch
import joblib
import pandas as pd
from torch import nn
from pathlib import Path 
from sklearn.metrics import classification_report,confusion_matrix


class PatientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=31,out_features=64)
        self.layer2 = nn.Linear(in_features=64,out_features=64)
        self.layer3 = nn.Linear(in_features=64,out_features=3)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x1 = self.dropout(self.relu(self.layer1(x)))
        x2 = self.dropout(self.relu(self.layer2(x1)))
        return self.layer3(x2)
    
if __name__ == "__main__":
    torch.manual_seed(42)
    model = PatientModel()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params= model.parameters(),lr=0.001)

    X_train = joblib.load('model/X_train.pkl')
    X_test = joblib.load('model/X_test.pkl')
    y_train = joblib.load('model/y_train.pkl')
    y_test = joblib.load('model/y_test.pkl')

    
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values


    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    epochs = 1000

    for epoch in range(epochs):
        model.train()
        train_logits = model(X_train)
        train_preds = torch.argmax(train_logits,dim=1)
        loss = loss_fn(train_logits,y_train)
        optim.zero_grad()
        loss.backward()
        optim.step()

        model.eval()
        with torch.inference_mode():  
            test_logits = model(X_test)
            test_preds = torch.argmax(test_logits,dim=1)
            test_loss = loss_fn(test_logits,y_test)

        if epoch % 100 == 0 :
            print(f"Epoch :{epoch},Train_Loss{loss:.2f},Test_Loss{test_loss :.2f}")

    model.eval()
    with torch.inference_mode():
        pred_logits = model(X_test)
        y_preds = torch.argmax(pred_logits,dim=1)


    y_preds = y_preds.detach().numpy()
    y_test = y_test.detach().numpy()


    
    classification = classification_report(y_test,y_preds)
    con_mat = confusion_matrix(y_test,y_preds)

  
    print(classification)
    print(con_mat)


    model_path = Path('Models')
    model_path.mkdir(parents = True,exist_ok= True)
    model_name = "Patients_Readmission_Multiclass_Classifcation_neural_network_model.pth"
    model_save_path = model_path/model_name
    torch.save(obj = model.state_dict(),f=model_save_path)