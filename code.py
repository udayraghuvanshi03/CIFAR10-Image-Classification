import math
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR10():
    def __init__(self,cifar_file):
        #Hyper paarams
        num_epochs=18
        batch_size=200
        learning_rate=0.01

        #Dataset
        path=cifar_file
        train_batch_files=['data_batch_1','data_batch_2','data_batch_3','data_batch_4']
        validation_file='data_batch_5'
        test_file='test_batch'

        class Image_dataset(Dataset):
            def __init__(self,data,labels,file_name):
                self.data=torch.tensor(data,dtype=torch.float32)
                self.data=self.data/255.0
                normalize_fun = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                self.data = normalize_fun(self.data)
                one_hot_labels = LabelBinarizer().fit_transform(labels)
                self.labels = torch.tensor(one_hot_labels, dtype=torch.float32)
                self.n_samples=self.data.shape[0]
                # classes = ('plane', 'car', 'bird', 'cat',
                #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
                class_dic={}
                class_dic[0] = 'plane'
                class_dic[1]='car'
                class_dic[2]='bird'
                class_dic[3]='cat'
                class_dic[4]='deer'
                class_dic[5]='dog'
                class_dic[6] = 'frog'
                class_dic[7] = 'horse'
                class_dic[8] = 'ship'
                class_dic[9] = 'truck'

                per_class_labels,counts=torch.unique(torch.where(self.labels==1)[1],return_counts=True)
                print(f'There are total {len(per_class_labels)} classes in {file_name}')
                print(f'There are total {self.n_samples} samples in {file_name}set')
                for i,j in zip(per_class_labels,counts):
                    print(f'Number of instances of {class_dic[i.item()]} in {file_name} is {j.item()}')
                print('\n')

            def __getitem__(self, index):
                return self.data[index],self.labels[index]

            def __len__(self):
                return self.n_samples

        train_data = []
        train_labels = []

        for file_name in train_batch_files:
            file_path=os.path.join(path,file_name)
            with open(file_path,'rb') as f:
                train_batch_data=pickle.load(f, encoding='bytes')
                train_data.append(train_batch_data[b'data'])
                train_labels.append(train_batch_data[b'labels'])
        train_data_final=np.concatenate(train_data,axis=0).astype(np.float32)
        train_labels_final=np.concatenate(train_labels,axis=0).astype(np.float32)
        train_data_final=train_data_final.reshape(-1,3,1024)
        train_data_final=train_data_final.reshape(-1,3,32,32)

        file_path=os.path.join(path,validation_file)
        with open(file_path,'rb') as f:
            validation_batch_data = pickle.load(f, encoding='bytes')
            validation_data=validation_batch_data[b'data']
            validation_labels=validation_batch_data[b'labels']
        validation_data=validation_data.reshape(-1,3,1024).astype(np.float32)
        validation_data=validation_data.reshape(-1,3,32,32).astype(np.float32)

        with open(file_path,'rb') as f:
            test_batch_data=pickle.load(f,encoding='bytes')
            test_data=test_batch_data[b'data']
            test_labels=test_batch_data[b'labels']
        test_data=test_data.reshape(-1,3,1024).astype(np.float32)
        test_data=test_data.reshape(-1,3,32,32).astype(np.float32)

        #Dataset and Dataloader
        train_dataset=Image_dataset(train_data_final,train_labels_final,'train set')
        validation_dataset=Image_dataset(validation_data,validation_labels,'validation set')
        test_dataset=Image_dataset(test_data,test_labels,'test set')

        train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
        validation_loader=DataLoader(dataset=validation_dataset,batch_size=batch_size,shuffle=False)
        test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

        class_dic = {}
        class_dic[0] = 'plane'
        class_dic[1] = 'car'
        class_dic[2] = 'bird'
        class_dic[3] = 'cat'
        class_dic[4] = 'deer'
        class_dic[5] = 'dog'
        class_dic[6] = 'frog'
        class_dic[7] = 'horse'
        class_dic[8] = 'ship'
        class_dic[9] = 'truck'

        class ConvNeuralNet(nn.Module):
            def __init__(self):
                super(ConvNeuralNet, self).__init__()
                self.conv1=nn.Conv2d(3,6,(5,5))
                self.pool=nn.MaxPool2d(2,2)
                self.conv2=nn.Conv2d(6,16,(5,5))
                self.l1=nn.Linear(16*5*5,120)
                self.l2=nn.Linear(120,84)
                self.l3=nn.Linear(84,10)

            def forward(self,x):
                out=self.pool(F.relu(self.conv1(x)))
                out=self.pool(F.relu(self.conv2(out)))
                out=out.view(-1,16*5*5)
                out=F.relu(self.l1(out))
                out=F.relu(self.l2(out))
                out=self.l3(out)
                return out

        model=ConvNeuralNet().to(device)

        #loss and criterion
        criterion=nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

        # training the model and validating to get best model
        n_total_steps=len(train_loader)
        best_val_loss=float('inf')
        best_model=None
        num_class=10
        train_loss_per_epoch=[]
        valid_loss_per_epoch = []

        for epoch in range(num_epochs):
            for i, (imgs, lab) in enumerate(train_loader):
                batch_correct=0
                imgs=imgs.to(device)
                lab=lab.to(device)
                #forward pass
                op=model(imgs)
                loss=criterion(op,lab)
                #Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (i+1)%100==0:
                #     print(f'epoch {epoch+1}/{num_epochs}, Step [{i+1}/{n_total_steps}], loss: {loss.item(): .4f}')

            # Evaluating on training set
            train_loss=0.0
            all_batch_mean_acc_tr=[]
            all_batch_mean_per_class_acc_tr=[0]*num_class
            count=0
            for f,l in train_loader:
                mean_per_class_acc_per_batch_tr=[0]*num_class
                class_correct_per_batch_tr = [0] * num_class
                class_total_per_batch_tr = [0] * num_class
                f=f.to(device)
                l=l.to(device)
                inv_l=torch.where(l==1)[1]
                train_output=model(f)
                train_loss+=criterion(train_output,l).item()
                _,train_pred=torch.max(train_output.data,1)
                correct_batch_samples_tr=(train_pred == inv_l).sum().item()

                for c in range(num_class):
                    class_correct_per_batch_tr[c]=((train_pred==inv_l)&(inv_l==c)).sum().item()
                    class_total_per_batch_tr[c]=(inv_l==c).sum().item()
                    if class_total_per_batch_tr[c] != 0:
                        mean_per_class_acc_per_batch_tr[c] = class_correct_per_batch_tr[c] / class_total_per_batch_tr[c]

                for m in range(num_class):
                    all_batch_mean_per_class_acc_tr[m] += mean_per_class_acc_per_batch_tr[m]

                mean_acc_per_batch_tr = correct_batch_samples_tr / l.size(0)
                all_batch_mean_acc_tr.append(mean_acc_per_batch_tr)

            all_batch_mean_per_class_acc_tr = [ele / len(train_loader) for ele in all_batch_mean_per_class_acc_tr]
            train_loss = train_loss / len(train_loader)
            all_batch_mean_acc_tr = sum(all_batch_mean_acc_tr) / len(train_loader)

            # if epoch == math.floor(num_epochs / 2):
            #     print(f'Epoch {epoch+1}, Mean classification accuracy, training set: {(all_batch_mean_acc_tr) * 100 :.4f}%')
            #     print(f'Epoch {epoch+1}, Mean per class accuracy, training set: {class_dic[0]}: {all_batch_mean_per_class_acc_tr[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_tr[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_tr[2] * 100 : .4f}% | {class_dic[3]}: {all_batch_mean_per_class_acc_tr[3] * 100 : .4f}% | {class_dic[4]}: {all_batch_mean_per_class_acc_tr[4] * 100 : .4f}% |'
            #           f' {class_dic[5]}: {all_batch_mean_per_class_acc_tr[5] * 100 : .4f}% | {class_dic[6]}: {all_batch_mean_per_class_acc_tr[6] * 100 : .4f}% | {class_dic[7]}: {all_batch_mean_per_class_acc_tr[7] * 100 : .4f}% | {class_dic[8]}: {all_batch_mean_per_class_acc_tr[8] * 100 : .4f}% | {class_dic[9]}: {all_batch_mean_per_class_acc_tr[9] * 100 : .4f}%')
            if epoch%5==0:
                print(f'Epoch {epoch}, Mean classification accuracy, training set: {(all_batch_mean_acc_tr) * 100 :.4f}%')
                print(f'Epoch {epoch}, Mean per class accuracy,  training set: {class_dic[0]}: {all_batch_mean_per_class_acc_tr[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_tr[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_tr[2] * 100 : .4f}% | {class_dic[3]}: {all_batch_mean_per_class_acc_tr[3] * 100 : .4f}% | {class_dic[4]}: {all_batch_mean_per_class_acc_tr[4] * 100 : .4f}% |'
                    f' {class_dic[5]}: {all_batch_mean_per_class_acc_tr[5] * 100 : .4f}% | {class_dic[6]}: {all_batch_mean_per_class_acc_tr[6] * 100 : .4f}% | {class_dic[7]}: {all_batch_mean_per_class_acc_tr[7] * 100 : .4f}% | {class_dic[8]}: {all_batch_mean_per_class_acc_tr[8] * 100 : .4f}% | {class_dic[9]}: {all_batch_mean_per_class_acc_tr[9] * 100 : .4f}%')

            valid_loss = 0.0
            with torch.no_grad():
                all_batch_mean_acc_valid = []
                all_batch_mean_per_class_acc_valid = [0]*num_class
                for images, labels in validation_loader:
                    mean_per_class_acc_per_batch_valid = [0]*num_class
                    class_correct_per_batch_valid = [0] * num_class
                    class_total_per_batch_valid = [0] * num_class
                    images = images.to(device)
                    labels = labels.to(device)
                    valid_output = model(images)
                    inv_labels = torch.where(labels == 1)[1]
                    valid_loss += criterion(valid_output, labels).item()
                    _, valid_pred = torch.max(valid_output.data, 1)
                    correct_batch_samples_valid = (valid_pred == inv_labels).sum().item()

                    for c in range(num_class):
                        class_correct_per_batch_valid[c]=((valid_pred==inv_labels)&(inv_labels==c)).sum().item()
                        class_total_per_batch_valid[c]=(inv_labels==c).sum().item()
                        if class_total_per_batch_valid[c] != 0:
                            mean_per_class_acc_per_batch_valid[c] = class_correct_per_batch_valid[c]/class_total_per_batch_valid[c]

                    for m in range(num_class):
                        all_batch_mean_per_class_acc_valid[m] += mean_per_class_acc_per_batch_valid[m]

                    mean_acc_per_batch_valid = correct_batch_samples_valid / labels.size(0)
                    all_batch_mean_acc_valid.append(mean_acc_per_batch_valid)

                all_batch_mean_per_class_acc_valid = [ele/len(validation_loader) for ele in all_batch_mean_per_class_acc_valid]
                valid_loss = valid_loss / len(validation_loader)
                all_batch_mean_acc_valid = sum(all_batch_mean_acc_valid) / len(validation_loader)

                # if epoch == math.floor(num_epochs / 2):
                #     print(f'Epoch {epoch+1}, Mean classification accuracy, validation set: {(all_batch_mean_acc_valid) * 100 :.4f}%')
                #     print(f'Epoch {epoch+1}, Mean per class accuracy, validation set: {class_dic[0]}: {all_batch_mean_per_class_acc_valid[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_valid[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_valid[2] * 100 : .4f}% | {class_dic[3]}: {all_batch_mean_per_class_acc_valid[3] * 100 : .4f}% | {class_dic[4]}: {all_batch_mean_per_class_acc_valid[4] * 100 : .4f}% |'
                #         f' {class_dic[5]}: {all_batch_mean_per_class_acc_valid[5] * 100 : .4f}% | {class_dic[6]}: {all_batch_mean_per_class_acc_valid[6] * 100 : .4f}% | {class_dic[7]}: {all_batch_mean_per_class_acc_valid[7] * 100 : .4f}% | {class_dic[8]}: {all_batch_mean_per_class_acc_valid[8] * 100 : .4f}% | {class_dic[9]}: {all_batch_mean_per_class_acc_valid[9] * 100 : .4f}%')
                if epoch%5==0:
                    print(f'Epoch {epoch}, Mean classification accuracy, validation set: {(all_batch_mean_acc_valid) * 100 :.4f}%')
                    print(f'Epoch {epoch}, Mean per class accuracy, validation set: {class_dic[0]}: {all_batch_mean_per_class_acc_valid[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_valid[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_valid[2] * 100 : .4f}% | {class_dic[3]}: {all_batch_mean_per_class_acc_valid[3] * 100 : .4f}% | {class_dic[4]}: {all_batch_mean_per_class_acc_valid[4] * 100 : .4f}% |'
                        f' {class_dic[5]}: {all_batch_mean_per_class_acc_valid[5] * 100 : .4f}% | {class_dic[6]}: {all_batch_mean_per_class_acc_valid[6] * 100 : .4f}% | {class_dic[7]}: {all_batch_mean_per_class_acc_valid[7] * 100 : .4f}% | {class_dic[8]}: {all_batch_mean_per_class_acc_valid[8] * 100 : .4f}% | {class_dic[9]}: {all_batch_mean_per_class_acc_valid[9] * 100 : .4f}%')

                if valid_loss<best_val_loss:
                    best_val_loss=valid_loss
                    best_model=model.state_dict()

            train_loss_per_epoch.append(train_loss)
            valid_loss_per_epoch.append(valid_loss)

        # Testing best model
        model.load_state_dict(best_model)
        # #Evaluating best model on test set
        with torch.no_grad():
            all_batch_mean_acc_test = []
            all_batch_mean_per_class_acc_test = [0]*num_class
            for features, label in test_loader:
                mean_per_class_acc_per_batch_test = [0]*num_class
                class_correct_per_batch_test = [0] * num_class
                class_total_per_batch_test = [0] * num_class
                features = features.to(device)
                label = label.to(device)
                inv_labels_test = torch.where(label == 1)[1]
                test_output = model(features)
                _, test_pred = torch.max(test_output.data, 1)
                correct_batch_samples_test = (test_pred == inv_labels_test).sum().item()

                for c in range(num_class):
                    class_correct_per_batch_test[c] = ((test_pred == inv_labels_test) & (inv_labels_test == c)).sum().item()
                    class_total_per_batch_test[c] = (inv_labels_test == c).sum().item()
                    if class_total_per_batch_test[c] != 0:
                        mean_per_class_acc_per_batch_test[c] = class_correct_per_batch_test[c]/class_total_per_batch_test[c]

                for m in range(num_class):
                    all_batch_mean_per_class_acc_test[m] += mean_per_class_acc_per_batch_test[m]

                mean_acc_per_batch_test = correct_batch_samples_test / label.size(0)
                all_batch_mean_acc_test.append(mean_acc_per_batch_test)

            all_batch_mean_per_class_acc_test = [ele / len(test_loader) for ele in all_batch_mean_per_class_acc_test]
            all_batch_mean_acc_test = sum(all_batch_mean_acc_test) / len(test_loader)

        # print(f'Mean classification accuracy, test data: {all_batch_mean_acc_test*100 :.4f}%')
        # print(f'Mean per class accuracy, test data: {class_dic[0]}: {all_batch_mean_per_class_acc_test[0] * 100 : .4f}% | {class_dic[1]}: {all_batch_mean_per_class_acc_test[1] * 100 : .4f}% | {class_dic[2]}: {all_batch_mean_per_class_acc_test[2] * 100 : .4f}% | {class_dic[3]}: {all_batch_mean_per_class_acc_test[3] * 100 : .4f}% | {class_dic[4]}: {all_batch_mean_per_class_acc_test[4] * 100 : .4f}% |'
        #         f' {class_dic[5]}: {all_batch_mean_per_class_acc_test[5] * 100 : .4f}% | {class_dic[6]}: {all_batch_mean_per_class_acc_test[6] * 100 : .4f}% | {class_dic[7]}: {all_batch_mean_per_class_acc_test[7] * 100 : .4f}% | {class_dic[8]}: {all_batch_mean_per_class_acc_test[8] * 100 : .4f}% | {class_dic[9]}: {all_batch_mean_per_class_acc_test[9] * 100 : .4f}%')

        # plotting training and validation losses
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(num_epochs), train_loss_per_epoch, label='Training Loss')
        ax.plot(range(num_epochs), valid_loss_per_epoch, label='Validation Loss')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.title(f'Training and Validation losses for CIFAR-10')
        plt.show()

if __name__=='__main__':
    train_file=r'C:\Users\udayr\PycharmProjects\MLfiles\cs6140-hw04-dataset\hw04_data\train.csv'
    validation_file=r'C:\Users\udayr\PycharmProjects\MLfiles\cs6140-hw04-dataset\hw04_data\validation.csv'
    test_file=r'C:\Users\udayr\PycharmProjects\MLfiles\cs6140-hw04-dataset\hw04_data\test.csv'
    print('Movie Genre Classification')
    # movie_classification=Movie_genre_classification(train_file,validation_file,test_file)
    CIFAR_dataset_file=r'C:\Users\udayr\PycharmProjects\MLfiles\cifar-10-batches-py'
    print('----------------------------------------------------------------')
    print('Image Recognition')
    image_class=CIFAR10(CIFAR_dataset_file)