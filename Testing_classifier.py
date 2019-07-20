import torch
import dataset
import Network_architecture as network
import torch.nn as nn
import numpy as np
import argparse

gpu_available = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='Testing the classifier')
parser.add_argument("--model",type=str, default='traffic_model_2.pt',help='Model to test (defaul = traffic_model_2.pt)')

args = parser.parse_args()

def imshow(img):
  img= img/2 + 0.5
  plt.imshow(np.transpose(img,(1,2,0)))

model = network.Network()
model_loaded = args.model
model.load_state_dict(torch.load(model_loaded,map_location=torch.device('cpu')))

train_loader,valid_loader,test_loader = dataset.loader(batch_size=64)

############ Cost function #############
criterion = nn.CrossEntropyLoss()



test_loss = 0
class_correct = list(0.for i in range(len(dataset.classes)))
class_total = list(0.for i in range(len(dataset.classes)))

model.eval()
print("Testing the model..")
for data,target in test_loader:
  if gpu_available:
    data,target = data.cuda(),target.cuda()

  output = model(data)
  loss = criterion(output,target)
  test_loss += loss.item()*data.size(0)
  _,pred = torch.max(output,1)
  correct_tensor = pred.eq(target.data.view_as(pred))
  correct = np.squeeze(correct_tensor.numpy()) if not gpu_available else np.squeeze(correct_tensor.cpu().numpy())

  for i in range(64):
    try:
      label = target.data[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1
    except:
      pass
test_loss = test_loss/len(test_loader.dataset)
print("Test Loss = {:.6f}".format(test_loss))

for i in range(len(dataset.classes)):
  if class_total[i] > 0:
    print("Test accuracy of %5s : %2d%% (%2d/%2d)" % (dataset.classes[i], 100*class_correct[i]/class_total[i],
                                                     np.sum(class_correct[i]),np.sum(class_total[i])))
  else:
    print("Test accuracy of %5s: N/A(no training example)"% (dataset.classes[i]))

print('\n Test accuracy overall: %2d%% (%2d/%2d)'% (100*np.sum(class_correct)/np.sum(class_total),np.sum(class_correct),np.sum(class_total)))
