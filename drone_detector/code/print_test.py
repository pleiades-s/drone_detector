# Test function
# 1. Calculate average accuracy
# 2. Calculate accuracy for each class
# 3. Print confusion matrix

import datetime
import torch
import numpy as np

def test_acc(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:

            # data.transpose_(1,2)
            data = torch.as_tensor(data, dtype=torch.double, device='cuda')
            labels = torch.as_tensor(labels, dtype=torch.long, device='cuda')
            model = model.double()
            outputs = model(data)
            pred = outputs.max(1)[1]

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total
    print('Accuracy of the network on the testset: {:.6f}%\n'.format(acc))

    return acc
    # f.write('Accuracy of the network on the testset: {}%\n'.format(100 * correct / total))

def test_acc_classes(model, test_loader, length, batch_size, f):
    outcome = []
    count = 0
    classes = ['section 1 forward', 'section 1 hovering', 
           'section 2 forward', 'section 2 hovering',
           'section 3 forward', 'section 3 hovering']

    model = model.double()

    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))

    for i in range(6):
        arr = []
        for j in range(6):
            arr.append(0)
        
        outcome.append(arr)
    
    with torch.no_grad(): #It means it's not gonna train for this model with dataset
        
        for data, labels in test_loader:
            
            count += 1
            if count == int(length / batch_size):
                break

            # data.transpose_(1, 2)

            data = torch.as_tensor(data, dtype=torch.double, device='cuda')
            labels = torch.as_tensor(labels, dtype=torch.long, device='cuda')
            
            outputs = model(data)
            pred = outputs.max(1)[1]
            
            for i in range(batch_size):
                # print("outcome {}, labels {} pred {}".format(np.shape(outcome), labels[i], pred[i]))
                outcome[labels[i]][pred[i]] += 1

            c = (pred == labels).squeeze()
            
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(6):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        f.write('Accuracy of %5s : %2d %%\n' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
    return outcome

def print_confusion_matrix(outcome, f):
    class_name = ['1-f','1-h', '2-f', '2-h', '3-f', '3-h']

    row_format ="{:>15}" * (len(outcome) + 1)
    print(row_format.format("", *class_name))
    f.write(row_format.format("", *class_name))
    for team, row in zip(class_name, outcome):
                print(row_format.format(team, *row))
                f.write(row_format.format(team, *row))


def inference_print(model, test_loader, f):
    classes = ['section 1 forward', 'section 1 hovering', 
           'section 2 forward', 'section 2 hovering',
           'section 3 forward', 'section 3 hovering']
    prediction = []

    for data, name in test_loader:

        # data.transpose_(1, 2)
        data = torch.as_tensor(data, dtype=torch.float, device='cuda')
        data = data.to('cuda')

        output = model(data)
        pred = output.max(1)[1]
        
        for i in range(len(pred)):
            
            dummy = []
            dummy.append(name[i].split('_')[0])
            dummy.append(pred[i].item())
            prediction.append(dummy)

            print("{}\t{}".format(name[i], classes[pred[i].item()]))
            f.write("{}\t{}\n".format(name[i], classes[pred[i].item()]))
    
    correct_pred = ['3-1', '3-2', '4-1', '4-2', '6-4']
    name = correct_pred[0]
    f.write(name+'\n')

    for i in range(len(prediction)):
        if prediction[i][0] in correct_pred:
            if(name != prediction[i][0]):
                name = prediction[i][0]
                f.write('\n'+name +'\n')

            f.write(str(prediction[i][1]) + ',')