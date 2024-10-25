import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader
import timm
from timm import optim, scheduler
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms as tfm
from sklearn import metrics as skmet
from jupyterplot import ProgressPlot
import matplotlib.pyplot as plt
import copy
from PIL import Image
import torch.utils.data as data_utils
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score
import torchvision.models as models
from dataset import ImageData
import torch
import os
import pickle
import load_splits
from timm import optim, scheduler
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset
import sklearn.metrics as skmet
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import math


parser = argparse.ArgumentParser(description="Train and evaluate the model")
parser.add_argument("--save_dir", type=str, default="/home/xiaofey/xray/xray-master/code/saved_models_MIL4",
                    help="Directory to save model checkpoints")
parser.add_argument("--plot_dir", type=str, default="/home/xiaofey/xray/xray-master/code/saved_models_MIL4",
                    help="Directory to save plot images")
parser.add_argument("--pretrained_model_dir", type=str, default="/home/xiaofey/xray/xray-master/code/test6",
                    help="Directory containing pretrained models")
parser.add_argument("--attention_branches", type=int, required = True,
                        help="Number of attention branches")
args = parser.parse_args()

# Create directories if they don't exist
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

save_dir = args.save_dir
pretrained_model_dir = args.pretrained_model_dir
attention_branches = args.attention_branches


kfold = 10  # number of folds in repeated k-fold
nrepeats = 1  # number of repeats in repeated k-fold
bs_train = 15  # batch size for training
bs_test = 25  # batch size for testing
num_workers = 0  # number of parallel data loading workers
res = 224 # pixel size along height and width
device = torch.device('cuda:0')
num_classes = 3
model = 'resnet50d'
lr = 0.001
lr_gamma = 0.92
dropout = 0.3
weight_decay = 0.001
pretrained=True
keep_views = ['AP_LIKE', 'Y']
unfreeze_after_n=8


df_img = pd.read_csv('../data/model_images.csv')
df_trg = pd.read_csv('../data/model_targets.csv')

tfms_train = tfm.Compose([
    tfm.Resize(res),
    tfm.CenterCrop(res),
    tfm.RandomEqualize(p=0.5),
    tfm.RandAugment(),
    tfm.ToTensor(),
    tfm.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250])),
    tfm.RandomHorizontalFlip(),
    tfm.RandomRotation((-45, 45)),
    tfm.RandomInvert(),
    tfm.Grayscale()
])

tfms_test = tfm.Compose([
    tfm.Resize(res),
    tfm.CenterCrop(res),
    tfm.ToTensor(),
    tfm.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250])),
    tfm.Grayscale()
])

class ImageDataBags(Dataset):
    """Image CSV Dataset for Multiple Instance Learning"""
    type_map = {1: 0, 2: 1, 3: 2, 4: 2}
    inv_type_map = {0: '1-part', 1: '2-part', 2: '3- or 4-part'}

    def __init__(self, data: pd.DataFrame, transforms, seed=1, train=True, view_filter = None):
        self.transforms = transforms
        self.data = data
        self.train = train
        self.r = np.random.RandomState(seed=1)
        # Recode targets
        self.data['trg'] = self.data['num_parts'].map(ImageDataBags.type_map)

        # Fill nan views with "Unknown"
        self.data['view_group'] = self.data['view_group'].fillna('Unknown')
        
        # Filter to given views
        if view_filter is not None:
            self.data = self.data.loc[self.data['view_group'].isin(view_filter)]

        self.bags_list, self.labels_list, self.patient_ids_list = self._create_bags()

    def _create_bags(self):
        bags_list = []
        labels_list = []
        patient_ids_list = []  
        patient_ids = self.data['patient_id'].unique()  

        # Determine the maximum number of images in any bag based on the patient with the most images
        max_images = max(self.data.groupby('patient_id').size())
        #print(max_images)
        for patient_id in patient_ids:
            patient_data = self.data[self.data['patient_id'] == patient_id]
            # print(patient_data)
            bag_images = []
            bag_labels = []
            #print("$" * 40)
            # Fill the bag with images from the patient, applying transformations
            for idx, row in patient_data.iterrows():
                img = self.transforms(Image.open(row['path']))
                trg = row['trg']
                # print(img.shape)
                bag_images.append(img)
                bag_labels.append(trg)
                #bag_labels = [1 if label == 1 else 0 for label in bag_labels]
                C, H, W = bag_images[0].shape
            bags_list.append(torch.stack(bag_images))
            bag_label = bag_labels[0]
            labels_list.append((torch.tensor(bag_label, dtype=torch.long), torch.tensor(bag_labels, dtype=torch.long)))
            patient_ids_list.append(patient_id)
        return bags_list, labels_list, patient_ids_list
    def __len__(self):
        #print("len")
        return len(self.bags_list)
        #return len(self.labels_list)

    def __getitem__(self, index):
        bag, (bag_label, instance_labels) = self.bags_list[index], self.labels_list[index]
        patient_id = self.patient_ids_list[index]
        #print(self.labels_list[index])

        return bag, (bag_label, instance_labels, patient_id)
    

def collate_fn(batch):
    # print("collate_fn called")
    # Initialize empty lists for combined bags and labels
    combined_bag_images = []
    combined_instance_labels = []
    combined_bag_labels = []  # Collect all bag labels
    combined_patient_ids = []
    markers = [0]  # Start with 0, the index for the start of the first bag
    
    total_images = 0
    for bag, (bag_label, instance_labels, patient_id) in batch:
        num_images = bag.shape[0]  # Number of images in the current bag
        total_images += num_images
        combined_bag_images.append(bag)
        combined_instance_labels.append(instance_labels)
        combined_bag_labels.append(bag_label)  # Collect bag labels for each study bag
        combined_patient_ids.append(patient_id)
        markers.append(total_images)  # Append cumulative image count

    # Combine all study bags into one along the 0th dimension (number of images)
    combined_bag_images = torch.cat(combined_bag_images, dim=0)  # Combined shape: [n1 + n2 + ..., channel, h, w]
    combined_instance_labels = torch.cat(combined_instance_labels, dim=0)  # Combined instance labels

    return combined_bag_images, (combined_bag_labels, combined_instance_labels, combined_patient_ids, markers)



class ModifiedModel(nn.Module):
    def __init__(self, attention_branches, pretrained_model_path = None, num_classes=3, in_chans=1, dropout=0.3):
        super(ModifiedModel, self).__init__()
        self.M = 2048
        self.L = 128
        self.ATTENTION_BRANCHES = attention_branches
        self.subspace_size = self.M // self.ATTENTION_BRANCHES
        self._scale = math.sqrt(self.subspace_size)
        model = timm.create_model('resnet50d', pretrained=False, num_classes=num_classes, in_chans=in_chans, drop_rate=dropout)
        model.load_state_dict(torch.load(pretrained_model_path))
        
        self.q = nn.Parameter(torch.randn(self.ATTENTION_BRANCHES, self.subspace_size))
        
        # Extract the feature extraction layers and freeze their parameters
        self.features = nn.Sequential(*list(model.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False

        self.fc = list(model.children())[-1]
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, x, markers):
        # Extract features from all images
        S = self.features(x).squeeze()  # Shape: [total_images, 2048]
        
        h_prime_list = []
        attention_weights_list = []

        # Loop through each bag based on markers
        for i in range(len(markers) - 1):
            # Extract features for the current bag
            h = S[markers[i]:markers[i + 1]]  # Shape: [n_i, 2048] (n_i: number of images in the current bag)
            h_vid, attn = self.attention_pool(h)

            h_prime_list.append(h_vid)
#             # Compute attention weights (dot product with query vector)
#             lamb = torch.matmul(h, self.q) # [n_i, branches]
#             att = torch.softmax(lamb, dim=0)  # [n_i, branches]

#             # Apply attention: weighted sum of features for the current bag
#             h_prime = torch.matmul(att.T, h)   # [branches, 2048]
#             h_prime = h_prime.squeeze(0)  # warning: might not generalize when branches > 1

            # Store the result for this bag
            attention_weights_list.append(attn)
        # Stack all h_prime tensors to form h_tol with shape [num_bags, 2048]
        h_tol = torch.stack(h_prime_list, dim=0)  # Shape: [num_bags, 2048]
        
        # Pass h_tol through the classifier to get the logits (predictions)
        logits = self.fc(h_tol).squeeze(1) # Shape: [num_bags, num_classes]
        Y_hat = torch.argmax(logits, dim=1)
        return logits, attention_weights_list, Y_hat
    
    def attention_pool(self, h):
        # Reshape input for multi-head attention
        h_query = h.view(-1, self.ATTENTION_BRANCHES, self.subspace_size)
        
        # Compute attention logits
        alpha = (h_query * self.q).sum(axis=-1) / self._scale
        
        # Normalize attention weights
        attn = torch.softmax(alpha, dim=0)

        # Apply attention and pool
        h_vid = torch.sum(h_query * attn.unsqueeze(-1), dim=0)
        
        # Reshape to combine all heads
        h_vid = h_vid.view(-1, self.M)
        
        return h_vid, attn
    
    def calculate_classification_error(self, X, Y, markers):
        _, _, Y_hat = self.forward(X, markers)
        error = 1.0 - Y_hat.eq(Y).float().mean().item()

        return error
    
def check_frozen_parameters(model):
    frozen_layers = []
    trainable_layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
        else:
            frozen_layers.append(name)
    
    print("Frozen Layers:")
    for layer in frozen_layers:
        print(f"{layer}: Frozen (requires_grad=False)")
    
    print("\nTrainable Layers:")
    for layer in trainable_layers:
        print(f"{layer}: Trainable (requires_grad=True)")


def train_one_epoch(model, train_dataloader, device):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, (bag_label, instance_labels, _, markers)) in enumerate(train_dataloader):
        data = data.to(device)
        bag_label = torch.stack(bag_label).to(device)
        markers = markers
        optimizer.zero_grad()
        
        logits, attention_weights, _ = model(data, markers)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, bag_label)
        error = model.calculate_classification_error(data, bag_label, markers)
        train_loss += loss.item()
        train_error += error
        loss.backward()
        optimizer.step()

    # calculate loss and error for epoch
    num_samples = len(train_dataloader)
    train_loss /= num_samples
    train_error /= num_samples
    print('Loss: {:.4f}, Train error: {:.4f}'.format(train_loss, train_error))
    return train_loss, train_error
    
def evaluate(model, test_dataloader, device):
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_labels = []
    all_predictions = []
    batch_info_list = []  # List to hold information for each batch

    with torch.no_grad():
        for batch_idx, (data, (bag_label, instance_labels, patient_id, markers)) in enumerate(test_dataloader):
            data, bag_label, instance_labels, patient_id = data.to(device), torch.stack(bag_label).to(device), instance_labels.to(device), patient_id
            markers = markers
            
            logits, attention_weights, predicted_label = model(data, markers)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, bag_label)
            error = model.calculate_classification_error(data, bag_label, markers)
            test_loss += loss.item()
            test_error += error
            attention_weights_numpy = []
            for attention in attention_weights:
                attention_weights_numpy.append(attention.cpu().numpy())  # Convert each bag's attention weights to NumPy
                
            # Store all labels and predictions
            all_labels.extend(bag_label.cpu().numpy())
            all_predictions.extend(logits.cpu().numpy()) 
            # Collecting information for the current batch
            batch_info = {
                'patient_id': patient_id,
                'attention_weights': attention_weights_numpy,
                'images': data.cpu().data.numpy(),
                'labels': bag_label.cpu().data.numpy(),
                'instance_labels': instance_labels.cpu().data.numpy(),
                'predicted_labels': predicted_label.cpu().data.numpy()
            }
            batch_info_list.append(batch_info)  # Appending current batch's information to the list
    num_samples = len(test_dataloader)
    test_error /= num_samples
    test_loss /= num_samples

    all_labels = np.array(all_labels).reshape(-1, 1)  # Ensure all_labels is a 2D array

    logits = torch.tensor(np.array(all_predictions))
    all_predictions = torch.softmax(logits, dim=1).cpu().numpy()
    predicted_labels = np.argmax(all_predictions, axis=1)

    y_true_onehot = label_binarize(all_labels, classes=[0, 1, 2]) 

    roc_auc = skmet.roc_auc_score(y_true_onehot, all_predictions, multi_class='ovr')

    average_precision = skmet.average_precision_score(y_true_onehot, all_predictions)
    accuracy = skmet.accuracy_score(all_labels, predicted_labels)
    balanced_accuracy = skmet.balanced_accuracy_score(all_labels, predicted_labels)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}, AUC: {:.4f}, Average Precision: {:.4f}, Accuracy: {:.4f}, Balanced Accuracy: {:.4f}'.format(
        test_loss, test_error, roc_auc, average_precision, accuracy, balanced_accuracy))
    
    return test_loss, test_error, roc_auc, average_precision, accuracy, balanced_accuracy, batch_info_list

num_epochs=10
with open('cv_splits.pkl', 'rb') as f:
    saved_splits = pickle.load(f)
    
split_results = []
# splits = cv.split(df_trg, df_trg['fracture_type'])
unfreeze_after_n=8


for ix, (train_ix, test_ix) in enumerate(saved_splits):
    print('-'*40)
    print(f"Split {ix+1} of {kfold*nrepeats}.")
    
    pretrained_model_path = os.path.join(pretrained_model_dir, f'loss_and_metrics_split_test6_{ix+1}.pth')
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model for Split {ix+1} from {pretrained_model_path}")
        m = ModifiedModel(attention_branches = attention_branches, pretrained_model_path=pretrained_model_path)
    else:
        print(f"No pretrained model found for Split {ix+1}, starting from scratch.")
        # m = ModifiedModel(num_classes=3, in_chans=1, dropout=0.3)  # Define your model from scratch

    m.to(device)
    
    df_train = df_trg.iloc[train_ix].merge(df_img)
    df_test = df_trg.iloc[test_ix].merge(df_img)
    
    # create datasets
    d_train = ImageDataBags(df_train, transforms = tfms_train, view_filter=keep_views)
    dl_train = DataLoader(d_train, batch_size=bs_train, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    
    d_test = ImageDataBags(df_test, transforms = tfms_test, view_filter=keep_views)
    dl_test = DataLoader(d_test, batch_size=bs_test, num_workers=num_workers, collate_fn=collate_fn)

    is_frozen=True
    # fit
    optimizer = optim.Adam(m.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
    loss_function = torch.functional.F.cross_entropy
    
    train_loss_ls = []
    train_error_ls = []
    test_loss_ls = []
    test_error_ls = []
    roc_auc_ls = []
    avg_precision_ls = []
    accuracy_ls = []
    balanced_accuracy_ls = []
    
    best_test_loss = 1e10
    state_dict = None
    for epoch in range(num_epochs):
        #print("-"*40)
        print(f"\rEpoch {epoch+1} of {num_epochs}:", end='')

        train_loss, train_error = train_one_epoch(m, dl_train, device)
        train_loss_ls.append(train_loss)

        test_loss, test_error, roc_auc, avg_precision, accuracy, balanced_accuracy, batch_info_list = evaluate(m, dl_test, device)
        test_loss_ls.append(test_loss)
        # test_error_ls.append(test_error)
        roc_auc_ls.append(roc_auc)
        avg_precision_ls.append(avg_precision)
        accuracy_ls.append(accuracy)
        balanced_accuracy_ls.append(balanced_accuracy)
        
        
        # save weights if improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            state_dict = copy.deepcopy(m.state_dict())
        
        scheduler.step()
        
    model_save_path = os.path.join(save_dir, f'fine_tuned_model_split_test_{ix+1}.pth')
    torch.save(state_dict, model_save_path)
    
    split_results.append({
        'split_ix': ix, 
        'best_test_loss': best_test_loss, 
        'state_dict': state_dict,
        'train_ix': train_ix,
        'test_ix': test_ix,
        'train_loss': train_loss_ls, 
        'test_loss': test_loss_ls,
        'roc_auc': roc_auc_ls,
        'average_precision': avg_precision_ls,
        'accuracy': accuracy_ls,
        'balanced_accuracy': balanced_accuracy_ls
    })



# for ix, res in enumerate(split_results):
#     # Create a figure with one subplot (ncols=2)
#     fig, ax = plt.subplots(ncols=2, sharex=True)
#     fig.set_size_inches(20, 10)

#     # x-axis values (epoch numbers)
#     x = np.arange(1, num_epochs + 1)

#     # Plot train and test loss for this split
#     ax[0].plot(x, res['train_loss'], color='k', label='train cross entropy')
#     ax[0].plot(x, res['test_loss'], color='b', label='test cross entropy')
#     ax[0].set_xlabel('Epochs')
#     ax[0].set_ylabel('Loss')
#     ax[0].legend()
#     ax[0].set_title(f'Cross-Entropy Loss - Split {ix+1}')

#     # Plot metrics for this split
#     ax[1].plot(x, res['roc_auc'], color='k', label='roc_auc')
#     ax[1].plot(x, res['average_precision'], color='b', label='average_precision')
#     ax[1].plot(x, res['accuracy'], color='g', label='accuracy')
#     ax[1].plot(x, res['balanced_accuracy'], color='r', label='balanced_accuracy')

#     ax[1].set_xlabel('Epochs')
#     ax[1].set_ylabel('Metric Value')
#     ax[1].legend()
#     ax[1].set_title(f'Evaluation Metrics - Split {ix+1}')

#     # Save the figure for each split
#     plt.savefig(os.path.join(args.plot_dir, f'second_model_loss_and_metrics_split_test_{ix+1}.png'), dpi=300, bbox_inches='tight')
    
#     # Show the plots for the current split (optional, if you want to display them during each iteration)
#     plt.show()
    
#     # Clear the figure after saving to avoid overlap between plots
#     plt.clf()
    
    
    
# # Example data
# train_losses = np.array([res['train_loss'] for res in split_results])
# test_losses = np.array([res['test_loss'] for res in split_results])
# roc_aucs = np.array([res['roc_auc'] for res in split_results])
# avg_precisions = np.array([res['average_precision'] for res in split_results])
# accuracies = np.array([res['accuracy'] for res in split_results])
# balanced_accuracies = np.array([res['balanced_accuracy'] for res in split_results])

# # Set font size
# plt.rcParams['font.size'] = 22

# # Create a figure with one subplot (ncols=1)
# fig, ax = plt.subplots(ncols=2, sharex=True)
# fig.set_size_inches(20, 10)

# # x-axis values (epoch numbers)
# x = np.arange(1, num_epochs + 1)

# # Plot each split result's train and test loss
# for s in split_results:
#     ax[0].plot(x, s['train_loss'], color='k', linestyle='--', alpha=0.4)
#     ax[0].plot(x, s['test_loss'], color='b', linestyle='--', alpha=0.4)

# # Plot the average train and test loss on the first subplot
# ax[0].plot(x, train_losses.mean(axis=0), color='k', label='train cross entropy')
# ax[0].plot(x, test_losses.mean(axis=0), color='b', label='test cross entropy')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Loss')
# ax[0].legend()
# ax[0].set_title('Cross-Entropy Loss')

# # Plot the four metrics on the second subplot
# for s in split_results:
#     ax[1].plot(x, s['roc_auc'], color='k', linestyle='--', alpha=0.4)
#     ax[1].plot(x, s['average_precision'], color='b', linestyle='--', alpha=0.4)
#     ax[1].plot(x, s['accuracy'], color='g', linestyle='--', alpha=0.4)
#     ax[1].plot(x, s['balanced_accuracy'], color='r', linestyle='--', alpha=0.4)

# # Plot the average of the four metrics on the second subplot
# ax[1].plot(x, roc_aucs.mean(axis=0), color='k', label='roc_auc')
# ax[1].plot(x, avg_precisions.mean(axis=0), color='b', label='average_precision')
# ax[1].plot(x, accuracies.mean(axis=0), color='g', label='accuracy')
# ax[1].plot(x, balanced_accuracies.mean(axis=0), color='r', label='balanced_accuracy')

# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Metric Value')
# ax[1].legend()
# ax[1].set_title('Evaluation Metrics')


# plt.savefig(os.path.join(args.plot_dir, 'second_model_loss_and_metrics_test_final.png'), dpi=300, bbox_inches='tight')
# # Show the plots
# plt.show()



from scipy import stats

all_roc_auc = [split['roc_auc'][-1] for split in split_results]
roc_auc_array = np.array(all_roc_auc)
overall_mean = np.mean(roc_auc_array)

# Calculate 90% confidence interval
confidence_level = 0.90
degrees_of_freedom = len(roc_auc_array) - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
margin_of_error = t_value * (np.std(roc_auc_array) / np.sqrt(len(roc_auc_array)))

print(f"Overall ROC AUC = {overall_mean:.4f} ± {margin_of_error:.4f} (90% CI)")

with open('all_roc_auc_results.txt', 'a') as f:  # 'a' for append mode
    f.write(f"Branches_{args.attention_branches}")  # Write branch number
    for value in all_roc_auc:
        f.write(f",{value}")  # Use comma as separator
    f.write("\n")


# results_filename = f'results_attention_branches_{attention_branches}.pkl'
# with open(os.path.join(save_dir, results_filename), 'wb') as f:
#     pickle.dump(split_results, f)

# print(f"Results saved for attention branches: {attention_branches}")

# # Collect the final ROC AUC score for each fold
# final_roc_auc_scores = [split['roc_auc'][-1] for split in split_results]
# # Calculate overall statistics
# overall_mean = np.mean(final_roc_auc_scores)
# overall_std = np.std(final_roc_auc_scores)
# # Create the boxplot
# plt.figure(figsize=(8, 6))
# sns.boxplot(y=final_roc_auc_scores)

# # Customize the plot
# plt.title('ROC AUC Scores Across 10-Fold Cross-Validation', fontsize=16)
# plt.ylabel('ROC AUC Score', fontsize=14)
# plt.ylim(0, 1)  # Set y-axis limits from 0 to 1 for AUC scores

# # Remove x-axis label
# plt.xlabel('')

# # Add the mean ± std text
# plt.text(0.5, -0.15, f"Mean ROC AUC = {overall_mean:.4f} ± {overall_std:.4f}", 
#          horizontalalignment='center', verticalalignment='center', 
#          transform=plt.gca().transAxes, fontsize=12)

# # Save the figure
# plt.savefig(os.path.join(args.plot_dir, 'roc_auc_overall_boxplot.png'), dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()

# # Print the scores for each fold
# for i, score in enumerate(final_roc_auc_scores, 1):
#     print(f"Fold {i}: ROC AUC = {score:.4f}")
    
# with open('MIL4_roc_auc_scores.pkl', 'wb') as f:  # Change MIL1 to MIL2, MIL3, etc. for each file
#     pickle.dump(final_roc_auc_scores, f)

    
    
    
# # Calculate for the Table
# def process_metric(metric_name):
#     all_scores = [split[metric_name][-1] for split in split_results]
#     scores_array = np.array(all_scores)
#     overall_mean = np.mean(scores_array)
#     overall_std = np.std(scores_array)
#     print(f"Overall {metric_name} = {overall_mean:.4f} ± {overall_std:.4f}")
#     return all_scores, overall_mean, overall_std

# # Process each metric
# metrics = ['roc_auc', 'accuracy', 'average_precision', 'balanced_accuracy']
# results = {}

# for metric in metrics:
#     scores, mean, std = process_metric(metric)
#     results[metric] = {
#         'scores': scores,
#         'mean': mean,
#         'std': std
#     }
    
#     # Print the scores for each fold
#     for i, score in enumerate(scores, 1):
#         print(f"Fold {i}: {metric} = {score:.4f}")

# with open('MIL4_metric_results.txt', 'w') as f:
#     for metric in metrics:
#         scores, mean, std = process_metric(metric)
#         results[metric] = {
#             'scores': scores,
#             'mean': mean,
#             'std': std
#         }
        
#         # Write overall result to file
#         f.write(f"Overall {metric} = {mean:.4f} ± {std:.4f}\n")
        
#         # Write individual fold scores to file
#         f.write("Individual fold scores:\n")
#         for i, score in enumerate(scores, 1):
#             f.write(f"Fold {i}: {score:.4f}\n")
        
#         f.write("\n")  # Add a blank line between metrics

# print("All metrics processed and saved to MIL4_metric_results.txt")