import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from i3d import InceptionI3d  
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from dataloader_p import *
from sklearn.model_selection import train_test_split


class IntraModalityAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.3):
        super(IntraModalityAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class InterModalityAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.3):
        super(InterModalityAttention, self).__init__()

        self.imu_to_rgbpose_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.rgb_to_imupose_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.pose_to_imu_rgb_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)

        self.norm_imu = nn.LayerNorm(embed_size)
        self.norm_rgb = nn.LayerNorm(embed_size)
        self.norm_pose = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, imu_features, rgb_features, pose_features):
        attn_output_imu, _ = self.imu_to_rgbpose_attention(imu_features, rgb_features + pose_features, rgb_features + pose_features)
        imu_output = self.norm_imu(imu_features + self.dropout(attn_output_imu))

        attn_output_rgb, _ = self.rgb_to_imupose_attention(rgb_features, imu_features + pose_features, imu_features + pose_features)
        rgb_output = self.norm_rgb(rgb_features + self.dropout(attn_output_rgb))

        attn_output_pose, _ = self.pose_to_imu_rgb_attention(pose_features, imu_features + rgb_features, imu_features + rgb_features)
        pose_output = self.norm_pose(pose_features + self.dropout(attn_output_pose))

        return imu_output, rgb_output, pose_output

class gated_fus(nn.Module):
    def __init__(self, feature_size):
        super(gated_fus, self).__init__()
        
        self.gate_imu_rgb = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.Sigmoid()
        )
        
        self.gate_fused_pose = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.Sigmoid()
        )

    def forward(self, imu_features, rgb_features, pose_features):
        imu_rgb_combined = torch.cat([imu_features, rgb_features], dim=-1)
        gate_imu_rgb_weights = self.gate_imu_rgb(imu_rgb_combined)
        fused_imu_rgb = gate_imu_rgb_weights * imu_features + (1 - gate_imu_rgb_weights) * rgb_features
        
        fused_imu_rgb_pose_combined = torch.cat([fused_imu_rgb, pose_features], dim=-1)
        gate_fused_pose_weights = self.gate_fused_pose(fused_imu_rgb_pose_combined)
        fused_all = gate_fused_pose_weights * fused_imu_rgb + (1 - gate_fused_pose_weights) * pose_features
        
        return fused_all


class eng_network(nn.Module):
    def __init__(self, imu_feature_dim, rgb_feature_dim, pose_feature_dim, num_classes, embed_size, num_heads, freeze_encoders=True):
        super(eng_network, self).__init__()

        self.imu_resnet_left = models.resnet18(pretrained=True)
        self.imu_resnet_right = models.resnet18(pretrained=True)
        self.imu_resnet_left.fc = nn.Linear(self.imu_resnet_left.fc.in_features, imu_feature_dim)
        self.imu_resnet_right.fc = nn.Linear(self.imu_resnet_right.fc.in_features, imu_feature_dim)

        self.rgb_i3d = InceptionI3d(400, in_channels=3)
        self.rgb_i3d.replace_logits(rgb_feature_dim)

        self.pose_linear = nn.Linear(pose_feature_dim, embed_size)

        if freeze_encoders:
            self.freeze_encoder_layers()

        self.imu_intra_attention = IntraModalityAttention(embed_size, num_heads)
        self.rgb_intra_attention = IntraModalityAttention(embed_size, num_heads)
        self.pose_intra_attention = IntraModalityAttention(embed_size, num_heads)

        self.inter_modality_attention = InterModalityAttention(embed_size, num_heads)

        self.gated_fusion = gated_fus(embed_size)

        self.classifier = nn.Linear(embed_size, num_classes)

    def freeze_encoder_layers(self):
        for param in self.imu_resnet_left.parameters():
            param.requires_grad = False
        for param in self.imu_resnet_right.parameters():
            param.requires_grad = False
        for param in self.rgb_i3d.parameters():
            param.requires_grad = False

    def forward(self, imu_left_images, imu_right_images, rgb_videos, pose_keypoints):
        imu_left_features = self.imu_resnet_left(imu_left_images)
        imu_right_features = self.imu_resnet_right(imu_right_images)
        imu_features = torch.cat([imu_left_features, imu_right_features], dim=1)

        rgb_features = self.rgb_i3d(rgb_videos).squeeze(-1)

        batch_size = pose_keypoints.shape[0]  
        pose_keypoints_flat = pose_keypoints.float().view(batch_size, -1)  
        pose_features = self.pose_linear(pose_keypoints_flat)  

        imu_features = self.imu_intra_attention(imu_features.unsqueeze(0))
        rgb_features = self.rgb_intra_attention(rgb_features.unsqueeze(0))
        pose_features = self.pose_intra_attention(pose_features.unsqueeze(0))

        imu_features, rgb_features, pose_features = self.inter_modality_attention(imu_features, rgb_features, pose_features)

        fused_features = self.gated_fusion(imu_features.squeeze(0), rgb_features.squeeze(0), pose_features.squeeze(0))

        logits = self.classifier(fused_features)

        return F.log_softmax(logits, dim=1)



def save_epoch_stats(epoch, train_loss, train_accuracy, test_loss, test_accuracy, stats_file_path):
    with open(stats_file_path, "a") as f:
        f.write(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

def train_supervised(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=10, save_dir="./model_saves", stats_file="train_test_stats.txt"):
    model.train() 

    os.makedirs(save_dir, exist_ok=True)
    stats_file_path = os.path.join(save_dir, stats_file)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for rgb_images, imu_left_images, imu_right_images, pose_keypoints, labels in train_dataloader:
            # print(rgb_images.shape, imu_left_images.shape, imu_right_images.shape, pose_keypoints.shape, labels)
            rgb_images, imu_left_images, imu_right_images, pose_keypoints, labels = (
                rgb_images.to(device),
                imu_left_images.to(device),
                imu_right_images.to(device),
                pose_keypoints.to(device),
                labels.to(device),
            )

            optimizer.zero_grad() 

            outputs = model(imu_left_images, imu_right_images, rgb_images, pose_keypoints)

            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()  

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct / total * 100

        test_loss, test_accuracy = evaluate_supervised(model, test_dataloader)

        save_epoch_stats(epoch, train_loss, train_accuracy, test_loss, test_accuracy, stats_file_path)

        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    print('Training completed.')

def evaluate_supervised(model, dataloader):
    model.eval()  
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():  
        for rgb_images, imu_left_images, imu_right_images, pose_keypoints, labels in dataloader:
            rgb_images, imu_left_images, imu_right_images, pose_keypoints, labels = (
                rgb_images.to(device),
                imu_left_images.to(device),
                imu_right_images.to(device),
                pose_keypoints.to(device),
                labels.to(device),
            )


            outputs = model(imu_left_images, imu_right_images, rgb_images, pose_keypoints)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    return test_loss, accuracy



def print_dataset_statistics(dataset, dataset_name="Dataset"):
    engaged_count = 0
    disengaged_count = 0
    total_samples = len(dataset)
    
    for _, _, _,_, label in dataset:
        if label == 0:
            engaged_count += 1
        elif label == 1:
            disengaged_count += 1

    print(f"---- {dataset_name} Statistics ----")
    print(f"Total samples: {total_samples}")
    print(f"Engaged: {engaged_count} ({(engaged_count/total_samples)*100:.2f}%)")
    print(f"Disengaged: {disengaged_count} ({(disengaged_count/total_samples)*100:.2f}%)")
    print("----------------------------\n")




 

root_path = '/workspace/cstudent4/RGBD_IMU_Dataset/exp/engagment_data/'
save_dir  = '/workspace/cstudent4/RGBD_IMU_Dataset/exp/exp_logs/exp_rgbd_imu_pose_full'
save_filer  = '/workspace/cstudent4/RGBD_IMU_Dataset/exp/exp_logs/exp_rgbd_imu_pose_full/train_test_stats.txt'
all_folders = [folder for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]

train_folders, test_folders = train_test_split(all_folders, test_size=0.3, random_state=42)

print(f"Training folders: {train_folders}")
print(f"Testing folders: {test_folders}")

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


imu_transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5) 
])



rgb_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


imu_test_transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])


pose_transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0], std=[1])  
])


train_dataset = R_IMU_data(root_path, train_folders, imu_transform=imu_transform, rgb_transform=rgb_transform, pose_transform=pose_transform)
test_dataset = R_IMU_data(root_path, test_folders, imu_transform=imu_test_transform, rgb_transform=rgb_test_transform, pose_transform=pose_transform)


print_dataset_statistics(train_dataset, dataset_name="Training Set")
print_dataset_statistics(test_dataset, dataset_name="Testing Set")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

model = eng_network(imu_feature_dim=256, rgb_feature_dim=512, pose_feature_dim=33*3, num_classes=2, embed_size=512, num_heads=8,freeze_encoders=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

train_supervised(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=50, save_dir=save_dir, stats_file=save_filer)
 