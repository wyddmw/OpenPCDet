import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, in_channel=64):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, N)
        return torch.cat([x, pointfeat], 1)
        

class PointNetBackbone(nn.Module):
    def __init__(self, in_channel=64):
        super(PointNetBackbone, self).__init__()
        self.feat = PointNetEncoder(in_channel=in_channel)
        self.conv1 = torch.nn.Conv1d(1152, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class VotingModule(nn.Module):
    def __init__(self, in_channel=64, feature_channel=128, num_voting=1):
        super(VotingModule, self).__init__()
        self.pointnet = PointNetBackbone(in_channel)
        self.conv1 = nn.Conv1d(feature_channel, feature_channel, 1, bias=False)
        self.conv2 = nn.Conv1d(feature_channel, feature_channel, 1, bias=False)   
        self.offset = nn.Conv1d(feature_channel, 2, 1, bias=False)
        self.stride = nn.Conv1d(feature_channel, 1, 1, bias=False)
        self.prob = nn.Conv1d(feature_channel, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(feature_channel)
        self.bn2 = nn.BatchNorm1d(feature_channel)

    def forward(self, input_feature):
        voting_feature = self.pointnet(input_feature)
        voting_feature = F.relu(self.bn1(self.conv1(voting_feature)))
        voting_feature = F.relu(self.bn2(self.conv2(voting_feature)))
        centering_offset = self.offset(voting_feature)
        stride = F.relu(self.stride(voting_feature))
        prob = self.sigmoid(self.prob(voting_feature))
        return centering_offset, stride, prob
	

if __name__ == '__main__':
    model = VotingModule()
    xyz = torch.rand(12, 64, 6000)
    data_dict = {'pillar_feature': xyz}
    output = model(data_dict)
