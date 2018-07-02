# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script file.
"""
# Import
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
np.set_printoptions(threshold=np.nan)
import math
from itertools import compress
import h5py
import matplotlib.ticker as ticker
torch.manual_seed(1)

def isnan(x):
    return x != x

def one_hot_encode(primary):
    """
    One-hot-encode an array of integer-labelled amino acids
    """
    aa_dict_onehot = {0:[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Zeros will be masked out
                      1:[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                      2:[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      3:[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      4:[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      5:[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      6:[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      7:[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      8:[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                      9:[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                      10:[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                      11:[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                      12:[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                      13:[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                      14:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                      15:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                      16:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],                                    
                      17:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                      18:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                      19:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                      20:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
    
    # One-hot-encode data
    primary_hot = np.zeros((len(primary),len(primary[0]),21))
    for i in range(len(primary)):
        for j in range(len(primary[0])):
            primary_hot[i,j,:] = aa_dict_onehot[primary[i,j]]
    
    return primary_hot

def string_encoder(primary):
    """
    Converts an array of integer-labelled amino acids to string format
    """    
    aa_dict = {"-": "0", 'A': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5', 'G': '6', 'H': '7', 'I': '8', 'K': '9', 'L': '10', 'M': '11', 'N': '12', 'P': '13', 'Q': '14', 'R': '15', 'S': '16', 'T': '17', 'V': '18', 'W': '19', 'Y': '20'}
    aa_dict_inverse = {v: k for k, v in aa_dict.items()}
    
    # String-encode data
    primary_string = []
    for i in range(len(primary)):
        for j in range(len(primary[i])):
            primary_string += aa_dict_inverse[str(int(primary[i,j]))]
    
    return primary_string

def load_h5py_batch(test, batch_size):
    """
    Loads batches directly from h5py file
    """
    if test == False:
        
        numprots = 224
        batchidx = np.random.choice(range(0,numprots),batch_size,replace=False)   
        batchidx = np.sort(batchidx)

        # Load batch
        infile = h5py.File("h5py_val",'r')
        primary = infile['primary_val'][batchidx,:]
        evolutionary = infile['evolutionary_val'][batchidx,:,:]
        tertiary = infile['tertiary_val'][batchidx,:,:,:]
        mask = infile['mask_val'][batchidx,:]
        dihedrals = np.load("dihedrals_val.npy")[batchidx]
        infile.close()
        maxseqlen = len(primary[0])

        # Convert primary sequences to both one-hot encoded and string-based versions
        primary_string = np.reshape(string_encoder(primary),(batch_size, maxseqlen))
        primary = one_hot_encode(primary)
        
        # Merge primary and evolutionary data to use as input
        seq = np.dstack((evolutionary,primary))
        
        # Convert coordinates from nanometers to angstromn
        tertiary = np.divide(tertiary,100)

        # Convert dihedrals to radians
        dihedrals = np.multiply(dihedrals,math.pi/180)            
    
        # Convert to PyTorch variables
        seq = Variable(torch.from_numpy(seq)).type(torch.FloatTensor).cuda()
        dihedrals = Variable(torch.from_numpy(dihedrals)).type(torch.FloatTensor).cuda()
        tertiary = Variable(torch.from_numpy(tertiary)).type(torch.FloatTensor).cuda()
        mask = Variable(torch.from_numpy(mask.astype("uint8"))).cuda()
        
    elif test == True:
        
        numprots = batch_size
        
        # Load batch
        infile = h5py.File("h5py_test",'r')
        primary = infile['primary_test'][:]
        evolutionary = infile['evolutionary_test'][:]
        tertiary = infile['tertiary_test'][:]
        mask = infile['mask_test'][:]
        dihedrals = np.load("dihedrals_test.npy")[:]
        infile.close()
        maxseqlen = len(primary[0])
        
        # Convert primary sequences to both one-hot encoded and string-based versions
        primary_string = np.reshape(string_encoder(primary),(numprots, maxseqlen))
        primary = one_hot_encode(primary)
        
        # Merge primary and evolutionary data to use as input
        seq = np.dstack((evolutionary,primary))
        
        # Convert coordinates from nanometers to angstromn
        tertiary = np.divide(tertiary,100)
       
        # Convert dihedrals to radians
        dihedrals = np.multiply(dihedrals,math.pi/180)     
        
        # Convert to PyTorch variables
        seq = Variable(torch.from_numpy(seq)).type(torch.FloatTensor).cuda()
        dihedrals = Variable(torch.from_numpy(dihedrals)).type(torch.FloatTensor).cuda()
        tertiary = Variable(torch.from_numpy(tertiary)).type(torch.FloatTensor).cuda()
        mask = Variable(torch.from_numpy(mask.astype("uint8"))).cuda()
        
    # Return 
    return seq, primary_string, dihedrals, tertiary, mask

def predictangle(p, phi_input_sin, phi_input_cos, psi_input_sin, psi_input_cos, omega_input_sin, omega_input_cos, gpu):
    """
    Computes dihedral angles based on softmax weights and mixture components
    """
    if gpu:
        p, phi_input_sin, phi_input_cos, psi_input_sin, psi_input_cos, omega_input_sin, omega_input_cos = p.cuda(), phi_input_sin.cuda(), phi_input_cos.cuda(), psi_input_sin.cuda(), psi_input_cos.cuda(), omega_input_sin.cuda(), omega_input_cos.cuda()
    
    eps = 10**(-4)
    phi = torch.atan2(phi_input_sin, phi_input_cos +eps)
    psi = torch.atan2(psi_input_sin, psi_input_cos +eps)
    omega = torch.atan2(omega_input_sin, omega_input_cos +eps)
    
    return torch.cat((phi.view(-1,1), psi.view(-1,1), omega.view(-1,1)),1)

class angulardeviation(torch.nn.Module):
    """
    Computes angular deviation between two Torch variables of angles [radians]
    """
    def __init__(self):
        super(angulardeviation,self).__init__()

    def forward(self, X1, X2):
        # Compute angular deviation
        X1 = X1.view(-1,1)
        X2 = X2.view(-1,1)
        
        if len(X1)>8:
            X1 = X1[1:-6,:]
            X2 = X2[1:-6,:]
        
        tmp = torch.min(torch.abs(X2-X1),2*math.pi-torch.abs(X2-X1))
        angdev = torch.sqrt(torch.mean(tmp**2))
        return angdev
    
def bondAngleConstants(atom1,atom2,atom3,res):
    """
    Engh, Huber, 1991
    """
    if atom1 == "C" and atom2 == "N" and atom3 == "CA":
        if res == "G":
            bondAngle = 120.6
        elif res == "P":
            bondAngle = 122.6
        else:
            bondAngle = 121.7

    if atom1 == "N" and atom2 == "CA" and atom3 == "C":
        if res == "G":
            bondAngle = 112.5
        elif res == "P":
            bondAngle = 111.8
        else:
            bondAngle = 111.2

    if atom1 == "CA" and atom2 == "C" and atom3 == "N":
        if res == "G":
            bondAngle = 116.4
        elif res == "PRO":
            bondAngle = 116.9
        else:
            bondAngle = 116.2

    # Return in radians
    return bondAngle*math.pi/180

def bondLengthConstants(atom1,atom2,res):
    if (atom1 == "N" and atom2 == "CA") or (atom1 == "CA" and atom2 == "N"):
        if res == "G":
            bondLength = 1.451
        elif res == "P":
            bondLength = 1.466
        else:
            bondLength = 1.458

    elif (atom1 == "CA" and atom2 == "C") or (atom1 == "C" and atom2 == "CA"):
        if res == "G":
            bondLength = 1.516
        else:
            bondLength = 1.525

    elif (atom1 == "C" and atom2 == "N") or (atom1 == "N" and atom2 == "C"):
        if res == "P":
            bondLength = 1.341
        else:
            bondLength = 1.329

    elif (atom1 == "CA" and atom2 == "CB") or (atom1 == "CB" and atom2 == "CA"):
        if res == "A":
            bondLength = 1.521
        elif res == "I" or res == "T" or res == "V":
            bondLength = 1.540
        else:
            bondLength = 1.530

    elif (atom1 == "C" and atom2 == "O") or (atom1 == "O" and atom2 == "C"):
        bondLength = 1.231

    return bondLength

def nerf(a, b, c, l, theta, chi, gpu):
    '''
    Nerf method of finding 4th coord (d)
    in cartesian space
    Params:
    a, b, c : coords of 3 points
    l : bond length between c and d
    theta : bond angle between b, c, d (in degrees)
    chi : dihedral using a, b, c, d (in degrees)
    Returns:
    d : tuple of (x, y, z) in cartesian space
    '''
 
    # Define scalar tensors
    theta = torch.Tensor([theta])
    l = torch.Tensor([l])
    
    # calculate unit vectors AB and BC
    ab_unit = torch.div((b-a),torch.norm(b-a,2))
    bc_unit = torch.div((c-b),torch.norm(c-b,2))
    
    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(ab_unit, bc_unit)
    n_unit = torch.div(n_unit,torch.norm(n_unit,2))
    p_unit = torch.cross(n_unit, bc_unit)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.cat((bc_unit.unsqueeze(0), p_unit.unsqueeze(0), 
                         n_unit.unsqueeze(0)),0).transpose(0,1)
    
    theta, l = Variable(theta), Variable(l)
    
    if gpu == True:
        theta, chi, l = theta.cuda(), chi.cuda(), l.cuda()

    # calculate coord pre rotation matrix
    d2 = torch.cat((-l*torch.cos(theta), 
                          l*torch.sin(theta)*torch.cos(chi), 
                          l*torch.sin(theta)*torch.sin(chi)))

    # calculate with rotation as our final output    
    result = c + torch.matmul(M, d2)     
    
    return result

def place_residues(torsions,aaSequence, gpu):
    '''
    Do the actual placement of residues
    based on nerf method.
    '''
    # place the 0th carbonyl C at origin
    n_i_zero = torch.Tensor([0.0, 0.0, 0.0])

    # using CH1E-NH1 bond length = 1.458 ang
    # place C_alpha on pos x axis
    c_a_zero = torch.Tensor([1.458, 0.0, 0.0])

    # place C_carbonyl on xy plane using
    # CH1E-C bond length = 1.525 ang
    # C-CH1E-NH1 bond angle = 111.2 deg
    l = torch.Tensor([1.525])
    theta = torch.Tensor([111.2*math.pi/180])  # in radians
    c_o_zero = torch.add(c_a_zero,torch.cat((-l*torch.cos(theta),l*torch.sin(theta),torch.Tensor([0]))))

    n_i_zero, c_a_zero, l, theta, c_o_zero, l, theta = Variable(n_i_zero), Variable(c_a_zero), Variable(l), Variable(theta), Variable(c_o_zero), Variable(l), Variable(theta)

    if gpu == True:
        n_i_zero, c_a_zero, l, theta, c_o_zero = n_i_zero.cuda(), c_a_zero.cuda(), l.cuda(), theta.cuda(), c_o_zero.cuda()
    
    # start a listing of atoms starting with the zeroth
    # residue represented as a tuple of N, C_alpha, C_carbonyl
    atom_coords = torch.cat((n_i_zero.unsqueeze(0), c_a_zero.unsqueeze(0), c_o_zero.unsqueeze(0)),0)
    
    torsions = torsions.view(-1,1)
    residx = 0
    for i in range(0,len(torsions.data)-3,3):
        
        # calculate coords
        n_i = atom_coords[i,:]
        c_a = atom_coords[i+1,:]
        c_o = atom_coords[i+2,:]

        phi = torsions[i+3]
        psi = torsions[i+1]
        omega  = torsions[i+2]
        
        n_i_plus_1 = nerf(n_i, c_a, c_o, l=bondLengthConstants("C","N", aaSequence[residx]), theta=bondAngleConstants("CA","C","N", aaSequence[residx]), chi=psi, gpu=gpu)
        c_a_i_plus_1 = nerf(c_a, c_o, n_i_plus_1,l=bondLengthConstants("N","CA", aaSequence[residx]), theta=bondAngleConstants("C","N","CA", aaSequence[residx]), chi=omega, gpu=gpu)
        c_o_i_plus_1 = nerf(c_o, n_i_plus_1, c_a_i_plus_1,l=bondLengthConstants("CA","C", aaSequence[residx]), theta=bondAngleConstants("N","CA","C", aaSequence[residx]), chi=phi, gpu=gpu)
        
        new_res = torch.cat((n_i_plus_1.unsqueeze(0), c_a_i_plus_1.unsqueeze(0), c_o_i_plus_1.unsqueeze(0)),0)
        atom_coords = torch.cat((atom_coords,new_res),0)
    
        residx += 1
    
    return atom_coords

def pairwise_distances(x, gpu):
    y = x
    dtype = x.data.type()
    dist_mat = Variable(torch.Tensor(x.size()[0], y.size()[0]).type(dtype))
    
    if gpu:
        dist_mat = dist_mat.cuda()

    for i, row in enumerate(x.split(1)):
        r_v = row.expand_as(y)
        sq_dist = torch.sum((r_v - y) ** 2, 1)
        dist_mat[i] = sq_dist.view(1, -1)

    eps = Variable(10**(-4)*torch.ones(x.size(0),x.size(0))) # Required as the gradient of the sqrt function at zero is nan
    if gpu == True:
        eps = eps.cuda()

    dist_mat = torch.sqrt(dist_mat+eps)
    
    return torch.clamp(dist_mat, 0.0, 10**3)

class dRMSD(torch.nn.Module):
    """
    Computes root-mean-squared-deviation between arrays of atomic positions
    Assumes array structure ala Nx3
    """

    def __init__(self):
        super(dRMSD,self).__init__()

    def forward(self, predictions, targets, gpu):
        if torch.sum(targets[0,:]).data[0] == 0:
            targets = targets[1:,:]
            predictions = predictions[1:,:]

        L = len(predictions)
        
        distmat_pred = pairwise_distances(predictions, gpu)
        distmat_tar = pairwise_distances(targets, gpu)
        
        D = distmat_pred - distmat_tar
        
        if L != 1:
            dRMSD = torch.norm(D,2)/math.sqrt((L*(L-1)))
        else:
            dRMSD = torch.norm(D,2)/1
        
        return dRMSD

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]/n

def write_to_pdb(atomic_coords, aaSequence, identifier):
    '''
    Write out calculated coords to PDB
    '''
    aa_name_dict = {"A": "ALA",
                    "R": "ARG",
                    "N": "ASN",
                    "D": "ASP",
                    "C": "CYS",
                    "E": "GLU",
                    "Q": "GLN",
                    "G": "GLY",
                    "H": "HIS",
                    "I": "ILE",
                    "L": "LEU",
                    "K": "LYS",
                    "M": "MET",
                    "F": "PHE",
                    "P": "PRO",
                    "S": "SER",
                    "T": "THR",
                    "W": "TRP",
                    "Y": "TYR",
                    "V": "VAL",
                    } 
    
    atomic_coords = np.round(atomic_coords,3)
    
    atom_names = ['N', 'CA', 'C']*(len(atomic_coords/3))
    out_file = open("protein_" + str(identifier) + ".pdb", 'w')

    res_number = 0
    for i, coord in enumerate(atomic_coords):

        if i % 3 == 0:
            res_number += 1
        out_file.write('ATOM' +
                       ' '*2 + ' '*(5-len(str(i+1))) + ' ' +
                       str(i+1) + ' '*(4-len(atom_names[i])) +
                       atom_names[i] + ' ' + aa_name_dict[aaSequence[res_number-1]] +
                       ' ' + "A" + ' '*(4-len(str(res_number))) +
                       str(res_number) + ' '*4 +
                       ' '*(8-len(str(coord[0]))) + str(coord[0]) +
                       ' '*(8-len(str(coord[1]))) + str(coord[1]) +
                       ' '*(8-len(str(coord[2]))) + str(coord[2]) +
                       '\n')
    out_file.close()

# Define net
class soft_to_angle(nn.Module):
    def __init__(self):
        super(soft_to_angle, self).__init__()       
        # Omega intializer
        omega_components1 = np.random.uniform(0,1,50) # Initialize omega 90/10 pos/neg
        omega_components2 = np.random.uniform(2,math.pi,450)
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)
        
        phi_components = np.genfromtxt("mixture_model_pfam_500.txt")[:,1]
        psi_components = np.genfromtxt("mixture_model_pfam_500.txt")[:,2]
        
        self.phi_components = nn.Parameter(torch.from_numpy(phi_components).contiguous().view(-1,1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components).contiguous().view(-1,1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components).view(-1,1).float())
        
    def forward(self, x):        
        phi_input_sin =  torch.matmul(x,torch.sin(self.phi_components))
        phi_input_cos = torch.matmul(x,torch.cos(self.phi_components))
        psi_input_sin =  torch.matmul(x,torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(x,torch.cos(self.psi_components))
        omega_input_sin =  torch.matmul(x,torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(x,torch.cos(self.omega_components))

        eps = 10**(-4)
        phi = torch.atan2(phi_input_sin, phi_input_cos +eps)
        psi = torch.atan2(psi_input_sin, psi_input_cos +eps)
        omega = torch.atan2(omega_input_sin, omega_input_cos +eps)        
    
        return torch.cat((phi, psi, omega),2)
soft_to_angle = soft_to_angle().cuda()

class Net(nn.Module):
    def __init__(self, inputdim, kernel, stride, lstm_layers, hidden_dim, out_classes):
        super(Net, self).__init__()
        self.inputdim = inputdim
        self.out_classes = out_classes
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(inputdim+600, hidden_dim, lstm_layers, bias=True, bidirectional=True, batch_first=True)       
        
        self.conv_init = nn.Conv1d(inputdim,200,kernel,stride,kernel//2,bias=True)
        self.conv1 = nn.Conv1d(200,200,kernel,stride,kernel//2,bias=True)
        self.conv2 = nn.Conv1d(200,200,kernel,stride,kernel//2,bias=True)
        self.conv4 = nn.Conv1d(200,300,kernel,stride,kernel//2,bias=True)
        self.conv5 = nn.Conv1d(300,300,kernel,stride,kernel//2,bias=True)
        self.conv6_expand = nn.Conv1d(200,300,1,stride,1//2,bias=True)
        self.conv7 = nn.Conv1d(300,400,kernel,stride,kernel//2,bias=True)
        self.conv8 = nn.Conv1d(400,400,kernel,stride,kernel//2,bias=True)
        self.conv9_expand = nn.Conv1d(300,400,1,stride,1//2,bias=True)
        self.conv10 = nn.Conv1d(400,500,kernel,stride,kernel//2,bias=True)
        self.conv11 = nn.Conv1d(500,500,kernel,stride,kernel//2,bias=True)
        self.conv12_expand = nn.Conv1d(400,500,1,stride,1//2,bias=True)
        self.conv13 = nn.Conv1d(500,600,kernel,stride,kernel//2,bias=True)
        self.conv14 = nn.Conv1d(600,600,kernel,stride,kernel//2,bias=True)
        self.conv15_expand = nn.Conv1d(500,600,1,stride,1//2,bias=True)
        self.conv16 = nn.Conv1d(600,600,kernel,stride,kernel//2,bias=True)
        self.conv17 = nn.Conv1d(600,600,kernel,stride,kernel//2,bias=True)
        self.conv18_expand = nn.Conv1d(600,600,1,stride,1//2,bias=True)
        self.conv19 = nn.Conv1d(600,600,kernel,stride,kernel//2,bias=True)
        self.conv20 = nn.Conv1d(600,600,kernel,stride,kernel//2,bias=True)
        self.conv21_expand = nn.Conv1d(600,600,1,stride,1//2,bias=True)
        self.conv22 = nn.Conv1d(600,600,kernel,stride,kernel//2,bias=True)
        self.conv23 = nn.Conv1d(600,600,kernel,stride,kernel//2,bias=True)
        self.conv24_expand = nn.Conv1d(600,600,1,stride,1//2,bias=True)
        
        self.conv_bn_init = nn.BatchNorm1d(200)
        self.conv_bn1 = nn.BatchNorm1d(200)
        self.conv_bn2 = nn.BatchNorm1d(200)
        self.conv_bn4 = nn.BatchNorm1d(200)
        self.conv_bn5 = nn.BatchNorm1d(300)
        self.conv_bn7 = nn.BatchNorm1d(300)
        self.conv_bn8 = nn.BatchNorm1d(400)
        self.conv_bn10 = nn.BatchNorm1d(400)
        self.conv_bn11 = nn.BatchNorm1d(500)
        self.conv_bn13 = nn.BatchNorm1d(500)
        self.conv_bn14 = nn.BatchNorm1d(600)
        self.conv_bn16 = nn.BatchNorm1d(600)
        self.conv_bn17 = nn.BatchNorm1d(600)
        self.conv_bn19 = nn.BatchNorm1d(600)
        self.conv_bn20 = nn.BatchNorm1d(600)
        self.conv_bn22 = nn.BatchNorm1d(600)
        self.conv_bn23 = nn.BatchNorm1d(600) 

        self.fc = nn.Linear(self.hidden_dim*2+600, out_classes, bias=True)
        
        self.bn = nn.BatchNorm1d(out_classes)

        self.soft = nn.LogSoftmax(2)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.lstm_layers*2, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.lstm_layers*2, batch_size, self.hidden_dim)))

    def forward(self, inputs, batch_size):        
        # Turn (batch x seqlen x features) into (batch x features x seqlen) for CNN
        init = inputs.transpose(1,2)
        init = F.leaky_relu(self.conv_bn_init(self.conv_init(init)))

        res = self.conv1(F.leaky_relu(self.conv_bn1(init)))
        res = self.conv2(F.leaky_relu(self.conv_bn2(res)))
        
        x_conv = res + init

        res = self.conv4(F.leaky_relu(self.conv_bn4(x_conv)))
        res = self.conv5(F.leaky_relu(self.conv_bn5(res)))
        
        x_conv = self.conv6_expand(x_conv)
        x_conv += res

        res = self.conv7(F.leaky_relu(self.conv_bn7(x_conv)))
        res = self.conv8(F.leaky_relu(self.conv_bn8(res)))
        
        x_conv = self.conv9_expand(x_conv)
        x_conv += res

        res = self.conv10(F.leaky_relu(self.conv_bn10(x_conv)))
        res = self.conv11(F.leaky_relu(self.conv_bn11(res)))
        
        x_conv = self.conv12_expand(x_conv)
        x_conv += res

        res = self.conv13(F.leaky_relu(self.conv_bn13(x_conv)))
        res = self.conv14(F.leaky_relu(self.conv_bn14(res)))
        
        x_conv = self.conv15_expand(x_conv)
        x_conv += res

        res = self.conv16(F.leaky_relu(self.conv_bn16(x_conv)))
        res = self.conv17(F.leaky_relu(self.conv_bn17(res)))
        
        x_conv = self.conv18_expand(x_conv)
        x_conv += res

        res = self.conv19(F.leaky_relu(self.conv_bn19(x_conv)))
        res = self.conv20(F.leaky_relu(self.conv_bn20(res)))
        
        x_conv = self.conv21_expand(x_conv)
        x_conv += res

        res = self.conv22(F.leaky_relu(self.conv_bn22(x_conv)))
        res = self.conv23(F.leaky_relu(self.conv_bn23(res)))
        
        x_conv = self.conv24_expand(x_conv)
        x_conv += res
        
        x_conv = x_conv.transpose(1,2)
        
        x_lstm = torch.cat((x_conv, inputs), 2) # Concatenate original input and output CNN to LSTM
        x_lstm, self.hidden = self.lstm(x_lstm)

        x_total = torch.cat((x_lstm, x_conv), 2) # Concatenate output from LSTM and output from CNN

        x_total = self.fc(x_total)
        x_total = x_total.view(batch_size, self.out_classes, -1)
        x_total = self.bn(x_total)
        x_total = x_total.view(batch_size, -1, self.out_classes)
        p = torch.exp(self.soft(x_total))
        output = soft_to_angle(p)
        
        return output

net = Net(42, 5, 1, 2, 600, 500).cuda()
print(net)

# Define loss function
criterion_angle = angulardeviation().cuda()
criterion_coords = dRMSD().cuda()
lr = 0.001
params = list(soft_to_angle.parameters()) + list(net.parameters())
optimizer = optim.Adam(params, lr, amsgrad=True)

# Misc parameters
batch_size_train = 5
batch_size_test = 81

loss_train_angles_vec = np.zeros((batch_size_train,1))
loss_train_coords_vec = np.zeros((batch_size_train,1))

loss_test_angles_vec = np.zeros(batch_size_test)
loss_test_coords_vec = np.zeros(batch_size_test)

batch_losses_train_angles = []
batch_losses_train_coords = []

batch_losses_test_angles = []
batch_losses_test_coords = []

testcycles = []

print('Starting training')
for cycle in range(14000):

    # Initialize
    net.train()
    
    # Load  batch
    inputs, aaSequence, target_angles, target_coords, mask = load_h5py_batch(test=False, batch_size=batch_size_train)
    
    # Run input through network
    net.hidden = net.init_hidden(batch_size_train)
    outputs = net(inputs, batch_size_train)
    
    for i in range(batch_size_train):
 
        # Split data into batches
        inputs_batch = inputs[i]
        aaSequence_batch = aaSequence[i]
        target_angles_batch = target_angles[i].view(-1,3)
        target_coords_batch = target_coords[i].view(-1,3)
        mask_batch = mask[i].view(-1,1)
        mask_batch_coords = mask_batch.unsqueeze(2).expand(-1,-1,9).contiguous().view(-1,3)
        mask_batch_angles = mask_batch.expand(-1,3)
        outputs_batch = outputs[i]

        if len(aaSequence_batch)<10: 
            print("Short protein fragment detected!")
            optimizer.zero_grad()
            break
        
        # Compute loss 1
        aaSequence_batch = list(compress(aaSequence_batch, np.array(mask_batch.squeeze(1).cpu().data.numpy(),dtype=bool).tolist()))
        outputs_batch = torch.masked_select(outputs_batch, mask_batch_angles).view(-1,3)
        target_angles_batch = torch.masked_select(target_angles_batch, mask_batch_angles).view(-1,3)
        loss_train_angles = criterion_angle(outputs_batch, target_angles_batch)
        
        # Compute loss 2 and optimize
        outputs_coords = place_residues(outputs_batch, aaSequence_batch,  gpu=True)
        target_coords_batch = target_coords_batch.view(-1,3)
        target_coords_batch = torch.masked_select(target_coords_batch, mask_batch_coords).view(-1,3)
        loss_train_coords = criterion_coords(outputs_coords, target_coords_batch, gpu=True)
        
        # Sum losses and save gradients
        loss_train = (0.5*loss_train_coords + 6*loss_train_angles)/batch_size_train
        loss_train.backward(retain_graph=True)

        # Clip gradients
        for p in params:
            p.grad.data.clamp_(min=-5,max=5)
            
        # Save intermediate results in batch vector   
        loss_train_angles_vec[i] = loss_train_angles.data[0]
        loss_train_coords_vec[i] = loss_train_coords.data[0]
            
        if int(i) == batch_size_train-1:
            # Update results 
            batch_losses_train_angles += [np.mean(loss_train_angles_vec)]
            batch_losses_train_coords += [np.mean(loss_train_coords_vec)]      
        
            # Step optimizer
            optimizer.step()
            optimizer.zero_grad()
        
    # Run validation
    if cycle % 1000 == 999:    # print every 1000th cycle
        testcycles.append(cycle+1)
        optimizer.zero_grad()
        net.eval()
               
        # Load  batch
        inputs, aaSequence, target_angles, target_coords, mask = load_h5py_batch(test=True, batch_size=batch_size_test)
        
        for i in range(batch_size_test):
     
            # Split data into batches
            aaSequence_batch = aaSequence[i]
            target_angles_batch = target_angles[i].view(-1,3)
            target_coords_batch = target_coords[i].view(-1,3)
            mask_batch = mask[i].view(-1,1)
            mask_batch_coords = mask_batch.unsqueeze(2).expand(-1,-1,9).contiguous().view(-1,3)
            mask_batch_angles = mask_batch.expand(-1,3)
            inputs_batch = inputs[i].unsqueeze(0)
    
            # Run input through network
            net.hidden = net.init_hidden(1)
            outputs_batch = net(inputs_batch, 1)
    
            # Compute loss 1
            aaSequence_batch = list(compress(aaSequence_batch, np.array(mask_batch.squeeze(1).cpu().data.numpy(),dtype=bool).tolist()))
            outputs_batch = torch.masked_select(outputs_batch, mask_batch_angles).view(-1,3)
            target_angles_batch = torch.masked_select(target_angles_batch, mask_batch_angles).view(-1,3)
            loss_test_angles = criterion_angle(outputs_batch, target_angles_batch)
            
            # Compute loss 2 and optimize
            outputs_coords = place_residues(outputs_batch, aaSequence_batch,  gpu=True)
            target_coords_batch = target_coords_batch.view(-1,3)
            target_coords_batch = torch.masked_select(target_coords_batch, mask_batch_coords).view(-1,3)
            loss_test_coords = criterion_coords(outputs_coords, target_coords_batch, gpu=True)
                
            loss_test_angles_vec[i] = loss_test_angles.data[0]
            loss_test_coords_vec[i] = loss_test_coords.data[0]
                
        batch_losses_test_angles += [np.mean(loss_test_angles_vec)]
        batch_losses_test_coords += [np.mean(loss_test_coords_vec)]
    
        print('Cycle %d \t Test angular dev.: %.1f \t Test dRMSD: %.1f' %(cycle + 1, batch_losses_test_angles[-1], batch_losses_test_coords[-1]))
        net = net.cuda()
        
print('Finished Training')

# Split test error into FM and TBM part
FM_idx = [1,5,7,9,10,13,15,16,26,35,43,55,68,72,73]
TBM_idx = [0,2,4,6,8,11,12,14,17,18,19,20,21,22,23,24,25,27,29,30,31,32,33,34,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54,56,57,58,59,61,62,63,64,65,66,67,69,70,71,75,76,77,78,79,80]
UC_idx = [3,28,60,74]

loss_test_angles_vec_FM = loss_test_angles_vec[FM_idx]
loss_test_coords_vec_FM = loss_test_coords_vec[FM_idx]

loss_test_angles_vec_TBM = loss_test_angles_vec[TBM_idx]
loss_test_coords_vec_TBM = loss_test_coords_vec[TBM_idx]

loss_test_angles_vec_UC = loss_test_angles_vec[UC_idx]
loss_test_coords_vec_UC = loss_test_coords_vec[UC_idx]

print('Test FM ang. dev.: %.1f \t Test FM dRMSD: %.1f \t Test TBM ang. dev.: %.1f \t Test TBM dRMSD: %.1f' %(np.mean(loss_test_angles_vec_FM), np.mean(loss_test_coords_vec_FM), np.mean(loss_test_angles_vec_TBM), np.mean(loss_test_coords_vec_TBM)))

# Smooth out training error
smoothsize = 500
batch_losses_train_angles_smooth = moving_average(batch_losses_train_angles, smoothsize)
batch_losses_train_coords_smooth = moving_average(batch_losses_train_coords, smoothsize)

# Convert to degrees
batch_losses_train_angles = np.multiply(batch_losses_train_angles, 180/math.pi)
batch_losses_test_angles = np.multiply(batch_losses_test_angles, 180/math.pi)
batch_losses_train_angles_smooth = np.multiply(batch_losses_train_angles_smooth, 180/math.pi)
batch_losses_test_angles = np.multiply(batch_losses_test_angles, 180/math.pi)

# Save results
np.save("Results_ConvLSTM_train_coords", batch_losses_train_coords)
np.save("Results_ConvLSTM_train_angles", batch_losses_train_angles)
np.save("Results_ConvLSTM_test_coords", batch_losses_test_coords)
np.save("Results_ConvLSTM_test_angles", batch_losses_test_angles)

# Plot loss
fig, (ax1_1, ax2_1) = plt.subplots(2, 1, sharey=False)
ax1_1.plot(batch_losses_train_coords_smooth,label="dRMSD", color="crimson")
ax1_1.set_ylim([0,60])
ax1_1.set_title("Training loss")
ax1_1.set_ylabel("dRMSD [Å]")
ax1_1.set_xlabel("Iterations $10^%i$" %int(np.log10(smoothsize)))
ax1_1.tick_params(axis='y')
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/smoothsize))
ax1_1.xaxis.set_major_formatter(ticks_x)
ax1_2 = ax1_1.twinx()
ax1_2.plot(batch_losses_train_angles_smooth,label="Ang. dev.", color="darkblue")
ax1_2.set_ylim([0,115])
ax1_2.set_ylabel("Ang. dev. [degrees]")
ax1_2.tick_params(axis='y')
lines, labels = ax1_1.get_legend_handles_labels()
lines2, labels2 = ax1_2.get_legend_handles_labels()
ax1_2.legend(lines + lines2, labels + labels2, loc="upper right")

ax2_1.plot(batch_losses_test_coords,label="dRMSD", color="crimson")
ax2_1.set_title("Test loss")
ax2_1.set_ylim([0,60])
ax2_1.set_ylabel("dRMSD [Å]")
ax2_1.set_xlabel("Iterations $10^%i$" %int(np.log10(smoothsize)))
ax2_1.tick_params(axis='y')
ax2_2 = ax2_1.twinx()
ax2_2.plot(batch_losses_test_angles,label="Ang. dev.", color="darkblue")
ax2_2.set_ylim([0,115])
ax2_2.set_ylabel("Ang. dev. [degrees]")
ax2_2.tick_params(axis='y')
ax2_2.legend(loc="upper right")
lines, labels = ax2_1.get_legend_handles_labels()
lines2, labels2 = ax2_2.get_legend_handles_labels()
ax2_2.legend(lines + lines2, labels + labels2, loc="upper right")
plt.tight_layout()
plt.savefig("Training_ConvLSTM.png")
plt.close(fig)

# Make test dRMSD vs sequence length plot
infile = h5py.File("h5py_test",'r')
primary = infile['primary_test'][:,:]
seqLengths = np.zeros(len(primary))
for i in range(len(primary)):
    for j in range(len(primary[i])):
        if primary[i][j] == 0 or j == len(primary[i])-1:
            seqLengths[i] = j
            if j == len(primary[i])-1:
                seqLengths[i] = j+1
            break
seqLengths_FM = seqLengths[FM_idx]
seqLengths_TBM = seqLengths[TBM_idx]            
seqLengths_UC = seqLengths[UC_idx]         

fig, ax1 = plt.subplots(1, 1, sharey=True)
ax1.scatter(seqLengths_FM, loss_test_coords_vec_FM, label="FM", color="darkblue")
ax1.set_ylabel("Test dRMSD [Å]")
ax1.set_xlabel("Sequence length")
plt.xlim([0,700])
ax1.scatter(seqLengths_TBM, loss_test_coords_vec_TBM, label="TBM", color="crimson")
ax1.scatter(seqLengths_UC, loss_test_coords_vec_UC, label="Unclassified", color="darkgreen")
ax1.legend(loc="upper right")
plt.tight_layout()
plt.savefig("Test_ConvLSTM.png")
plt.close(fig)
    
# Make visualizations
inputs, aaSequence, target_angles, target_coords, mask = load_h5py_batch(test=True)
pdblist = np.argpartition(loss_test_coords_vec, 6)[:6]
for i in pdblist:
    inputs_batch = inputs[i].unsqueeze(0)
    aaSequence_batch = aaSequence[i]
    mask_batch = mask[i].view(-1,1)
    aaSequence_batch = list(compress(aaSequence_batch,np.array(mask_batch.squeeze(1).cpu().data.numpy(),dtype=bool).tolist()))
    outputs_batch = net(inputs_batch, 1)
    mask_batch_angles = mask_batch.expand(-1,3)
    outputs_batch = torch.masked_select(outputs_batch, mask_batch_angles).view(-1,3)
    outputs_batch = place_residues(outputs_batch, aaSequence_batch, gpu=True)
    write_to_pdb(outputs_batch.data.cpu().numpy(), aaSequence_batch, str(i)+"_prediction_ConvLSTM_newVisuals")

print(pdblist)
print(loss_test_coords_vec[pdblist[0]])
print(loss_test_coords_vec[pdblist[1]])
print(loss_test_coords_vec[pdblist[2]])
print(loss_test_coords_vec[pdblist[3]])
print(loss_test_coords_vec[pdblist[4]])
print(loss_test_coords_vec[pdblist[5]])

# Plot componenets
phi_components = list(soft_to_angle.parameters())[0]*180/math.pi
psi_components = list(soft_to_angle.parameters())[1]*180/math.pi

plt.scatter(phi_components.cpu().data.numpy(), psi_components.cpu().data.numpy(), marker='.', s=5, alpha=0.5,color="darkblue")
plt.xlabel("$\phi$ angle [degrees]")
plt.ylabel("$\psi$ angle [degrees]")
plt.savefig("Ramachandran_ConvLSTM.png")
plt.close()