# imports
import sys
import re
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
# Constants
NUM_DIMENSIONS = 3
num_evo_entries = 21

_aa_dict = {'A': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5', 'G': '6', 'H': '7', 'I': '8', 'K': '9', 'L': '10', 'M': '11', 'N': '12', 'P': '13', 'Q': '14', 'R': '15', 'S': '16', 'T': '17', 'V': '18', 'W': '19', 'Y': '20'}
_dssp_dict = {'L': '0', 'H': '1', 'B': '2', 'E': '3', 'G': '4', 'I': '5', 'T': '6', 'S': '7'}
_mask_dict = {'-': '0', '+': '1'}

class switch(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5
            self.fall = True
            return True
        else:
            return False

def letter_to_num(string, dict_):
    """ Convert string of letters to list of ints """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num

def read_record(file_, num_evo_entries):
    """ Read a protein record from file and convert into dict. """
        
    dict_ = {}

    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                id_ = file_.readline()[:-1]
                dict_.update({'id': id_})
            elif case('[PRIMARY]' + '\n'):
                primary = letter_to_num(file_.readline()[:-1], _aa_dict)
                dict_.update({'primary': primary})
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries): evolutionary.append([float(step) for step in file_.readline().split()])
                dict_.update({'evolutionary': evolutionary})
            elif case('[SECONDARY]' + '\n'):
                secondary = letter_to_num(file_.readline()[:-1], _dssp_dict)
                dict_.update({'secondary': secondary})
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord) for coord in file_.readline().split()])
                dict_.update({'tertiary': tertiary})
            elif case('[MASK]' + '\n'):
                mask = letter_to_num(file_.readline()[:-1], _mask_dict)
                dict_.update({'mask': mask})
            elif case('\n'):
                return dict_
            elif case(''):
                return None

def write_to_pdb(residue_coords,aaSequence):
    '''
    Write out calculated coords to PDB
    '''

    atom_names = ['N', 'CA', 'C']*(len(residue_coords))
    out_file = open('pdb_test.pdb', 'w')
    
    res_number = 0
    for i in range((len(residue_coords))):
        coord = np.round(residue_coords[i])
        
        if i % 3 == 0:
            res_number += 1
        out_file.write('ATOM' +
                           ' '*2 + ' '*(5-len(str(i+1))) + ' ' +
                           str(i+1) + ' '*(4-len(atom_names[i])) +
                           atom_names[i] + ' ' + 'ALA' +
                           ' ' + aaSequence[res_number-1] + ' '*(4-len(str(res_number))) +
                           str(res_number) + ' '*4 +
                           ' '*(8-len(str(coord[0]))) + str(coord[0]) +
                           ' '*(8-len(str(coord[1]))) + str(coord[1]) +
                           ' '*(8-len(str(coord[2]))) + str(coord[2]) +
                           '\n')
    out_file.close()
 
def new_dihedral(p):
    """
    Praxeolitic formula
    1 sqrt, 1 cross product
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    return np.degrees(np.arctan2(y, x))

# Load data           
try:
    file_ = open("testing", "r")
except IOError as error:
    sys.stderr.write("File I/O error, reason: "+str(error)+"\n")
    sys.exit(1)

prot_dicts = []
flag = True
while flag == True:
    new_prot = read_record(file_,num_evo_entries)
    if new_prot != None:
        prot_dicts.append(new_prot)    
    else:
        flag = False  

# Add angle data
for i in range(len(prot_dicts)):
    residue_coords = prot_dicts[i]["tertiary"] 
    residue_coords = np.transpose(residue_coords)
    dihedrals = np.zeros((len(residue_coords),1))
    for j in range(1,len(residue_coords)-3):
        p = residue_coords[j-1:j-1+4] # Dihedral angle needs previous atom for computation
        dihedral = new_dihedral(p)
        dihedrals[j] = dihedral
    dihedrals = np.reshape(dihedrals,(-1,3))
    prot_dicts[i]["dihedrals"] = dihedrals

# Omega distribution
dihedral_distribution = np.zeros(0)
for i in range(len(prot_dicts)):
    dihedrals = prot_dicts[i]["dihedrals"][:,2]
    mask = prot_dicts[i]["mask"]
    dihedrals_masked = ma.masked_array(dihedrals,mask)
    dihedral_distribution = np.append(dihedral_distribution,dihedrals_masked)
dihedral_distribution = dihedral_distribution[~np.isnan(dihedral_distribution)]

# Broken axis plot
f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
x = np.arange(0,len(dihedral_distribution))
y = abs(dihedral_distribution)
ax.hist(y, bins="auto", color="darkblue")
ax2.hist(y, bins="auto", color="darkblue")
ax.set_ylabel("Count")
ax.set_xlabel("Omega angle [degrees]")
ax.set_xlim(0,10)
ax2.set_xlim(160,180)
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright='off')
ax2.yaxis.tick_right()
d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d,1+d), (-d,+d), **kwargs)
ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((-d,+d), (-d,+d), **kwargs)
plt.savefig("Omega_distribution")
plt.show()

# Do padding
numprots = len(prot_dicts)

seqLengths = np.empty(numprots,dtype="int32")
for i in range(numprots):
    seqLengths[i] = len(prot_dicts[i]["primary"])

maxseqlen = max(seqLengths)

primary_padded = np.zeros((numprots,maxseqlen))
evolutionary_padded = np.zeros((numprots,maxseqlen,21))
tertiary_padded = np.zeros((numprots,maxseqlen*3,3))
dihedrals_padded = np.zeros((numprots,maxseqlen,3))
mask_padded = np.zeros((numprots,maxseqlen))

for i in range(len(seqLengths)):
    start = 0
    stop = seqLengths[i]
    prot = prot_dicts[i]
    
    primary_padded[i,start:stop] = np.asarray(prot["primary"])[start:stop]
    evolutionary_padded[i,start:stop,:] = np.transpose(np.asarray(prot["evolutionary"])[:,start:stop])
    tertiary_padded[i,start:stop*3,:] = np.transpose(np.asarray(prot["tertiary"])[:,start:stop*3])
    dihedrals_padded[i,start:stop,:] = np.asarray(prot["dihedrals"])[start:stop]
    mask_padded[i,start:stop] = np.asarray(prot["mask"])[start:stop]

tertiary_padded = np.reshape(tertiary_padded,((numprots,maxseqlen,3,3)))

# Save data in NumPy format
np.save("primary_test",primary_padded)
np.save("evolutionary_test",evolutionary_padded)
np.save("tertiary_test",tertiary_padded)
np.save("dihedrals_test",dihedrals_padded)
np.save("mask_test",mask_padded)