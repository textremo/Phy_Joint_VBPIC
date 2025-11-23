import numpy as np
from numpy import arange, ones, zeros, eye, kron, reshape, einsum, sqrt, exp, conj
from numpy import tile as repmat
from numpy.random import randn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import *;
from textremo_phy_mod_otfs import OTFS, OTFSResGrid

from config.genconfig import genconfig

genconfig("OTFS", "EMBED", "toy-p1")