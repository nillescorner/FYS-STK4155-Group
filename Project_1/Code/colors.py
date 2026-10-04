"""
FARGEKODER:
"""
#OVER HELE PROSJEKTET/FLERE OPG

OLS = '#F433DA'	#magenta/hotpink
RIDGE = '#7326E6'	#violet/purple
LASSO = "#8D1336"

TRAIN_DATA = '#8C564B'  #brown 
TEST_DATA = '#0AC23E'	#bright green	
BEST = 'grey'; ls='--' #(DEGREE, LAMBDA, ANYTHING AS XVLINE):

NOISE_FLOOR = 'black'; ls=':'

ERROR = '#FF2B2B'	#bright red
BIAS = '#4E20A1'	#dark indigo (similar to ridge but its ok)
VARIANCE = '#FFB107'	#amber/golden yellow

#OPPGAVE D:
CROSS_VAL_K5 = '#0F2D9A'	#dark navy/royal blue
CROSS_VAL_K10 = '#FB7100'	#bright orange (close to variance amber but not in same fig)

#OPPGAVE E
GAMMA_MAX = '#E62663'	#raspberry / crimson
GAMMA_BEST = '#26B3E6'	#sky blue / cyan

#OPPGAVE G
#Disse er kanskje for like??
LASSO_GD = '#5DA0B6'			#muted teal / steel blue
LASSE_COORDINATE_DESCENT = '#65AF60'	#muted green /sage

THETA_COLORS = [
    '#E8546B',  #coral red
    '#4C9F70',  #emerald green
    '#5B7FDE',  #periwinkle blue
    '#E8A33D',  #gold
    '#9B59B6',  #purple
    '#2FB8AF',  #teal
    '#E67E22',  #burnt orange
    '#34495E',  #slate
]

#OPPGAVE H
"""Studying varying learning rate (light -> dark = increasing gamma)"""
LR_LOW  = '#AFC6F0'   # pale periwinkle
LR_MID  = '#5C85D6'   # medium blue
LR_HIGH = '#1B3B80'   # deep navy-blue

"""Studying varying batch sizes (light -> dark = increasing M)"""
BATCH_SMALL = '#F2B6C0'   # pale rose
BATCH_MED   = '#D9607A'   # medium rose/coral
BATCH_LARGE = '#8A1F3D'   # deep wine
