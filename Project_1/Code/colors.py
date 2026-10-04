"""
FARGEKODER:
"""
#OVER HELE PROSJEKTET/FLERE OPG

OLS = '#F433DA'	#magenta/hotpink
OLS2 = "#801772"
RIDGE = '#7326E6'	#violet/purple
RIDGE2 = "#361369"	
LASSO ="#EB3568"
LASSO2 = "#8D1336"

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

#OPPGAVE F OG H
PLAIN    = "#777777"
MOMENTUM = "#004488"
ADAGRAD  = "#DDAA33"
RMSPROP  = "#228833"
ADAM     = "#BB5566"

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


"""
"#F433DA", magenta 
"#7326E6", violet
"#84BCED" light blue/pale sky blue

"""
""" For when larger ranges of colors are needed """
broad_range = ["#CA2626", "#CA6826", "#DFCA13", "#87DF13", "#0DB832",
          "#1696D1", "#3216D1", "#8C16D1", "#D11699"]
#OPPGAVE H
"""Studying varying learning rate (light -> dark = increasing gamma)"""
LR_LOW  = '#AFC6F0'   # pale periwinkle
LR_MID  = '#5C85D6'   # medium blue
LR_HIGH = '#1B3B80'   # deep navy-blue

"""Studying varying batch sizes (light -> dark = increasing M)"""
BATCH_SMALL = '#F2B6C0'   # pale rose
BATCH_MED   = '#D9607A'   # medium rose/coral
BATCH_LARGE = '#8A1F3D'   # deep wine
