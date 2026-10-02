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

#OPPGAVE H

#Disse 2 bruker samme farger, tror ikke det er super viktig men bare noterer det ned:

"""Studying varying learning rate"""
labels_gamma = [r"$0.01\, \gamma_{\max}$", r"$0.1\, \gamma_{\max}$", r"$0.5\, \gamma_{\max}$"]
colors = ["#F433DA", "#7326E6", "#84BCED"]

"""Studying varying batch sizes"""
labels_M = lambda batches: [f"$M = {batches[0]}$", f"$M = {batches[1]}$", f"$M = {batches[2]}$"]
colors = ["#F433DA", "#7326E6", "#84BCED"]


"""
"#F433DA", magenta 
"#7326E6", violet
"#84BCED" light blue/pale sky blue

"""