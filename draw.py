# import matplotlib.pyplot as plt
# import numpy as np
# from string import ascii_letters
# Libraries
import matplotlib

# import pandas as pd
# import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def draw_line():
    #draw the distribution of roles in scenes
    train = [1.44,0.53,0.35,0.29,0.24,0.23,0.21,0.20,0.19,0.19]
    x1=range(10)
    suftrain=[0.84,0.40,0.27]
    x2=range(3)

    plt.figure(figsize=(8, 4))
    plt.plot(x1, train, "r", linewidth=1)
    plt.plot(x2, suftrain, "b--", linewidth=1)
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("Line plot")
    plt.show()

    # plt.plot(values)
    # plt.xlabel("Role number in a scene")
    # plt.ylabel("Scene number")
  #  plt.savefig("role_distribution.png")
  #   plt.show()

   # draw the sentences count of roles
   #  height = [9295,9203,8517,8428,8354,7538,361,254,217,200]
   #  bars = ('Rachel','Rose','Chandler','Monica','Joey','Phoebe','Mike','Richard','Janice')
   #  y_pos = np.arange(0,1*len(bars),1)
   #
   #  plt.bar(y_pos, height,color=(0.2, 0.4, 0.6, 0.6))
   #  plt.xticks(y_pos, bars)
   #  fig = matplotlib.pyplot.gcf()
   #  fig.set_size_inches(8.5, 7.5)
   #
   #  plt.xlabel("Role name ranged by speaking times")
   #  plt.ylabel("Speaking times in Friends 10 seasons")
   # plt.savefig("sentences_count.png",dpi=200)
    plt.show()



def draw_relation(data_test,data_iu):

    df = pd.DataFrame(data_test)
    df2 = pd.DataFrame(data_iu)
    #Default heatmap: just a visualization of this square matrix
    with plt.rc_context(dict(sns.axes_style("whitegrid"),
                             **sns.plotting_context("notebook", font_scale=1.1,rc={'axes.labelsize':4}))):
        p1 = sns.heatmap(df,cmap='Blues')
    plt.figure()
    with plt.rc_context(dict(sns.axes_style("whitegrid"),
                                 **sns.plotting_context("notebook", font_scale=1.1, rc={'axes.labelsize': 4}))):
        p2=sns.heatmap(df2,cmap='Blues')

    #Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2,
    # Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r,
    #  Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r,
    # PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r,
    # RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r,
    # Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr,
    # YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr,
    # bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth,
    # gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r,
    # gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r,
    # hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r,
    # ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r,
    # spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r,
    # terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

    plt.show()



def draw_eval():

    plt.style.use("ggplot")
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    df = pd.DataFrame()
    df["Score_seq2seq"] = [2,]

    df["Score_MIDS"] = [3,]
    df["Turns_seq2seq"] = [4,]

    df["Turns_MIDS"] = [5,]

    sns.boxplot(df["Score_seq2seq"],color='skyblue')
    # sns.boxplot(df["Score_MIDS"], color='red')
    # plt.boxplot(x=df.values, labels=df.columns, whis=1.5)
    plt.show()

if __name__=='__main__':
    draw_line()
    # draw_relation([ 1.40497342E-01	,	1.3055712E-01],[ 1.40497342E-01	,	1.3055712E-01])
    # draw_eval()