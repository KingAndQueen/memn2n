# import matplotlib.pyplot as plt
# import numpy as np
# from string import ascii_letters
# Libraries
import matplotlib
import pdb
# import pandas as pd
# import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_line():
    #draw the loss of replace
    # train = [0.79,0.46,0.36,0.30,0.27,0.25,0.23,0.23,0.22]
    train = [0.79, 0.36, 0.27, 0.23, 0.22]
    train = [1286, 882, 547, 355, 273]
    train = [1212, 400, 99, 29, 18]
    train = [305, 133, 61, 40, 33]
    train = [144, 23, 9,6, 5]
    train = [630, 535, 409, 291, 232]
    train = [297, 74, 39, 28, 23]
    train = [285, 76, 45, 33, 30]
    train = [442, 134, 34, 15, 10]
    train = [382, 75, 25, 12, 9.5]
    train = [163, 29, 9.7, 5.9, 4.8]
    train = [1.5, 0.4, 0.3, 0.25, 0.23]
    train = [146, 31, 10, 6.3, 5.2]
    train = [126, 2, 0.85, 0.64, 0.56]
    train = [2151, 1941, 1718, 1594, 1584]
    train = [1.07, 0.55, 0.42, 0.37, 0.35]
    # rename:
    train = [0.57, 0.30, 0.23, 0.20, 0.19]
    train = [884, 178, 12, 4.64, 3.0]
    train = [1155, 398, 120, 48, 34]
    train = [621, 47, 11, 3.2, 2]
    train = [493, 196, 68, 34, 25]
    train = [378, 104, 47, 32, 27]
    train = [386, 119, 47, 28, 23]
    train = [226, 60, 31, 23, 17.9]
    train = [0.84, 0.34, 0.25, 0.22, 0.2]
    train = [122, 61, 40, 28, 23]

    # suftrain=[0.47,0.29,0.21,0.17,0.13,0.11,0.10,0.08,0.08]
    suftrain = [0.47, 0.21, 0.13, 0.10, 0.08]
    suftrain = [462, 253, 99, 5, 2]
    suftrain = [608, 322, 271, 309, 91]
    suftrain = [40, 2, 1, 0.4, 0.3]
    suftrain = [3.3, 1.5, 0.9, 0.6, 0.5]
    suftrain = [451, 370, 287, 248, 204]
    suftrain = [82, 15, 8.7, 10, 4.9]
    suftrain = [97, 97, 38, 36, 43]
    suftrain = [69, 1.4, 0.6, 0.4, 0.3]
    suftrain = [175, 66, 37, 2.6, 1.25]
    suftrain = [75, 32, 2.3, 1.1, 0.75]
    suftrain = [0.46, 0.21, 0.13, 0.09, 0.07]
    suftrain = [100, 70, 22, 3.5, 1.6]
    suftrain = [87, 2.9, 0.6, 0.36, 0.23]
    suftrain = [2182, 2123, 2061, 2050, 2055]
    suftrain = [0.68, 0.34, 0.22, 0.17, 0.13]
    # rename:
    suftrain = [0.35, 0.17, 0.11, 0.08, 0.06]
    suftrain = [143, 78.6, 1.12, 0.39, 0.24]
    suftrain = [651, 503, 402, 375, 425]
    suftrain = [1.4, 0.44, 0.24, 0.16, 0.12]
    suftrain = [218, 137, 172, 66, 13]
    suftrain = [97, 60, 35, 15, 49]
    suftrain = [25, 1.1, 0.5, 0.38, 0.28]
    suftrain = [160, 87, 13, 45, 5.9]
    suftrain = [0.35, 0.16, 0.1, 0.08, 0.06]
    suftrain = [98, 68, 52, 53, 37]

    x1 = range(20, 101, 20)
    x2=range(20,101,20)



    plt.figure(figsize=(5.1, 4))
    plt.xticks(x1)
    plt.plot(x1, train, "r", linewidth=1, label='Train')
    plt.plot(x2, suftrain, "b--", linewidth=1, label='Suf-train')
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.title("Task 13")
    plt.legend()
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

def drew_embedding(final_embeddings,reverse_dictionary,name=None):
    def plot_with_labels(low_dim_embs, labels, filename=name+'.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        # zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', size=5)
        plt.figure(figsize=(38, 38))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y, 100,color='black')
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(15, 12),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')#, fontproperties=zhfont)

        plt.savefig(filename)


    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        reverse_dictionary[0]='<pad>'
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        # plot_only = 200
        low_dim_embs = tsne.fit_transform(final_embeddings)
        # pdb.set_trace()
        labels = reverse_dictionary.values()
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn and matplotlib to visualize embeddings.")

if __name__=='__main__':
    embedding=[[1,0,2,1,0,0,1,0.4]]
    idx_word={0:'test',1:'test1'}
    drew_embedding(embedding,idx_word)
    # draw_line()
    # draw_relation([ 1.40497342E-01	,	1.3055712E-01],[ 1.40497342E-01	,	1.3055712E-01])
    # draw_eval()