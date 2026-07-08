import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os

psevdo_dict = {
    'original'    : ["original"],
    'mitochondria': ['mitochondria', 'mitohondrion'],
    "vesicles"    : ['vesicles'],
    'boundaries'  : ['boundaries', 'membranes'],
    'background'  : ['background'],
    'axon'        : ['axon'],
    'PSD'         : ["PSD", "psd"],
}

def get_val_by_psevdo(d, name):
    for search_name in psevdo_dict[name]:
        if search_name in d.keys():
            return search_name

    raise ValueError(f"Name {name} no found in {d}")

def readTensor(path, name):
    g = glob.glob(path+ '/**/' + name, recursive=True)
    # print(g)
    d = {}
    for f in g:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        f= f.replace(path, '')
        f= f.replace(name, '')
        f= f.replace('//', '')
        f = f.replace('\\', '')
        d[f] = img
    return d

def getHist(maskname, d):
    temp = np.copy(d[get_val_by_psevdo(d, 'original')])
    temp = temp.astype(int)
    if not np.any(d[get_val_by_psevdo(d, maskname)] > 0):
        return np.zeros(256), np.arange(257)
    temp[d[get_val_by_psevdo(d, maskname)] == 0] = -1
    return np.histogram(temp.ravel(), bins=256, range=(0.0, 255.0), density=True)

def calcAreas(path, name):
    print('calc', path, name)
    d = readTensor(path, name)
    print(d.keys())
    background = 255 - (d[get_val_by_psevdo(d,'vesicles')] + d[get_val_by_psevdo(d, 'mitochondria')] + d[get_val_by_psevdo(d, 'boundaries')] + d[get_val_by_psevdo(d, 'axon')] + d[get_val_by_psevdo(d, 'PSD')])
    d['background'] = background
    num = background.shape[0] * background.shape[1]
    # masks have white color 255
    vesicles = np.sum(get_val_by_psevdo(d, 'vesicles')) * 100 / (255 * num)
    mitochondria = np.sum(get_val_by_psevdo(d, 'mitochondria')) * 100 / (255 * num)
    axon = np.sum(get_val_by_psevdo(d, 'axon')) * 100 / (255 * num)
    PSD = np.sum(get_val_by_psevdo(d, 'PSD')) * 100 / (255 * num)
    boundaries = np.sum(get_val_by_psevdo(d, 'boundaries')) * 100 / (255 * num)
    ground = np.sum(get_val_by_psevdo(d, 'background')) * 100 / (255 * num)

    print(num, vesicles, mitochondria, boundaries, ground)

    return vesicles, mitochondria, boundaries, axon, PSD, ground

def calcSlice(path, name):
    print('calc', path, name)
    d = readTensor(path, name)
    print(d.keys())
    background = 255 - (d[get_val_by_psevdo(d,'vesicles')] + d[get_val_by_psevdo(d, 'mitochondria')] + d[get_val_by_psevdo(d, 'boundaries')] + d[get_val_by_psevdo(d, 'axon')])
    d['background'] = background

    vesicles, bin_edges = getHist('vesicles', d)
    mitochondria, bin_edges = getHist('mitochondria', d)
    axon, bin_edges = getHist('axon', d)
    PSD, bin_edges = getHist('PSD', d)
    boundaries, bin_edges = getHist('boundaries', d)
    ground, bin_edges = getHist('background', d)

    return bin_edges, vesicles, mitochondria, boundaries, axon, PSD, ground

def printPlot(title, bin_edges, vesicles, mitochondria, boundaries, axon, PSD, ground, view_data=False, save_path=None):
    plt.rcParams['figure.figsize'] = (12, 3)
    plt.rcParams.update({'font.size': 15})
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.82)

    plt.text(6, 0.03, title, bbox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}, fontsize=15)
    #plt.title(title)
    #plt.xlabel("grayscale value")
    plt.ylabel("density")
    plt.xlim([0.0, 255.0])
    plt.ylim([0.0, 0.04])


    plt.plot(bin_edges[0:-1], ground, 'g', label = 'background')
    plt.plot(bin_edges[0:-1], vesicles, 'r', label = 'vesicles')
    plt.plot(bin_edges[0:-1], mitochondria, 'm', label = 'mitochondria')
    plt.plot(bin_edges[0:-1], axon, 'm', label = 'axon')
    plt.plot(bin_edges[0:-1], PSD, 'm', label = 'PSD')
    plt.plot(bin_edges[0:-1], boundaries, 'k', label = 'boundaries')
    plt.legend()

    separate = False
    if save_path:
        plt.savefig(f"{os.path.join(save_path, title)}_all.png")
    if view_data:
        plt.show()
    else:
        plt.close()

    if separate:
        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], ground, 'g', label = 'background')
        plt.legend()
        if save_path:
            plt.savefig(f"{os.path.join(save_path, title)}_background.png")
        if view_data:
            plt.show()
        else:
            plt.close()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], vesicles, 'r', label = 'vesicles')
        plt.legend()
        if save_path:
            plt.savefig(f"{os.path.join(save_path, title)}_vesicles.png")
        if view_data:
            plt.show()
        else:
            plt.close()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], mitochondria, 'm', label = 'mitochondria')
        plt.legend()
        if save_path:
            plt.savefig(f"{os.path.join(save_path, title)}_mitochondria.png")
        if view_data:
            plt.show()
        else:
            plt.close()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], axon, 'm', label = 'axon')
        plt.legend()
        if save_path:
            plt.savefig(f"{os.path.join(save_path, title)}_axon.png")
        if view_data:
            plt.show()
        else:
            plt.close()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], PSD, 'm', label = 'PSD')
        plt.legend()
        if save_path:
            plt.savefig(f"{os.path.join(save_path, title)}_PSD.png")
        if view_data:
            plt.show()
        else:
            plt.close()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], boundaries, 'k', label = 'membranes')
        plt.legend()
        if save_path:
            plt.savefig(f"{os.path.join(save_path, title)}_membranes.png")
        if view_data:
            plt.show()
        else:
            plt.close()

def printTwoPlot(title, bin_edges, original, synthetic, view_data=False, save_path = None):
    plt.title(title)
    plt.xlabel("grayscale value")
    plt.ylabel("density")
    plt.xlim([0.0, 255.0])
    plt.ylim([0.0, 0.05])

    plt.plot(bin_edges[0:-1], original, 'g', label = 'original')
    plt.plot(bin_edges[0:-1], synthetic, 'r', label = 'synthetic')
    plt.legend()
    if save_path:
        plt.savefig(f"{os.path.join(save_path, title)}.png")
    if view_data:
        plt.show()
    else:
        plt.close()

def printTreePlot(title, bin_edges, original, original2, synthetic, save_path = None):
    plt.rcParams['figure.figsize'] = (12, 3)
    plt.rcParams.update({'font.size': 15})
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.82)

    #plt.title(title)
    plt.text(6, 0.03, title, bbox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}, fontsize=15)
    #plt.xlabel("grayscale value")
    plt.ylabel("density")
    plt.xlim([0.0, 255.0])
    plt.ylim([0.0, 0.04])

    plt.plot(bin_edges[0:-1], original, 'g', label = 'originals all', linewidth = 3, alpha=0.5)
    plt.plot(bin_edges[0:-1], original2, 'r', label = 'original 0', linewidth = 2, alpha=0.5)
    plt.plot(bin_edges[0:-1], synthetic, 'b', label = 'synthetic', linewidth = 2, alpha=0.5)
    plt.legend()

    if save_path:
        if os.path.isdir(os.path.join(save_path, '3 print color')) is False:
            print(f"создаю {os.path.join(save_path, '3 print color')}")
            os.makedirs(os.path.join(save_path, '3 print color'))
        plt.savefig(os.path.join(save_path, '3 print color', title + '_.png'), dpi=600)
    else:
        plt.show()



def run_check_statistic_syn_dataset(syn_dataset_path, view_data=False):
    etal_path = r"D:/Projects/UnetClass/pytorch3D/segmentation/data/original data/training"
    go = glob.glob(etal_path +"//original//*.png")

    sumvesicles = np.zeros(256)
    summitochondria = np.zeros(256)
    sumaxon = np.zeros(256)
    sumPSD = np.zeros(256)
    sumboundaries = np.zeros(256)
    sumground = np.zeros(256)

    for f in go:
        f = f.split('\\')[-1]
        bin_edges, vesicles, mitochondria, boundaries, axon, PSD, ground = calcSlice(etal_path, f)
        sumvesicles = sumvesicles + vesicles
        summitochondria = summitochondria + mitochondria
        sumaxon = sumaxon + axon
        sumPSD = sumPSD + PSD
        sumboundaries = sumboundaries + boundaries
        sumground = sumground + ground

    o_vesicles = sumvesicles / len(go)
    o_mitochondria = summitochondria / len(go)
    o_axon = sumaxon / len(go)
    o_PSD = sumPSD / len(go)
    o_boundaries = sumboundaries / len(go)
    o_ground = sumground / len(go)


    #printPlot('Original all layers',bin_edges, o_vesicles, o_mitochondria, o_boundaries, o_axon, o_PSD, o_ground)
    bin_edges2, vesicles2, mitochondria2, boundaries2, axon2, PSD2, ground2 = calcSlice(etal_path, "training0000.png")
    #printPlot('Original 0 layer',bin_edges2, vesicles2, mitochondria2, boundaries2, axon2, PSD2, ground2)

    g = glob.glob(syn_dataset_path +"//original//*.png")

    sumvesicles = np.zeros(256)
    summitochondria = np.zeros(256)
    sumaxon = np.zeros(256)
    sumPSD = np.zeros(256)
    sumboundaries = np.zeros(256)
    sumground = np.zeros(256)


    for f in g:
        f = f.split('\\')[-1]
        bin_edges, vesicles, mitochondria, boundaries, axon, PSD, ground = calcSlice(syn_dataset_path, f)
        sumvesicles = sumvesicles + vesicles
        summitochondria = summitochondria + mitochondria
        sumaxon = sumaxon + axon
        sumPSD = sumPSD + PSD
        sumboundaries = sumboundaries + boundaries
        sumground = sumground + ground

    sumvesicles = sumvesicles / len(g)
    summitochondria = summitochondria / len(g)
    sumaxon = sumaxon / len(g)
    sumPSD = sumPSD / len(g)
    sumboundaries = sumboundaries / len(g)
    sumground = sumground / len(g)

    printPlot('Synthetic', bin_edges, sumvesicles, summitochondria, sumboundaries, sumaxon, sumPSD, sumground, view_data=view_data, save_path=syn_dataset_path)

    printTwoPlot('vesicles', bin_edges, o_vesicles, sumvesicles, view_data=view_data, save_path=syn_dataset_path)
    printTwoPlot('mitochondria', bin_edges, o_mitochondria, summitochondria, view_data=view_data, save_path=syn_dataset_path)
    printTwoPlot('axon', bin_edges, o_axon, sumaxon, view_data=view_data, save_path=syn_dataset_path)
    printTwoPlot('PSD', bin_edges, o_PSD, sumPSD, view_data=view_data, save_path=syn_dataset_path)
    printTwoPlot('membranes', bin_edges, o_boundaries, sumboundaries, view_data=view_data, save_path=syn_dataset_path)
    printTwoPlot('ground', bin_edges, o_ground, sumground, view_data=view_data, save_path=syn_dataset_path)

    #printTreePlot('Vesicles', bin_edges, o_vesicles, vesicles2, sumvesicles, syn_dataset_path)
    #printTreePlot('Mitochondria', bin_edges, o_mitochondria, mitochondria2, summitochondria, syn_dataset_path)
    #printTreePlot('Axon', bin_edges, o_axon, axon2, sumaxon, syn_dataset_path)
    #printTreePlot('PSD', bin_edges, o_PSD, PSD2, sumPSD, syn_dataset_path)
    #printTreePlot('Membranes', bin_edges, o_boundaries, boundaries2, sumboundaries, syn_dataset_path)
    #printTreePlot('Ground', bin_edges, o_ground, ground2, sumground, syn_dataset_path)

if __name__ == "__main__":
    syn_dataset_path = r"D:/Projects/Synthetics/Synthetic3D/datasets/dataset_2026_07_04__00_30_29"
    run_check_statistic_syn_dataset(syn_dataset_path, view_data=True)
