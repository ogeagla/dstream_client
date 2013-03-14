# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
'''
sys.path.append('libs/networkx-1.7-py2.7.egg')'''
import sys
sys.path.append('/home/octavian/github/dstream/src')
sys.path.append('/home/octavian/github/utils/src')
from utils import Utils
from dstream import DStreamClusterer
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ClusterDisplay2D():


    @staticmethod
    def display_all(grids, class_keys, ref_data, partitions_per_dimension, domains_per_dimension, plot_name='dstream', save=False, plot_count=None):
        class_key_colors = {}
        color_map = cm.get_cmap('hsv') 
        for i in range(class_keys.size):
            class_key_colors[class_keys[i]] = color_map(np.float(i)/np.float(class_keys.size))
        #scat = ax.scatter(xPlot,yPlot,s=area, marker='o', c='y', linewidths=0.1, label='metric data')
        
        x_domain = domains_per_dimension[0]
        x_domain_size = x_domain[1] - x_domain[0]
        x_partitions = partitions_per_dimension[0]
        x_cluster_width = x_domain_size/x_partitions
        
        y_domain = domains_per_dimension[1]
        y_domain_size = y_domain[1] - y_domain[0]
        y_partitions = partitions_per_dimension[1]
        y_cluster_width = y_domain_size/y_partitions
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if plot_count is not None:
            plot_info = '_' +str(plot_count[0]).zfill(len(str(plot_count[1])))+ '_of_' + str(plot_count[1]) + '_t=' + str(ref_data[:,0].size)
        else:
            plot_info = '_t=' + str(ref_data[:,0].size)
        ax.set_title(plot_name+ plot_info)
        #print x_domain, y_domain, x_domain_size, y_domain_size, x_partitions, y_partitions
        x_ticks = np.arange(x_domain[0], x_domain[1]+ x_domain[1]/1000.0, x_cluster_width)
        y_ticks = np.arange(y_domain[0], y_domain[1] + y_domain[1]/1000.0 , y_cluster_width)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        for x_tick in x_ticks:
            ax.axvline(x=x_tick, c='b')
        for y_tick in y_ticks:
            ax.axhline(y=y_tick, c= 'b')
        '''for i in range(binCount + 1):
        line = ax.axvline(x=lineCounter, c='g')
        lineCounter += deltaBin'''
        
        
        ref_data_scatter = ax.scatter(ref_data[:, 0], ref_data[:, 1], marker = 'o', c = 'r', linewidths = 0.1, label = 'ref data', s=2.0)
        

            
        for indices, grid in grids.items():
            class_key = grid.label
            
            class_color = (0.8, 0.8, 0.8)   
            if class_key != None:
                if class_key_colors.has_key(class_key):
                    class_color = class_key_colors[class_key]
                
                    
                
            x = indices[0] * x_cluster_width + x_cluster_width * 0.5
            y = indices[1] * y_cluster_width + y_cluster_width * 0.5
            
            density_category = grid.density_category
            
            if density_category != None:
                mark = {'SPARSE':'^', 'TRANSITIONAL':'s', 'DENSE':'h'}[density_category]
            else:
                mark = 'x'
            
            ax.scatter(x, y, marker = mark, c = class_color, s=grid.density*10, linewidths = 0.1,label= ' ' + str(class_key))
        if save:
            plt.savefig('../figs/out/dstream' + '_' + plot_name + plot_info + '.png', bbox_inches = 0)
            #plt.savefig(filename + '.pdf', bbox_inches = 0)
        #leg = ax.legend(loc=2)
class NMeanSampler2D():
    
    def __init__(self, means, means_scales, x_domain, y_domain, seed, means_sample_times=None, noise_coeff = 0.0):
        
        means[:, 0] = means[:,0] * (x_domain[1] - x_domain[0])
        means[:, 1] = means[:,1] * (y_domain[1] - y_domain[0])
    
        means_scales[:, 0] = means_scales[:,0] * (x_domain[1] - x_domain[0])
        means_scales[:, 1] = means_scales[:,1] * (y_domain[1] - y_domain[0])        
        
        self.means = means
        self.means_scales = means_scales
        self.domains = (x_domain, y_domain)
        #print means.shape, means_scales.shape
        self.seed = seed
        np.random.seed(seed)
        
        if means_sample_times != None:
            
            self.means_sample_times = means_sample_times
            self.noise_coeff = noise_coeff
            
            self.current_mean_index = 0
            self.current_mean_count = 0
        
    
    def get_random_mean_index(self):
        rand_uni = np.random.uniform()
        mean_index = np.int(np.floor(rand_uni * self.means.shape[0]))
        return mean_index
    
    def get_sample(self):
        
        mean_index = self.get_random_mean_index()
        
        mean = self.means[mean_index, :]
        mean_scale = self.means_scales[mean_index, :]
        #print 'sampling from mean: ', mean
        x = np.random.normal(loc = mean[0], scale = mean_scale[0])
        y = np.random.normal(loc = mean[1], scale = mean_scale[1])
        
        return np.array([x, y])
        
    def get_noisy_time_dep_sample(self):
        
        rand_uni = np.random.uniform()
        
        if rand_uni <= self.noise_coeff:
            x, y = np.random.uniform()*(self.domains[0][1]-self.domains[0][0]), np.random.uniform()*(self.domains[1][1]-self.domains[1][0])
            #print 'noise ', x, y       
            return np.array([x, y])
        
            
        if self.current_mean_count >= self.means_sample_times[self.current_mean_index]:
            '''print 'means sample count ', self.means_sample_times
            print 'rotating means ', self.current_mean_count, self.current_mean_index
            print 'now: ', self.means, self.means_scales'''
            if self.current_mean_index == self.means_sample_times.size - 1:
                self.current_mean_index = 0
            else:
                self.current_mean_index += 1
            #print 'now ', self.current_mean_index
            self.current_mean_count = 0
        
        mean = self.means[self.current_mean_index, :]
        mean_scale = self.means_scales[self.current_mean_index, :]
        
        x = np.random.normal(loc = mean[0], scale = mean_scale[0])
        y = np.random.normal(loc = mean[1], scale = mean_scale[1])
                
        self.current_mean_count += 1
        #print x, y, ' about ', mean, mean_scale
        return np.array([x, y])
def pngs_to_gif(pngs_str, gif_str, delay=100):
    '''
    this doesnt work for some reason...equivalent cmd works in cl
    
    '''
    import subprocess
    
    params = []
    params += ["-delay", str(delay)]
    print 'calling ',["convert"] + params + [pngs_str,gif_str]
   
    subprocess.call(["convert"] + params + [pngs_str,gif_str], shell=True)    
    
def run_emulated_data():
    metricDataMatrix, columnNamesVector, timesVector = Utils.get_emu_data()
    

def run_boa2_data():
    allMEIds, perMEData, perMEMetricNames, perMETimes = Utils.get_boa2_data()
    MEId_of_interest = '536282'
    
def run_test():
    pass
    
if __name__ == "__main__":
    
    raw_input("press start to go")
    means_count = 3
    test_data_size = 20000
    display_times = 1
    
    x_domain = (0.0, 100.0)
    y_domain = (0.0, 100.0)
    partitions_per_domain = (10, 10)
    

    means = np.ndarray(shape=(means_count, 2))    
    means_scales = np.ndarray(shape=(means_count, 2))    
    
    means[0,0] = 0.16#x1, normalized
    means[0,1] = 0.2#y1 
    means_scales[0,0] = 0.05
    means_scales[0,1] = 0.08
        
    
    means[1,0] = 0.68
    means[1,1] = 0.29
    means_scales[1,0] = 0.03
    means_scales[1,1] = 0.06
    
    means[2,0] = 0.42
    means[2,1] = 0.63
    means_scales[2,0] = 0.06
    means_scales[2,1] = 0.05
    
    
    #nms2d = NMeanSampler2D(means, means_scales, x_domain, y_domain, 331)
    nms2d_exp = NMeanSampler2D(means, means_scales, x_domain, y_domain, 331, np.array([test_data_size/6., test_data_size/6., test_data_size/6.]), 0.15)
    #cluster_test_data = np.ndarray(shape=(test_data_size,2))
    cluster_test_data_exp = np.ndarray(shape=(test_data_size,2))
        
    for i in range(test_data_size):
        #datum = nms2d.get_sample()
        #cluster_test_data[i, :] = datum
        
        datum_exp = nms2d_exp.get_noisy_time_dep_sample()
        cluster_test_data_exp[i, :] = datum_exp
        
    #print cluster_test_data
    #ClusterDisplay2D.display_ref_data(cluster_test_data, par/tions_per_domain)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat = ax.scatter(cluster_test_data_exp[:, 0], cluster_test_data_exp[:, 1])
    plt.show()
    

    d_stream_clusterer = DStreamClusterer(3.0, 0.8, 0.3, 0.998, 2, (x_domain, y_domain), partitions_per_domain)
    
    
    plot_count = 1
    total_plots = 24
    display_times = test_data_size/total_plots#d_stream_clusterer.gap_time * 1000
    for i in range(test_data_size):
 
        x = cluster_test_data_exp[i, 0]
    
        y = cluster_test_data_exp[i, 1]
        
        if x <= 100.0 and x >= 0.0 and y <= 100.0 and y >= 0.0:
            d_stream_clusterer.add_datum((x, y))
        else:
            continue
        
        if np.mod(i, display_times) == 0 and i > 0:
            ClusterDisplay2D.display_all(d_stream_clusterer.grids, d_stream_clusterer.class_keys, d_stream_clusterer.data, d_stream_clusterer.partitions_per_dimension, d_stream_clusterer.domains_per_dimension, 'streaming', True, (plot_count, total_plots))
            print i, '/', test_data_size
            plot_count += 1
    ClusterDisplay2D.display_all(d_stream_clusterer.grids, d_stream_clusterer.class_keys, d_stream_clusterer.data, d_stream_clusterer.partitions_per_dimension, d_stream_clusterer.domains_per_dimension, 'final clusters', True)
    subprocess.call('cp ../figs/out/dstream_streaming* ../figs/anim', shell=True)    
    subprocess.call(["convert", "-delay", "100", "../figs/anim/*.png", "../figs/anim/anim_py.gif"])
    '''plt.show()'''
    
    
    
    #grids, class_keys, ref_data, partitions_per_dimension, domains_per_dimension):
    #ClusterDisplay2D.display_all(d_stream_clusterer.grids, d_stream_clusterer.class_keys, cluster_test_data, d_stream_clusterer.partitions_per_dimension, d_stream_clusterer.domains_per_dimension)
    
    '''den_mat = d_stream_clusterer.get_density_nmatrix(d_stream_clusterer.grids)
    per_cluster_id_den_mat = d_stream_clusterer.get_per_cluster_density_nmatrix_dict()
    
    myColorMap = cm.get_cmap('hot')    
    
    fig = plt.figure()
    im = plt.imshow(den_mat, cmap=myColorMap)
    plt.colorbar()
    
    for class_key, class_den_mat in per_cluster_id_den_mat.items():
        print 'class: ', class_key, 'grids: ', len(d_stream_clusterer.get_grids_of_cluster_class(class_key).keys())
        fig = plt.figure()
        im = plt.imshow(class_den_mat, cmap=myColorMap)
        plt.colorbar()
    
    plt.show()'''
    
