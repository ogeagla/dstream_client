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
    def display_all(grids, class_keys, ref_data, partitions_per_dimension, domains_per_dimension, plot_name='dstream', save_dir=None, plot_count=None):
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
            #print ref_data.shape, plot_count
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
        if save_dir != None:
            plt.savefig(save_dir + '/dstream' + '_' + plot_name + plot_info + '.png', bbox_inches = 0)
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
    
def run_clusterer(x, y, out_dir, total_plots = 10, needs_scale=True, partitions=(10, 10), c_m = 3.0, c_l = 0.8, beta = 0.3, decay = .998):
    '''
    dense_threshold_parameter = 3.0,#3.0, #C_m
                 sparse_threshold_parameter = 0.8,#0.8,  #C_l
                 sporadic_threshold_parameter = 0.3,#0.3, #beta
                 decay_factor = 0.998,#0.998, #lambda
                 dimensions = 2, 
                 domains_per_dimension = ((0.0, 100.0), (0.0, 100.0)),
                 partitions_per_dimension = (5, 5),
                 initial_cluster_count = 4,
                 seed = 331
    '''    
    #c_m = 3.0
    #c_l = 0.8
    #beta = 0.3
    #decay = 0.998
    dims = 2
    domains = ((0.0, 100.0), (0.0, 100.0))
    #partitions = (10, 10)
    cluster_count_init = 4
    seed = 331
    
    d_stream_clusterer = DStreamClusterer(c_m, c_l, beta, decay, dims, domains, partitions, cluster_count_init, seed)
    
    if needs_scale:
        
        x_scalar = 100.0/np.max(x)
        y_scalar = 100.0/np.max(y)
        print 'scaling by {} {}'.format(x_scalar, y_scalar)
        x = Utils.scale_vector(x, x_scalar, False)
        y = Utils.scale_vector(y, y_scalar,  False)
    Utils.show_scatter(x, y, 'all presampled data')
                
    data_size = x.size
    plot_count = 1
    #print data_size, total_plots
    display_times = data_size/total_plots
    has_not_clustered = True
    
    for i in range(data_size):
        #print 'data {} of total {}'.format(i+1, data_size)
        #print x[i], y[i]
        if x[i] <= 100.0 and x[i] >= 0.0 and y[i] <= 100.0 and y[i] >= 0.0:
            #print 'adding ', x[i], y[i]
            d_stream_clusterer.add_datum((x[i], y[i]))
        
        if has_not_clustered == True:
            if d_stream_clusterer.has_clustered_once == True:
                print 'clustering intialized!'
                has_not_clustered = False
                ClusterDisplay2D.display_all(d_stream_clusterer.grids, d_stream_clusterer.class_keys, d_stream_clusterer.data, d_stream_clusterer.partitions_per_dimension, d_stream_clusterer.domains_per_dimension, 'initial clusters', out_dir)
    
        if np.mod(i+1, display_times) == 0:
            #print 'time to plot'
            ClusterDisplay2D.display_all(d_stream_clusterer.grids, d_stream_clusterer.class_keys, d_stream_clusterer.data, d_stream_clusterer.partitions_per_dimension, d_stream_clusterer.domains_per_dimension, 'streaming', out_dir, (plot_count, total_plots))
            print i, '/', data_size
            plot_count += 1
            
    ClusterDisplay2D.display_all(d_stream_clusterer.grids, d_stream_clusterer.class_keys, d_stream_clusterer.data, d_stream_clusterer.partitions_per_dimension, d_stream_clusterer.domains_per_dimension, 'final clusters', out_dir)
    
    
    
def run_emulated_data(out_dir):
    metricDataMatrix, columnNamesVector, timesVector = Utils.get_emu_data()
    

def run_boa2_data(out_dir):
    allMEIds, perMEData, perMEMetricNames, perMETimes = Utils.get_boa2_data()
    MEId_of_interest = '536282'
    
    run_index = allMEIds.index(MEId_of_interest)
    run_data = perMEData[run_index]
    run_names = perMEMetricNames[run_index]
    times = perMETimes[run_index]
    
    metric_index = 4
    load_index = 29
    #print run_data.shape, run_data
    #print run_names.shape, run_names
    
    metric_data, load_data = run_data[:,metric_index], run_data[:,load_index]
    
    metric_name, load_name = run_names[metric_index], run_names[load_index]    
    
    load_data, metric_data, times = Utils.sanitize_data(load_data, metric_data, times)
    
    print 'size of {} is {}, size of {} is {} (load)'.format(metric_name, metric_data.size, load_name, load_data.size)
    
    #Utils.show_scatter(load_data, metric_data, 'boa2 data')
    
    run_clusterer(load_data, metric_data, out_dir, 10, True, (12, 12))#,c_m = 1.5, c_l = 0.4, beta = 0.8, decay = .998)
    
def run_test_data(out_dir):
    means_count = 3
    test_data_size = 20000
    
    
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

    #Utils.show_scatter(cluster_test_data_exp[:, 0], cluster_test_data_exp[:, 1], 'all presampled data')
    run_clusterer(cluster_test_data_exp[:, 0], cluster_test_data_exp[:,1], out_dir, 24, False, partitions_per_domain)
    
if __name__ == "__main__":
    
    #raw_input("press enter to go")
    
    test_dir = '../figs/test'
    emu_dir = '../figs/emu'
    boa2_dir = '../figs/boa2'
    out_dirs = [test_dir, emu_dir, boa2_dir]
    
    run_index = 2
    run_dir = out_dirs[run_index]
        
        
    #run_test_data(run_dir)
    {0:run_test_data, 1:run_emulated_data, 2:run_boa2_data}[run_index](run_dir)
    
    cp_str = 'cp {}/dstream_streaming* {}/anim'.format(run_dir, run_dir)
    subprocess.call(cp_str, shell=True)    
    pngs_str = '{}/anim/*.png'.format(run_dir)
    gif_str = '{}/anim/anim_py.gif'.format(run_dir)
    subprocess.call(["convert", "-delay", "100", pngs_str, gif_str])
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
    
