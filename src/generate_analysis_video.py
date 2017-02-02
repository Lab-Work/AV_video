import sys
import time
from collections import OrderedDict
from os.path import exists
import os
import sqlite3
import csv

import matplotlib.pyplot as plt
import numpy as np

"""
This script contains the code for generating a sequence of pictures for analyzing the performance of two estimators.

Each frame contains 3x2 figures. The left column is for 1st order model, and the 2nd column is for 2nd order model.
- Top: True density, estimated density and 5/95% variation in the 10 PF runs
- Middle: True fraction of AVs, estimated fraction of AVs and 5/95% variation in the 10 PF runs
- Bottom: The animation of microsimulation in AIMSUN.


To combine pictures to a video:
- install ffmpeg
- cd to the folder of pics
- run "ffmpeg -framerate 12 -pattern_type glob -i '*.png' -vf scale=1480:-2 -vcodec libx264 -pix_fmt yuv420p out.mp4"
"""

# configure some global constants
av_id = 436
car_id = 53

# idx in traj file
_veh_idx = 1
_lane_idx = 4
_time_idx = 7
_dist_idx = 9

# the discretization in the estimation models
dt = 5  # s
dx = 179    # m ~=1/9 mile
t_max = 3600.0  # s
x_max = 4833.0  # m
num_cells = 27
num_steps = 720

plot_traj = True

def main(argv):

    # sce_seeds = [(75, 3252), (75, 59230), (100, 24654), (100, 45234)]
    # sce_seeds = [(100, 24654)]
    sce_seeds =[(75, 3252)]
    for sce, seed in sce_seeds:

        # =================================================================
        # the trajectory file
        traj_file = '../Simulation/traj_data/sim_sce{0}_seed{1}.csv'.format(sce, seed)
        veh_type_file = '../Simulation/traj_data/sim_sce{0}_seed{1}_vehtype.csv'.format(sce, seed)

        load_dir = '../Video/Video_sce{0}_seed{1}_new_paras/'.format(sce, seed)
        est_density_prefix = load_dir + 'EstimationDensity_PR_{0}_Seed{1}_'.format(sce, seed)
        est_w_prefix = load_dir + 'EstimationW_PR_{0}_Seed{1}_'.format(sce, seed)
        true_density_file = load_dir + 'TrueDensity_PR_{0}_Seed{1}.npy'.format(sce, seed)
        true_w_file = load_dir + 'TrueW_PR_{0}_seed{1}.npy'.format(sce, seed)

        save_dir = load_dir + 'pics/'.format(sce, seed)

        if not exists(save_dir):
            os.mkdir(save_dir)

        # =================================================================
        # load all the data
        true_density = np.load(true_density_file)
        true_w = np.load(true_w_file)
        est_density_1st = []
        est_density_2nd = []
        est_w_2nd = []

        print('Loading estimation results...')
        for run in range(0, 10):
            est_density_1st.append(np.load(est_density_prefix+'1st_'+'{0}.npy'.format(run)))
            est_density_2nd.append(np.load(est_density_prefix+'2nd_'+'{0}.npy'.format(run)))
            est_w_2nd.append(np.load(est_w_prefix+'2nd_'+'{0}.npy'.format(run)))
        est_density_1st = np.asarray(est_density_1st)
        est_density_2nd = np.asarray(est_density_2nd)
        est_w_2nd = np.asarray(est_w_2nd)

        # get the mean of 10 runs
        mean_density_1st = np.mean(est_density_1st, 0)
        mean_density_2nd = np.mean(est_density_2nd, 0)
        mean_w_2nd = np.mean(est_w_2nd, 0)
        std_density_1st = np.std(est_density_1st, 0)
        std_density_2nd = np.std(est_density_2nd, 0)
        std_w_2nd = np.std(est_w_2nd, 0)

        print('Loaded estimation results with shape: 1st_rho {0}, 2nd_rho {1}, 2nd_2 {2}.\n'.format(est_density_1st.shape,
                                                                                                    est_density_2nd.shape,
                                                                                                    est_w_2nd.shape))
        # get the trajectory
        print('Loading vehicle trajectory...')
        veh_type = get_veh_type(veh_type_file)
        if plot_traj:
            snaps = get_snapshot(traj_file, veh_type, time_step=5.0)
        print('\nLoaded vehicle trajectory.\n')

        # =================================================================
        # for each time step, plot one figure and save the figure in save_dir
        # test_t = [15000]
        for t in sorted(snaps.keys()):
        # for t in test_t:
            # note, t is integer, unit 0.1s
            f, axarr = plt.subplots(3, 2, sharex=True, figsize=(18,10))

            # --------------------------------------------------
            # plot the density
            t_row = int(t/10.0/dt)
            x_grid = np.arange(0, dx*(num_cells+1), dx)

            # conf_intrvl = 2     # 5/95 for Gaussian
            conf_intrvl = 1.15  # 5/95 for Gaussian

            # first order model
            axarr[0,0].step(x_grid, np.concatenate([[0], mean_density_1st[t_row, :]]),
                            color='b', linewidth=2, label='1st')
            axarr[0,0].step(x_grid, np.concatenate([[0], mean_density_1st[t_row, :]-conf_intrvl*std_density_1st[t_row,:]]),
                            color='b', linestyle='--', linewidth=2)
            axarr[0,0].step(x_grid, np.concatenate([[0], mean_density_1st[t_row, :]+conf_intrvl*std_density_1st[t_row,:]]),
                            color='b', linestyle='--', linewidth=2)
            axarr[0,0].set_title('Estimated density (1st)', fontsize=20)

            # second order model
            axarr[0,1].step(x_grid, np.concatenate([[0], mean_density_2nd[t_row, :]]),
                            color='g', linewidth=2, label='2nd')
            axarr[0,1].step(x_grid, np.concatenate([[0], mean_density_2nd[t_row, :]-conf_intrvl*std_density_2nd[t_row,:]]),
                            color='g', linestyle='--', linewidth=2)
            axarr[0,1].step(x_grid, np.concatenate([[0], mean_density_2nd[t_row, :]+conf_intrvl*std_density_2nd[t_row,:]]),
                            color='g', linestyle='--', linewidth=2)
            axarr[0,1].set_title('Estimated density (2nd)', fontsize=20)

            for col in [0, 1]:
                axarr[0,col].step(x_grid, np.concatenate([[0], true_density[t_row, :]]),
                                  color='r', linewidth=2, label='true')

                axarr[0,col].set_ylabel('Density (veh/mile)')
                axarr[0,col].set_xlim([0, dx*num_cells])
                axarr[0,col].set_ylim([0, 850])
                axarr[0,col].set_xticks([0, dx*9, dx*18, dx*27])
                axarr[0,col].set_xticklabels(['0', '1', '2', '3'])
                axarr[0,col].legend(ncol=3)
                text_str = 'Time: {0} s'.format(int(t/10.0))
                axarr[0,col].annotate(text_str, xy=(0.05, 0.88), xycoords='axes fraction', fontsize=16)

            # --------------------------------------------------
            # plot the penetration rate

            # first order uses a constant value sce/2
            avg_w_1st = np.ones(mean_w_2nd.shape[1]).astype(float)*sce/2.0
            axarr[1,0].step(x_grid, np.concatenate([[0], avg_w_1st]), color='b', linewidth=2, label='1st')
            axarr[1,0].set_title('Fraction of AVs (1st)', fontsize=20)

            # second order
            axarr[1,1].step(x_grid, 100.0*np.concatenate([[0], mean_w_2nd[t_row, :]]),
                            color='g', linewidth=2, label='2nd')
            axarr[1,1].step(x_grid, 100.0*np.concatenate([[0], mean_w_2nd[t_row, :]-conf_intrvl*std_w_2nd[t_row,:]]),
                            color='g', linestyle='--', linewidth=2)
            axarr[1,1].step(x_grid, 100.0*np.concatenate([[0], mean_w_2nd[t_row, :]+conf_intrvl*std_w_2nd[t_row,:]]),
                            color='g', linestyle='--', linewidth=2)
            axarr[1,1].set_title('Fraction of AVs (2nd)', fontsize=20)

            # plot the true penetration rate
            # print('len x_grid: {0}'.format(x_grid))
            for col in [0,1]:

                true_w_row = true_w[t_row, :]
                # axarr[1, col].step(x_grid, true_w_100, color='r',
                #                    linewidth=2, label='true')

                # plot smoothed w: w is taken average every three cells
                true_w_smoothed = []
                for i in range(0, len(true_w_row)):
                    i_s = np.max([0, i-1])
                    i_e = np.min([i+2, len(true_w_row)])
                    true_w_smoothed.append(np.mean(true_w_row[i_s:i_e]))

                axarr[1, col].step(x_grid, 100*np.concatenate([[0], true_w_smoothed]), color='r',
                                   linewidth=2, label='true')

                axarr[1, col].set_ylim([0, 100])
                axarr[1, col].set_ylabel('AV fraction (%)')
                axarr[1, col].legend(ncol=2)
                text_str = 'Time: {0} s'.format(int(t/10.0))
                axarr[1, col].annotate(text_str, xy=(0.05, 0.88), xycoords='axes fraction', fontsize=16)

            # --------------------------------------------------
            # plot simulation
            # plot lanes
            for col in [0,1]:
                axarr[2, col].plot([0, dx*num_cells], [2, 2], 'k', linewidth=2)
                axarr[2, col].plot([0, dx*num_cells], [5, 5], color='k', linestyle='--', linewidth=2)
                axarr[2, col].plot([0, dx*num_cells], [8, 8], 'k', linewidth=2)

                # plot detectors
                # axarr[2].plot([5, 5], [0, 10], 'b', linewidth=3)
                # axarr[2].plot([dx*9, dx*9], [0, 10], 'b', linewidth=3)
                # axarr[2].plot([dx*18, dx*18], [0, 10], 'b', linewidth=3)
                # axarr[2].plot([dx*27-5, dx*27-5], [0, 10], 'b', linewidth=3)
                # axarr[2].annotate('Detectors', xy=(0.14, 0.015), xycoords='axes fraction')
                # axarr[2].annotate('Detectors', xy=(0.8, 0.015), xycoords='axes fraction')
                # axarr[2].annotate('', xy=(0.005, 0.05), xycoords='axes fraction',
                #                   xytext=(0.135, 0.05), textcoords='axes fraction',
                #                   arrowprops=dict(arrowstyle="->", connectionstyle='arc3'))
                # axarr[2].annotate('', xy=(0.33, 0.05), xycoords='axes fraction',
                #                   xytext=(0.203, 0.05), textcoords='axes fraction',
                #                   arrowprops=dict(arrowstyle="->", connectionstyle='arc3'))
                # axarr[2].annotate('', xy=(0.67, 0.05), xycoords='axes fraction',
                #                   xytext=(0.79, 0.05), textcoords='axes fraction',
                #                   arrowprops=dict(arrowstyle="->", connectionstyle='arc3'))
                # axarr[2].annotate('', xy=(0.995, 0.05), xycoords='axes fraction',
                #                   xytext=(0.865, 0.05), textcoords='axes fraction',
                #                   arrowprops=dict(arrowstyle="->", connectionstyle='arc3'))
                axarr[2, col].set_xlabel('Space (mile)', fontsize=18)
                axarr[2, col].set_ylim([0, 10])
                axarr[2, col].set_ylabel('Lane ID')
                axarr[2, col].set_yticks([3.5, 6.5])
                axarr[2, col].set_yticklabels(['1', '2'])
                axarr[2, col].set_title('Aimsun simulation', fontsize=20)

                text_str = 'Time: {0} s'.format(int(t/10.0))
                axarr[2, col].annotate(text_str, xy=(0.05, 0.88), xycoords='axes fraction', fontsize=16)

                # vehicle annotation
                dot_size = 80
                axarr[2, col].scatter([dx*20], [1], color='r', s=dot_size)
                axarr[2, col].annotate('AV', xy=(0.76, 0.06), xycoords='axes fraction', fontsize=16)
                axarr[2, col].scatter([dx*23], [1], color='b', s=dot_size)
                axarr[2, col].annotate('Car', xy=(0.87, 0.06), xycoords='axes fraction', fontsize=16)

                if plot_traj:
                    # plot each vehicle as a dot
                    snapshot = np.asarray(snaps[t])
                    car_idx = (snapshot[:, 1] == car_id)
                    av_idx = (snapshot[:, 1] == av_id)
                    lane1_idx = (snapshot[:, 2] == 1)
                    lane2_idx = (snapshot[:, 2] == 2)

                    # cars on lane 1
                    idx = (car_idx & lane1_idx)
                    dists = snapshot[idx, 3]
                    axarr[2, col].scatter(dists, 3.25*np.ones(len(dists)), color='b', s=dot_size)

                    # cars on lane 2
                    idx = (car_idx & lane2_idx)
                    dists = snapshot[idx, 3]
                    axarr[2, col].scatter(dists, 6.25*np.ones(len(dists)), color='b', s=dot_size)

                    # avs on lane 1
                    idx = (av_idx & lane1_idx)
                    dists = snapshot[idx, 3]
                    axarr[2, col].scatter(dists, 3.75*np.ones(len(dists)), color='r', s=dot_size)

                    # avs on lane 2
                    idx = (av_idx & lane2_idx)
                    dists = snapshot[idx, 3]
                    axarr[2, col].scatter(dists, 6.75*np.ones(len(dists)), color='r', s=dot_size)

            # --------------------------------------------------
            # save in folder
            plt.savefig(save_dir + '{0:05d}.png'.format(int(t)), bbox_inches='tight')
            plt.clf()
            plt.close()


def get_veh_type(veh_type_file):
    """
    This function gets the type of each vehicle id
    :return: a dict: veh_type[veh_id] = veh_type
    """
    veh_type = {}

    with open(veh_type_file, 'r') as f:
        data = csv.reader(f)
        for row in data:
            type_id = int(row[2])
            veh_id = int(row[1])
            veh_type[veh_id] = type_id

    return veh_type

# @profile
def get_snapshot(traj_file, veh_type, time_step=0.2):
    """
    This function returns the snapshot of the vehicle positions on the road.
    :param traj_file: the trajectory file name
    :param veh_type: the vehicle type dict
    :param time_step: s, the time step for taking snapshot
    :return: snapshot dict, t unit is 0.1s to remove rounding error
            snap[t] = [ [veh_id, veh_type, lane_idx, trip_dist] ]
    """
    snaps = OrderedDict()

    # create keys first to speed up
    times = np.arange(3000, 36000 + 20*time_step, 10*time_step).astype(int)
    for t in times:
        snaps[t] = []

    with open(traj_file, 'r') as f:
        i = 0
        for line in f:
            row = line.strip().split(',')

            sys.stdout.write('\r')
            sys.stdout.write('Processing row {0}/{1}'.format(i, 11555526))
            sys.stdout.flush()

            t = int(float(row[_time_idx])*10)
            # check if should extract
            if int(t)%int(time_step*10) == 0:
                # check type of vehicle
                veh_id = int(row[_veh_idx])
                try:
                    snaps[t].append([veh_id, veh_type[veh_id], int(row[_lane_idx]), float(row[_dist_idx])])
                except KeyError:
                    print('Warning: skip time key: {0}'.format(t/10.0))
            i+=1

    # remove unused keys
    for key in snaps.keys():
        if len(snaps[key]) == 0:
            del snaps[key]
            print('Removed unused time key: {0}'.format(key/10.0))

    return snaps


if __name__ == "__main__":
    sys.exit(main(sys.argv))