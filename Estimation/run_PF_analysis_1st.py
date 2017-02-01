import os
import sys
import time
from os.path import exists
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *

sys.path.append(os.getcwd() + '/TrafficModel/')
from ql_model_normal import *


def init_weight(sample):
    ## particle weight update function
    weight = 1.0 / sample * ones(sample)
    return weight


def init_state(DensityMeaInit, cellNumber, laneNumber, sample):
    mean = (DensityMeaInit[0, 8] + DensityMeaInit[0, 17]) / 2
    std = 0.05 * mean
    state = random.normal(mean, std, (sample, cellNumber))
    return state


def plot_density(data, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    plt.ylabel('Time Step', fontsize=20)
    plt.clim(0.0, 560)
    plt.xlabel('Cell Number', fontsize=20)
    plt.colorbar()
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    # plt.clf()


def plot_error(data, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    plt.ylabel('Time Step', fontsize=20)
    plt.xlabel('Cell Number', fontsize=20)
    plt.colorbar()
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    # plt.clf()


def plot_property(data, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    plt.ylabel('Time Step', fontsize=20)
    plt.clim(0.0, 1.0)
    plt.xlabel('Cell Number', fontsize=20)
    plt.colorbar(ticks=[0.0, 0.5, 1.0])
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    # plt.clf()


def plot_sampleMatch(sampleMatch, bool, savefile, directorySave):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.plot(sampleMatch, range(len(sampleMatch)))
    plt.ylabel('Time Step', fontsize=20)
    plt.xlabel('Number of distinct particles', fontsize=20)
    plt.xticks([500, 1000, 1500, 2000, 2500])
    plt.yticks([0, 30, 60, 90, 120, 150, 180])
    if bool == True:
        plt.savefig(directorySave + savefile + '.pdf', bbox_inches='tight')
    # plt.show()
    # plt.clf()


def compute_likelihood(state, densityMea, densityMeaMean, densityMeaStd, sample, sensorLocation):
    hasMeasurement = True
    if densityMea[sensorLocation[0]] == 0 and densityMea[sensorLocation[1]] == 0:
        hasMeasurement = False
        likelihood = nan

    else:
        likelihood = zeros(int(sample))
        for j in range(sample):
            modelLikelihood = 1.0
            for i in range(len(sensorLocation)):
                diff = densityMea[sensorLocation[i]] - state[j, sensorLocation[i]]
                modelLikelihood = modelLikelihood * 1.0 / (densityMeaStd * sqrt(2 * pi)) * exp(
                    -(diff - densityMeaMean) * (diff - densityMeaMean) / (2 * densityMeaStd * densityMeaStd))
            likelihood[j] = modelLikelihood
        likelihood = likelihood / sum(likelihood)
    return likelihood, hasMeasurement


def update_weight(likelihood, weight):
    ## particle weight update function
    weight = likelihood * weight
    weight = weight / sum(weight)
    return weight


def resampling(state, w, sample, cellNumber, weight):
    Cum = cumsum(weight)
    stateCopy = state.copy()
    wCopy = w.copy()

    step = 1.0 / sample
    i = 1
    u1 = random.uniform(0, step)
    for j in range(sample):
        uj = u1 + step * (j - 1)
        while uj > Cum[i]:
            i = i + 1
        state[j] = stateCopy[i]
        w[j] = wCopy[i]
    return (state, w)


def rhoc_to_w(fdpNB, PR, prRhocDict):
    rhoc = prRhocDict[PR]

    for i in range(10):
        # print('{0}<={1}<={2}'.format(fdpNB[i], rhoc, fdpNB[i+1]))

        if fdpNB[i] <= rhoc <= fdpNB[i + 1]:
            rhoc_min = fdpNB[i]
            rhoc_max = fdpNB[i + 1]
            if i == 0:
                w_min = 0.01
                w_max = 0.1
            elif i == 9:
                w_min = 0.9
                w_max = 0.99
            else:
                w_min = i / 10.0
                w_max = (i + 1) / 10.0
    w = (rhoc - rhoc_min) / (rhoc_max - rhoc_min) * (w_max - w_min) + w_min
    return w


def generate_prRhocDict(PRset, rhoc1st):
    prRhocDict = dict()
    i = 0
    for PR in PRset:
        prRhocDict[PR] = rhoc1st[i]
        i = i + 1
    return prRhocDict


###############################################################################################################

def run_estimators(fdpNB, rhoc1st, directoryLoad, directorySave, sample, sensorLocationSeed, PRsetTest):

    __dubug = True

    PRset = [0, 5, 15, 25, 35, 45, 50, 55, 65, 75, 85, 95, 100]

    # PRsetTest = [0, 25, 50, 75, 100]

    prRhocDict = generate_prRhocDict(PRset, rhoc1st)

    ###############################################################################################################
    ## noise parameters

    # TODO: Tuning the noise ===============================
    modelNoiseMean = 0.0
    modelNoiseStd = 18.0

    densityMeaMean = 0.0
    densityMeaStd = 10.0
    # TODO: Tuning the noise ===============================

    ################################################################################################################
    ## discretization
    dt = 5.0
    dx = 0.1111
    length = 3.0
    cellNumber = int(floor(length / dx))

    ################################################################################################################
    ## simulation parameter
    Lambda = dt / 3600 / dx
    timeStep = int(3600 / dt)

    # trafficModelSet = ['1st', '2nd']
    trafficModelSet = ['1st']
    # sensorLocationSeed = [1355, 2143, 3252, 8763, 12424, 23424, 24232, 24654, 45234, 59230]
    # #sensorLocationSeed = [1355]

    errorStore = zeros((len(PRsetTest), len(sensorLocationSeed), 2))
    count1 = 0
    count2 = 0

    sensorLocation = [8, 17]
    savefigure = True

    for PR in PRsetTest:

        seedCount = 0
        for seed in sensorLocationSeed:
            # For each scenario and each simulation

            densityTrue = load(directoryLoad + 'TrueDensity_' + str(PR) + '_' + str(seed) + '.npy')
            wTrue = load(directoryLoad + 'TrueW_PR_' + str(PR) + '_' + 'seed' + str(seed) +'.npy')
            densityMeasurement = load(directoryLoad + 'mea_' + str(PR) + '_' + str(seed) + '.npy')

            # TODO: Replace the noisy detector measurement by Edie's definition =================
            densityMeasurement[:, 8] = densityTrue[:, 8]
            densityMeasurement[:, 17] = densityTrue[:, 17]
            # TODO: Replace the noisy detector measurement by Edie's definition =================

            for modelMarker in trafficModelSet:
                # for two models

                marker = 'PR_' + str(PR) + '_Seed' + str(seed) + '_' + modelMarker

                #####################################################################################################

                boundary = load(directoryLoad + 'boundary_' + str(PR) + '_' + str(seed) + '.npy')

                # TODO: Replace the boundary condition by true state in boundary cells by Edie's definition ============
                boundary[:, 0] = densityTrue[:, 0]
                boundary[:, 2] = densityTrue[:, -1]
                boundary[:, 1] = wTrue[:, 0]
                boundary[:, 3] = wTrue[:, -1]
                # TODO: Replace the boundary condition by true state in boundary cells by Edie's definition ============

                if modelMarker == '1st':
                    wBoundary1st = rhoc_to_w(fdpNB, PR, prRhocDict)
                    boundary[:, 1] = wBoundary1st
                    boundary[:, 3] = wBoundary1st

                #####################################################################################################
                # Create array to save results
                estimatedState = zeros((timeStep, cellNumber))
                estimatedw = zeros((timeStep, cellNumber))

                #####################################################################################################
                # Initialization
                state = 1.0 * ones((sample, cellNumber))
                w = 0.5 * ones((sample, cellNumber))
                weight = init_weight(sample)    # equal weight

                estimatedState[0] = average(state, axis=0)
                estimatedw[0] = boundary[0, 1] * ones(cellNumber)

                #        start_time = time.time()
                #
                for k in range(1, timeStep):
                # for k in range(0, 330):
                    print('Estimating step {0}'.format(k))
                    #                if mod(k,100) == 0:
                    #                    print 'this is time step',k
                    #############################################################################################
                    # PF
                    # Set left and right boundary conditions (density(veh/mile), w)
                    bdl = boundary[k, 0], boundary[k, 1]
                    bdr = boundary[k, 2], boundary[k, 3]
                    _bdl = bdl
                    _bdr = bdr
                    # print bdl
                    # print bdr

                    for j in range(sample):

                        # TODO: Added measurement noise to the boundary conditions =================
                        wMeaMean = 0.0
                        wMeaStd = 0.01
                        boundDensMeaMean = 0.0
                        boundDensMeaStd = 3.0

                        bdl = (max([0.0, bdl[0] + random.normal(boundDensMeaMean, boundDensMeaStd)]),
                               max([0.0, bdl[1] + random.normal(wMeaMean, wMeaStd)]))
                        bdr = (max([0.0, bdr[0] + random.normal(boundDensMeaMean, boundDensMeaStd)]),
                               max([0.0, bdr[1] + random.normal(wMeaMean, wMeaStd)]))
                        # TODO: Added measurement noise to the boundary conditions =================

                        # eveolve the model prediction
                        state[j], w[j] = ctm_2ql(state[j], w[j], fdpNB, Lambda, bdl, bdr, modelNoiseMean, modelNoiseStd,
                                                 inflow=-1.0, outflow=-1.0)
                        # TODO: Added model noise to the boundary w =================
                        w[j] = w[j] + random.normal(0, 0.01, len(w[j]))
                        # TODO: Added model noise to the boundary w =================

                    densityMea = densityMeasurement[k]

                    likelihood, hasMeasurement = compute_likelihood(state, densityMea, densityMeaMean, densityMeaStd,
                                                                    sample, sensorLocation)

                    #                    hasMeasurement = False
                    if hasMeasurement:
                        weight = update_weight(likelihood, weight)

                        # TODO: Analyzing the PF =========================================
                        if __dubug is True and k >= 60:
                            # plot_sorted_weight(weight, 'Particle weight: {0} step {1}'.format(modelMarker, k))
                            dens_meas = [_bdl[0], densityMea[8], densityMea[17], _bdr[0]]
                            w_meas = [_bdl[1], _bdr[1]]
                            plot_PF_analysis(weight, k, densityTrue, wTrue, state, w, dens_meas, w_meas)

                            if False:
                                plt.show()
                            else:
                                fig_dir = os.getcwd() + '/figs_PF_analysis/'
                                plt.savefig(fig_dir + '{0:05d}.png'.format(int(k*5)), bbox_inches='tight')
                                plt.clf()
                                plt.close()
                        # TODO: Analyzing the PF =========================================

                        state, w = resampling(state, w, sample, cellNumber, weight)


                    estimatedState[k] = average(state, axis=0)
                    estimatedw[k] = average(w, axis=0)

                    # re-assign equal weight
                    weight = init_weight(sample)

                error = average(abs(estimatedState - densityTrue))

                print marker, error

                errorStore[count1, seedCount, count2] = error
                count2 = count2 + 1

                if count2 == 2:
                    count2 = 0

                # plot_density(densityTrue, True, 'PlotTrue' + marker, directorySave)
                # plot_density(estimatedState, True, 'PlotEstimationDensity' + marker, directorySave)
                # plot_property(estimatedw, True, 'PlotEstimationW' + marker, directorySave)
                save(directorySave + 'EstimationDensity_' + marker, estimatedState)
                save(directorySave + 'EstimationW_' + marker, estimatedw)
                save(directorySave + 'TrueDensity_' + marker, densityTrue)

            seedCount = seedCount + 1

            print('Finished sce {0}, seed {1}'.format(PR, seed))

        count1 = count1 + 1


    errorStoreAveSeed = average(errorStore, axis=1)

    #
    # save(directorySave+'ErrorSummary', errorStore)
    #
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.plot(PRsetTest, errorStoreAveSeed[:, 0], color='b', label='1st model')
    plt.plot(PRsetTest, errorStoreAveSeed[:, 1], color='r', label='2nd model')
    plt.ylabel('Error (veh/mile)', fontsize=20)
    plt.xlabel('Variance of AVs (%)', fontsize=20)
    plt.legend(loc=4)
    plt.xlim([0, 100])
    plt.ylim([0, 75])
    plt.savefig(directorySave + 'ErrorSummary.pdf', bbox_inches='tight')
    # plt.show()
    # plt.clf()


def plot_sorted_weight(weight, title):
    # decreasing
    sorted_w = sort(weight)[::-1]

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(arange(0, len(sorted_w)), sorted_w, color='g', linewidth=2)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Particle number')
    ax.set_ylabel('Weight')
    plt.draw()


def plot_PF_analysis(weight, k, true_dens, true_w, est_state, est_w, dens_meas, w_meas):

    fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(18, 10))

    # ------------------------------------------------------------
    # ax0 plot the sorted weight
    sorted_w = sort(weight)[::-1]
    ax0.plot(arange(0, len(sorted_w)), sorted_w, color='g', linewidth=2)
    ax0.set_title('Weight: step {0}'.format(k), fontsize=16)
    # ax0.set_xlabel('Particle number')
    ax0.set_ylabel('Weight')
    ax0.set_xlim([0, 500])
    # ax0.set_ylim([0, 0.3])

    # ------------------------------------------------------------
    # ax1 plot the true density, estimated density and the bound
    x_ticks = arange(0, 28)
    # plot true
    ax1.step(x_ticks, concatenate([[0], true_dens[k, :]]), color='r', label='True')
    # estimated
    ax1.step(x_ticks, concatenate([[0], mean(est_state, 0)]), color='g', label='estimates')
    ax1.step(x_ticks, concatenate([[0], mean(est_state, 0)-2.0*std(est_state,0)]), linestyle='--',
             color='g', label='5-95')
    ax1.step(x_ticks, concatenate([[0], mean(est_state, 0)+2.0*std(est_state,0)]), linestyle='--',
             color='g')
    # ax1.legend()
    ax1.set_title('Density: step {0}'.format(k), fontsize=16)
    # ax1.set_xlabel('Cell number')
    ax1.set_ylabel('Density (veh/mile)')
    ax1.set_xlim([-1, 28])
    ax1.set_ylim([0, 800])

    # plot the measurement
    ax1.scatter([0, 9, 18, 27], dens_meas, marker='*', s=80, color='b')

    # ------------------------------------------------------------
    # ax1 plot the true w, estimated w and the bound
    x_ticks = arange(0, 28)
    # plot true
    ax2.step(x_ticks, 100*concatenate([[0], true_w[k, :]]), color='r', label='True')
    # estimated
    ax2.step(x_ticks, 100*concatenate([[0], mean(est_w, 0)]), color='g', label='estimates')
    ax2.step(x_ticks, 100*concatenate([[0], mean(est_w, 0)-2.0*std(est_w,0)]), linestyle='--',
             color='g', label='5-95')
    ax2.step(x_ticks, 100*concatenate([[0], mean(est_w, 0)+2.0*std(est_w,0)]), linestyle='--',
             color='g')
    # ax2.legend()
    ax2.set_title('Penetration: step {0}'.format(k), fontsize=16)
    # ax2.set_xlabel('Cell number')
    ax2.set_ylabel('Penetration (veh/mile)')
    ax2.set_ylim([-10, 150])
    ax2.set_xlim([-1, 28])

    # plot the measurement
    ax2.scatter([0, 27], 100.0*array(w_meas), marker='*', s=80, color='b')


###############################################################################################################
def main(argv):
    ## fd parameters
    ## 2nd model parameters
    # rhoc_25: updated 86.57; old 86
    # rhoc_75: updated 145.63; old 142
    rhoc_0 = 71.14       # 67.58 (-5%); 71.14; 74.70 (+5%)
    rhoc_1 = 71.14       # 67.58 (-5%); 71.14; 74.70 (+5%)
    rhoc_10 = 76.27       # 72.46 (-5%); 76.27; 80.08 (+5%)
    rhoc_20 = 83.77       # 79.58 (-5%); 83.77; 87.96 (+5%)
    rhoc_25 = 86.57       # 82.24 (-5%); 86.57; 90.90 (+5%)
    rhoc_30 = 91.23       # 86.67 (-5%); 91.23; 95.79 (+5%)
    rhoc_40 = 97.53       # 92.65 (-5%); 97.53; 102.41 (+5%)
    rhoc_50 = 107.63       # 102.25 (-5%); 107.63; 113.01 (+5%)
    rhoc_60 = 116.85       # 111.01 (-5%); 116.85; 122.69 (+5%)
    rhoc_70 = 134.65       # 127.92 (-5%); 134.65; 141.38 (+5%)
    rhoc_75 = 145.63       # 138.35 (-5%); 145.63; 152.91 (+5%)
    rhoc_80 = 151.56       # 143.98 (-5%); 151.56; 159.14 (+5%)
    rhoc_90 = 183.24       # 174.08 (-5%); 183.24; 192.40 (+5%)
    rhoc_99 = 214.06       # 203.36 (-5%); 214.06; 224.76 (+5%)
    rhoc_100 = 214.06       # 203.36 (-5%); 214.06; 224.76 (+5%)

    rhom_all = 644
    # rhom_all = 460

    vmax_all = 76.28   # 72.47 (-5%); 76.28;  80.1 (+5%)

    beta = 600

    fdpNB = rhoc_0, rhoc_10, rhoc_20, rhoc_30, rhoc_40, rhoc_50, rhoc_60, \
            rhoc_70, rhoc_80, rhoc_90, rhoc_100, rhom_all, vmax_all, beta,

    # 1st model parameters
    rhoc_0 = 71.14  # 71.14; 74.7 (+5%);  update   71.13;  old 71.98
    rhoc_5 = 71.98
    rhoc_15 = 73.97
    rhoc_25 = 76.76  # 72.92 (-5%); 76.76; 80.60 (+5%)
    rhoc_35 = 79.87
    rhoc_45 = 82.91
    rhoc_50 = 84.29  # 80.08 (-5%); 84.29; 88.5 (+5%);  updated 84.29 ; old 83.5
    rhoc_55 = 85.38
    rhoc_65 = 88.81
    rhoc_75 = 92.94         # 88.3 (-5%); 92.94; 97.59 (+5%)
    rhoc_85 = 98.28
    rhoc_95 = 105.8
    rhoc_100 = 119.46  #113.52 (-5%); 119.49; 125.46 (+5%);  update    119.49 ; old 105.8

    rhoc1st = rhoc_0, rhoc_5, rhoc_15, rhoc_25, rhoc_35, rhoc_45, \
              rhoc_50, rhoc_55, rhoc_65, rhoc_75, rhoc_85, rhoc_95, rhoc_100

    # =============================================
    # set the number of samples and test sets
    # for one replication: 50:5 min; 100: 8 min; 500: 43 min; 1000: 86 min
    # samples = [50, 100, 500, 1000]  # one set takes 2.75 hr
    samples = [500]  # one set takes 2.75 hr

    # PRsetTest = [75]
    # sensorLocationSeed = [2352,59230]
    PRsetTest = [100]
    sensorLocationSeed = [24654]


    # =============================================
    directoryLoad = os.getcwd() + '/DATA/'

    for sample in samples:

        for i in range(0, 1):
            # only run one time

            directorySave = os.getcwd() + '/Result/PF_{0}/'.format(i)

            if not exists(directorySave):
                os.makedirs(directorySave)

            t_start = time.time()

            run_estimators(fdpNB, rhoc1st, directoryLoad, directorySave, sample, sensorLocationSeed, PRsetTest)

            t_end = time.time()

            print('Finished PF run {0}: {1} s'.format(i, t_end - t_start))

            #        end_time = time.time()

            # plot_density(estimatedState, True, 'state'+marker, directorySave)
            # plot_density(densityTrue, True, 'state'+marker, directorySave)

            #

            #
            #        save(directorySave+'state'+marker, estimatedState)
            ##            save(directorySave+'w'+marker, estimatedw)
            #        save(directorySave+'model'+marker, estimatedModel)
            ##            save(directorySave+'sampleMatch'+marker, sampleMatch)
            #
            #        bool = True
            #        plot_density(estimatedState, bool, 'state'+marker, directorySave)
            ##            plot_property(estimatedw, bool, 'property'+marker, directorySave)
            #        plot_parameter(estimatedModel, bool, 'model'+marker, directorySave)
            ##            plot_sampleMatch(sampleMatch, bool, 'sampleMatch'+marker, directorySave)
            #
            #
            # '''



if __name__ == "__main__":
    sys.exit(main(sys.argv))
