# This script copies and organizes data from the estimation result folder to Video folder. 



import os
import shutil
import tempfile

sce_seeds = [(75, 3252), (75, 59230), (100, 24654), (100, 45234)]

result_dir = '../Estimation/Result/'
video_dir = '../Video/'


for sce, seed in sce_seeds:

	os.mkdir(video_dir + 'Video_sce{0}_seed{1}_data/'.format(sce, seed))

	for run in range(0, 10):

		src_dens = result_dir + 'PF_{0}/EstimationDensity_PR_{1}_Seed{2}_1st.npy'.format(run, sce, seed)
		src_w = result_dir + 'PF_{0}/EstimationW_PR_{1}_Seed{2}_1st.npy'.format(run, sce, seed)

		# copy files
		shutil.copy(src_dens, src_dens.replace('.npy', '_{0}.npy'.format(run)))
		shutil.copy(src_w, src_w.replace('.npy', '_{0}.npy'.format(run)))

		if run == 0:
			# copy the true state
			src_true = result_dir + 'PF_{0}/TrueDensity_PR_{1}_Seed{2}_1st.npy'.format(run, sce, seed)
			shutil.copy(src_true, src_true.replace('_1st.npy', '.npy'))


	print('Finished sec {0} seed {1}'.format(sce, seed))
