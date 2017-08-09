import dicom
import numpy as np
import pickle
from os import listdir
import os
import argparse
from scipy.misc import imsave
import glob

import build_cohort




def load_data(filenames, subjectID, labels, outputfolder):
	data = []
	targetFiles = []

	for i, subject_dir in enumerate(filenames):
		base_path = os.path.join(FLAGS.output_data_dir, outputfolder[i])
		# Create a directory for each type of scan (T1, T2, etc)
		if os.path.isdir(base_path):	
			pass
		else:
			os.makedirs(base_path)

		#dicoms_paths = sorted(listdir(os.path.join(subject_dir, "*.dcm")))
		dicoms_paths = sorted(glob.glob(os.path.join(subject_dir, "*.dcm")))
		if len(dicoms_paths) == 0:
			print("No dcm file found in directory %s" % subject_dir)
			continue
		print(os.path.join(subject_dir, dicoms_paths[0]))
		# slice_0 = dicom.read_file(subject_dir + '/' + dicoms_paths[0])
		slice_0 = dicom.read_file( os.path.join(subject_dir, dicoms_paths[0]) )
		ConstPixelDims = (int(slice_0.Rows), int(slice_0.Columns), len(dicoms_paths))
		ConstPixelSpacing = (float(slice_0.PixelSpacing[0]), float(slice_0.PixelSpacing[1]), float(slice_0.SliceThickness))
		x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
		y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
		z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
		ArrayDicom = np.zeros(ConstPixelDims, dtype=slice_0.pixel_array.dtype)
		print(subjectID[i], labels[i], ConstPixelDims)

		flag = 0
		for dicom_slices in dicoms_paths:
			slice_i = dicom.read_file (os.path.join(subject_dir,dicom_slices) )
			SliceIndex = str(slice_i.InstanceNumber).zfill(4)
			try:
				ArrayDicom[:, :, dicoms_paths.index(dicom_slices)] = slice_i.pixel_array				
			except TypeError:
				print ('error with dicom file:' + subject_dir + '/' + dicom_slices)
				flag = 1

			# save as jpg
			try:
				im1 = slice_i.pixel_array
				maxVal = float(im1.max())
				height = im1.shape[0]
				width = im1.shape[1]
				image = np.zeros((height,width,3), 'uint8')
				image[...,0] = (im1[:,:].astype(float)  / maxVal * 255.0).astype(int)
				image[...,1] = (im1[:,:].astype(float)  / maxVal * 255.0).astype(int)
				image[...,2] = (im1[:,:].astype(float)  / maxVal * 255.0).astype(int)

				outputfilename = os.path.join(base_path, subjectID[i] + '_' + str(labels[i]) + "_" + str(slice_i.InstanceNumber).zfill(4) + '.jpg')

				imsave(outputfilename,image)
			except:
				print("image %s failed to be saved as jpg" % dicom_slices)

		outputfilename = os.path.join(base_path, subjectID[i] + '_' + str(labels[i]) + '.np.pkl')
		print (subject_dir, outputfilename)

		if flag == 0:
			targetFiles.append(outputfilename)
			save_data(outputfilename, ArrayDicom)
		else:
			targetFiles.append('null')
		
		data.append(ArrayDicom)

	return data, targetFiles



def save_data(fname, variable):
	directory = os.path.dirname(fname)
	if not os.path.exists(directory):
		os.makedirs(directory)
	
	if os.path.exists(fname):
		print ('warning: subject have another scan' + fname)
		fname = fname+ '_extra'

	pickle.dump(variable, open(fname, 'wb'), -1)

def run(FLAGS):
	print('loading the reference file')
	y, filenames, subjectID, outputfolder = build_cohort.load_ref(FLAGS.reference_file, FLAGS.home_data_dir, FLAGS.pecent_valid, FLAGS.pecent_test)

	print('starting the preprocessing of the files')
	data, targetFiles = load_data(filenames,subjectID,y, outputfolder)

	save_data(FLAGS.output_data_dir + '/alldata.pkl', data)
	save_data(FLAGS.output_data_dir + '/meta_info.pkl', (y, filenames, subjectID, targetFiles) )

	return (data, targetFiles)

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--reference_file',
		type=str,
		default='/ifs/data/razavianlab/brain/MRIdata.csv',
		help="""\
		Metadata files for dcm from database.\
		"""
	)
	parser.add_argument(
		'--home_data_dir',
		type=str,
		default='/ifs/data/razavianlab/brain/TCIA/',
		help='Main dcm input directory.'
	)
	parser.add_argument(
		'--output_data_dir',
		type=str,
		default='/ifs/data/razavianlab/brain/test_output/',
		help='Output directory.'
	)
	parser.add_argument(
		'--pecent_valid',
		type=float,
		default='15',
		help='Percentage of patient to put in validation set.'
	)
	parser.add_argument(
		'--pecent_test',
		type=float,
		default='15',
		help='Percentage of patient to put in test set.'
	)
	FLAGS, unparsed = parser.parse_known_args()

	run(FLAGS)

