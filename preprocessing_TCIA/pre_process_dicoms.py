import dicom
import numpy as np
import pickle
from os import listdir
import os

import settings
import build_cohort

def load_data(filenames, subjectID, labels):
	data = []
	targetFiles = []

	for i, subject_dir in enumerate(filenames):

		dicoms_paths = sorted(listdir(subject_dir))
		# print(dicoms_paths)
		slice_0 = dicom.read_file(subject_dir + '/' + dicoms_paths[0])
		ConstPixelDims = (int(slice_0.Rows), int(slice_0.Columns), len(dicoms_paths))
		ConstPixelSpacing = (float(slice_0.PixelSpacing[0]), float(slice_0.PixelSpacing[1]), float(slice_0.SliceThickness))
		x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
		y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
		z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
		ArrayDicom = np.zeros(ConstPixelDims, dtype=slice_0.pixel_array.dtype)
		print(subjectID[i], labels[i], ConstPixelDims)

		flag = 0
		for dicom_slices in dicoms_paths:
			slice_i = dicom.read_file(subject_dir + '/' + dicom_slices)
			try:
				ArrayDicom[:, :, dicoms_paths.index(dicom_slices)] = slice_i.pixel_array				
			except TypeError:
				print ('error with dicom file:' + subject_dir + '/' + dicom_slices)
				flag = 1

		outputfilename = settings.output_data_dir + '/'+ subjectID[i] + '_' + str(labels[i]) + '.np.pkl'
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

def run():
	print('loading the reference file')
	y, filenames, subjectID = build_cohort.load_ref()

	print('starting the preprocessing of the files')
	data, targetFiles = load_data(filenames,subjectID,y)

	save_data(settings.output_data_dir + '/alldata.pkl', data)
	save_data(settings.output_data_dir + '/meta_info.pkl', (y, filenames, subjectID, targetFiles) )

	return (data, targetFiles)

if __name__=='__main__':
	run()