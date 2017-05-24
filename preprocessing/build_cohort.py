import pandas as pd
import pickle
import settings
from os import listdir
import numpy as np

def load_ref():
	dataref = pd.read_excel(settings.reference_file)
	labelref = pd.read_excel(settings.label_file)
	y = []
	filenames = []	
	subjectID = []
	
	for i in range(dataref.shape[0]):
		pid, name, collection, study = dataref.ix[i,:]
		if study == 'T1 AX':
			subjectID.append(name)
			print(name, collection)
			list1, list2 = listdir(settings.home_data_dir + '/TCGA-LGG'), listdir(settings.home_data_dir + '/TCGA-GBM')
			if name in list1:
				y.append(0)
				gotodir = settings.home_data_dir + '/TCGA-LGG/' + name
			if name in list2:
				y.append(1)
				gotodir = settings.home_data_dir + '/TCGA-GBM/' + name
			list3 = listdir(gotodir)
			for item in list3:
				gotodiritem = gotodir+'/'+item
				try:
					list4 = listdir(gotodiritem)
				except NotADirectoryError:
					pass
				if collection in list4:
					filenames.append(gotodiritem+'/'+collection)

	return y,filenames,subjectID


if __name__=='__main__':
	load_ref()