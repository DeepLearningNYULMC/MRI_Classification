import pandas as pd
from os import listdir

def load_ref(reference_file, home_data_dir, pecent_valid, pecent_test):
	dataref = pd.read_csv(reference_file, encoding = "ISO-8859-1")
	y = []
	filenames = []	
	subjectID = []
	outputfolder = []
	Nb_train = 0.0
	Nb_valid = 0.0
	Nb_test = 0.0
	Nb_tot = 0.0

	# First read all the dates associated with each patient (so later we can remove the last one(s) which may be associated with post-operation scans)
	ScansDates = {}
	for i in range(dataref.shape[0]):
		#pid, name, collection, study = dataref.ix[i,:]
		collection, pid, date, _, _, description, _, _, _, UID, _, _, _, _, _,_,_,_,_,_ = dataref.ix[i,:]
		# Convert date from mm/dd/yy format to yyyymmdd
		if int(date.split("/")[2]) < 18:
			newdate = int("20"+ str(date.split("/")[2]).zfill(2)  + str(date.split("/")[1]).zfill(2) + str(date.split("/")[0]).zfill(2))
		else:
			newdate = int("19"+ str(date.split("/")[2]).zfill(2)  + str(date.split("/")[1]).zfill(2) + str(date.split("/")[0]).zfill(2))

		if pid in ScansDates.keys():
			if newdate not in ScansDates[pid]:
				ScansDates[pid].append(newdate)
				ScansDates[pid].sort()
		else:
			ScansDates[pid] = [newdate]

	for i in range(dataref.shape[0]):
		#pid, name, collection, study = dataref.ix[i,:]
		collection, pid, date, _, _, description, _, _, _, UID, _, _, _, _, _,_,_,_,_,_ = dataref.ix[i,:]
		print(pid, date, description)
		# if a patient has scans acquired on several days, ignore the last ones which may be the post-operation scan
		# Convert date from mm/dd/yy format to yyyymmdd
		if int(date.split("/")[2]) < 18:
			newdate = int("20"+ str(date.split("/")[2]).zfill(2)  + str(date.split("/")[1]).zfill(2) + str(date.split("/")[0]).zfill(2))
		else:
			newdate = int("19"+ str(date.split("/")[2]).zfill(2)  + str(date.split("/")[1]).zfill(2) + str(date.split("/")[0]).zfill(2))

		if len(ScansDates[pid])==1:
			print("only one acquisition date for this patient --> keep the scan")
		elif ScansDates[pid].index(newdate) == len(ScansDates[pid]):
			print("This patient had several MRI dates, and this is the last one --> do not keep this scan, may be post-operation scan")
			continue
		elif ScansDates[pid].index(newdate) == 0:
			print("This is the first baseline scan for this patient --> keep")
		else:
			print("This is one of the intermediate baseline(s) scan for this patient --> keep or remove?")
			
		print(description.upper())
		if 'AX' not in description.upper():
			print("description '%s' ignored (may not be axial)." % description)
			continue
		elif 'T1' in description.upper():
			print("Yes, T1")
			if 'FLAIR' in description.upper():
				outputfolder_tmp = 'T1_FLAIR'
			elif any(ext in description.upper() for ext in ['GD', 'GAD']):
				if 'PRE' in description.upper():
					outputfolder_tmp = 'T1'
				else:
					outputfolder_tmp = 'T1_GD'
			else:
				outputfolder_tmp = 'T1'
		elif 'T2' in description.upper():
			if 'FLAIR' in description.upper():
				outputfolder_tmp = 'T2_FLAIR'
			else:
				outputfolder_tmp = 'T2'
		elif 'FLAIR' in description.upper():
				outputfolder_tmp = 'T2_FLAIR'
		else:
			print("description '%s' ignored." % description)
			continue
		print("description '%s' associated to folder named: %s " % (description, outputfolder_tmp) )



		list1, list2 = listdir(home_data_dir + '/TCGA-LGG'), listdir(home_data_dir + '/TCGA-GBM')
		if pid in list1:
			y_tmp = 'LGG'
			gotodir = home_data_dir + '/TCGA-LGG/' + pid
		elif pid in list2:
			y_tmp = 'GBM'
			gotodir = home_data_dir + '/TCGA-GBM/' + pid
		else:
			print("pid %s not found" % pid)
			continue

		list3 = listdir(gotodir)
		for item in list3:
			gotodiritem = gotodir+'/'+item
			try:
				list4 = listdir(gotodiritem)
			except NotADirectoryError:
				continue
			if UID in list4:
				if (pid + '_dataset') in ScansDates.keys() :
					dataset = ScansDates[pid + '_dataset']
				elif Nb_tot == 0:
					dataset = 'train_'
					Nb_train = 1.0
					Nb_tot = 1.0
				else:
					if (Nb_test / Nb_tot) < (Nb_tot / 100.0):
						dataset = 'test_'
						Nb_test += 1.0
					elif (Nb_valid / Nb_tot) < (Nb_tot / 100.0):
						dataset = 'valid_'
						Nb_valid += 1.0
					else:
						dataset = 'train_'
						Nb_train += 1.0
					Nb_tot += 1.0

				ScansDates[pid + '_dataset'] = dataset
				print(Nb_tot, Nb_train, Nb_valid, Nb_test, Nb_test / Nb_tot, Nb_valid / Nb_tot)
				
				filenames.append(gotodiritem+'/'+UID)
				subjectID.append(dataset + pid + "_" + str(newdate))
				y.append(y_tmp)
				outputfolder.append(outputfolder_tmp)
				#print(pid, UID)

	return y, filenames, subjectID, outputfolder


if __name__=='__main__':
	load_ref()
