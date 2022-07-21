# Incorporating Polar Field Data for Solar Flare Prediction

This reprository includes the codes of the paper. To generate the results, follow the steps:

1. Create environment using requirements.txt
2. Download SHARP/SMARP/GOES/Polar datasets from https://drive.google.com/drive/folders/12EBAdsk-fnxV-_xT75xHnCQkshi1Ip2q?usp=sharing
3. Modify self.path_goes, self.path_smarp, self.path_sharp in generate_event_dataset.py to indicate download paths. 
4. Modify path_polar in generate_instance_dataset.py to indicate download paths. 
5. Run generate_event_dataset.py
6. Run generate_instance_dataset.py
7. Run run_exps.py
