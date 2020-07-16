The all TXT file is the feature out of all models, and finally all results can be merged averagely to obtain the last result.
 You can also start from scratch using our training files.
 If you need to reproduce the results of generating TXT files, please download our own weights for training.
 All of the diFF and RGB in here are the weights of slowfast's network, which has the same structure so it has common weights, which are trained with different intensities of data, respectively, including raw data training and using Concat inversion plus raw data training, and different interval sampling frame training.
 
 
 Model Download:
 The weight of TSM can be downloaded in https://drive.google.com/file/d/1UibJjxA-klsEa3q-BxtrNhd66Y2rKlqi/view?usp=sharing
 The all slowfastnetwork can be downloaded in https://drive.google.com/file/d/19op536Vlv3vcewJMxsZPoHuamWHCrYyb/view?usp=sharing
 https://drive.google.com/file/d/1l0e3RNvLIXRA0jJ8sKYasNFJMfE9fFbn/view?usp=sharing
 https://drive.google.com/file/d/1YoqalfYs5Qh3X-zA7f5sfvQ9rGN2yk2t/view?usp=sharing
 https://drive.google.com/file/d/1lVGa7r5BHnc8f_2IQN67UTR0yRlHQi0f/view?usp=sharing
 
 https://drive.google.com/file/d/1cnjrfWsWZGBnwKSZ-6P-fkHHTea4mpJu/view?usp=sharing
 https://drive.google.com/file/d/1HZ5pPetnkpIzj7cXUfFHGwOoVu7ed7E5/view?usp=sharing
Note: DIFF and RGB are the same network,  the difference is  inter-frame information and single frame information.

prepare dataset:
The ucf101 videos needs to be put in '/data/xuwd/video/'. 
The label need a txt file, where the video name corresponds to the category ranging from 1 to 101.

trainRGB:
python trainrgb.py

traindiff:
python traindiff.py

traintsm:
python traintsm.py

each python file has a submit fuction, you can run it to submit.
Or you can run python test mult_model.py to submit.
An example can be seen in test_multmodel_new.py.
