outputFolder = fullfile('Caltech101');
rootFolder = fullfile(outputFolder,'101_ObjectCategories');
categories = {'airplanes','ferry','laptop'}; % category of images you want to classify
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames'); % Names each images by the folder names
tbl = countEachLabel(imds); % Find the number of images
minSetCount = min(tbl{:,2}); % find the minimum number of images
imds = splitEachLabel(imds,minSetCount, 'randomize'); %%Choose 67 images randomly and reduce them to 67
countEachLabel(imds); % count the number of images after reducing the images to 67
% the number of images are now reduced to 67
airplanes = find(imds.Labels == 'airplanes',1); % find the images located in imds function
ferry = find(imds.Labels == 'ferry',1);
laptop = find(imds.Labels == 'laptop',1);

% figure; % shows airplane image
% subplot(2,2,1);
% imshow(readimage(imds,airplanes));
% subplot(2,2,2); % shows ferry images
% imshow(readimage(imds,ferry));
% subplot (2,2,3); % shows laptop images
% imshow(readimage(imds,laptop));

% load the pretrained network
net = resnet50();
% figure; %plot the pretrained network
% plot(net);
% title('Architecture of Resnet-50');
% %  the figure is complex to understand. to resize the figure to understand it more
% % gca load the current access
% %YLim or y limit [150,]
% set(gca,'YLim',[150 170]);
%  net.Layers(1);
% net.Layers(end);

numel(net.Layers(end).ClassNames)
% to divide the images so as to train some of the images and some to test
% the network. the splitFunctionlabel works on images stored in imds
% 30 percent of the images for training. select images randomly
[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');
% get the required size of input images                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
imageSize = net.Layers(1).InputSize;
% colour preprocessing == converts gray iamge to rgb colour image
augmentedTrainingSet = augmentedImageDatastore(imageSize,trainingSet, ...
    'ColorPreprocessing','gray2rgb'); %resizes an converts any image to RGB image

augmentedTestSet = augmentedImageDatastore(imageSize,testSet, ...
    'ColorPreprocessing','gray2rgb');%resizes an converts any image to RGB image
% get the weight of the cnn
w1 = net.Layers(2).Weights;
% the images is in matrix form. we convert it to image form
w1 = mat2gray(w1);

% figure;
% montage(w1);
% title('First Convolutional Layer Weight' );
% %extract the features of a layer
featureLayer = 'fc1000';
% miniBatch size is set to 32 to ensure that the cnn and imagedata  fit
% into gpu memory. extracts the features

trainingFeatures = activations(net,augmentedTrainingSet,featureLayer,...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
% labels of the training set
trainingLabels = trainingSet.Labels;
 %to train rhe svm use
classifier = fitcecoc(trainingFeatures,trainingLabels,'Learner','Linear','Coding',...
    'onevsall','ObservationsIn','columns');
% it returns a full trained multi class output model
% it uses the k(k-1)/2 binary suppot vector machine k is the numbers of
% unique class arguments
testFeatures = activations(net,augmentedTestSet,featureLayer,...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
% gives the features of test images. we can measure the accuracy of the
% trained classifier.
predictLabels =  predict(classifier,testFeatures,'ObservationsIn','columns');
testLabels = testSet.Labels;
% evaluates the perfomance of the classifier confusionmat generates the
% matrix
% confMat = confusionmat(testLabels,predictLabels );
%  to generate the percentage value from matrix bsxfun

% confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
%sum(confMat,2);  calcultates the number of entire row of and puts value as the first element
% 47
% mean(diag(confMat)); % acuracy of the model
% classifier for augmented test set is 99.29%

newImage = imread(fullfile('test1-1.png.jpg'));
ds = augmentedImageDatastore(imageSize,newImage,...
    'ColorPreprocessing','gray2rgb');
 imageFeatures = activations(net,ds,featureLayer,...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
label =  predict(classifier,imageFeatures,'ObservationsIn','columns');
sprintf(' The loaded image belongs to %s class',label)
newImage2 = imread(fullfile('laptoptest102.jpg'));
ds2 = augmentedImageDatastore(imageSize,newImage2,...
    'ColorPreprocessing','gray2rgb');
imageFeatures2 = activations(net,ds2,featureLayer,...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
label1 =  predict(classifier,imageFeatures2,'ObservationsIn','columns');
sprintf(' The loaded image belongs to %s class',label1)
