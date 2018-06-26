import numpy as np
from sklearn.cross_validation import StratifiedKFold
import copy

class StackedPredictionCV(object):
	'''Stacked models object which takes in an array of several pre_defined classifier objects and an X and y arrays
	and performs an m fold cross validation over X and y where each model is built over each of the 1,m-1 folds
	and is used to predict the mth fold.  The out of fold predictions for each model are stacked (essentially rebuilding)
	and converted into a n x p matrix where n is the number of observations in X and p is the number of models listed in the
	array of models used in the creation of the object.  The matrix can then be used as part of a separate process where
	the predictions outputted or the predicted probabilities can be used as features in a different model
	Parameters:
	------------
	models_array: an array of pre-defined model objects (with the right hyper parameters)
	X: an n x m matrix of features where n is the number of samples and m is the number of features
	y: a vector of length n which contains the target variable
	n_folds: the number of folds of the cross-validation process
	seed: random state used in the StratifiedKFold
	'''

	def __init__(self,models_array,second_level_clf,use_probs=True,n_folds=10,random_state=0):
		'''initialize using an array of model objects'''
		self.models_array = models_array
		self.second_level_clf = second_level_clf
		self.n_folds = n_folds
		self.random_state = random_state
		self.use_probs = use_probs

	def _fit_transform_level_1(self,X,y):
		'''fit the models by passing in X feature matrix and y target vector'''
		
		kfold = StratifiedKFold(y,n_folds=self.n_folds,random_state=self.random_state)

		#initializing the matrices which will hold the actuals and predictions of all the models
		y_actual_stack = np.array([])
		y_pred_stack = np.array([])
		y_pred_proba_stack = np.array([])
		model_stack = []

		for iteration,model in enumerate(self.models_array): 
		    
		    #initializing the arrays which will hold the predictions of each individual model and the actual values
		    y_test_folds = np.array([]).reshape((0,1))
		    y_pred_folds = np.array([]).reshape((0,1))
		    y_pred_proba_folds = np.array([]).reshape((0,1))

		    #holds the individually trained models for model i
		    model_i_container = []

		    for k,(train,test) in enumerate(kfold):
		        
		        #fit the current model on the current number of data folds then make a prediction using the test fold
		        model.fit(X[train],y[train])
		        model_i_container.append(copy.deepcopy(model)) #store the model for fold k into the model container
		        y_pred = model.predict(X[test])
		        y_pred_proba = model.predict_proba(X[test])[:,1]
		        #saving the y_actual fold to stack later --> it will end up being reshuffled because of the StratifiedKFold
		        y_actual_fold = y[test]

		        #stacking the folds of actuals and predictions into vectors
		        if iteration == 0: #only perform this for the first iteration (saving the y target values after folding is done)
		            y_test_folds = np.vstack((y_test_folds,y_actual_fold.reshape((len(y_actual_fold),1))))
		            if k==len(kfold)-1: #if in the last iteration of the k-fold - save to the y_actual_stack
		                y_actual_stack = y_test_folds
		        
		        #these need to be run for every fold and for every model
		        y_pred_folds = np.vstack((y_pred_folds,y_pred.reshape((len(y_pred),1))))
		        y_pred_proba_folds  = np.vstack((y_pred_proba_folds,y_pred_proba.reshape((len(y_pred_proba),1))))
		        if k==len(kfold)-1: #if in the last iteration of the k-fold - horizontally stack
		            if iteration == 0: #for the very first iteration initialize the stacked models as the current vectors
		                y_pred_stack = y_pred_folds
		                y_pred_proba_stack = y_pred_proba_folds
		            else:
		                y_pred_stack = np.hstack((y_pred_stack,y_pred_folds))
		                y_pred_proba_stack = np.hstack((y_pred_proba_stack,y_pred_proba_folds))

		    model_stack.append(model_i_container)
		n=len(y_actual_stack)
		self.y_actual_stack = y_actual_stack.reshape((n,))
		self.y_pred_stack = y_pred_stack
		self.y_pred_proba_stack = y_pred_proba_stack
		self.model_stack = model_stack

	def _predict(self):
		'''returns the n x p matrix with model predicted values'''
		return self.y_pred_stack

	def _predict_proba(self):
		'''returns the n x p matrix of predicted probability values'''
		return self.y_pred_proba_stack

	def _actual_target(self):
		'''returns the target values which have been reshuffled after the x-validation'''
		return self.y_actual_stack.reshape((len(self.y_actual_stack),))

	def _transform_level_1(self,X):
		'''a function which takes in an X vector and processes it iteratively by the pre-trained model stack
		for each model, voting is done across the models such that only one 'predicted' value is returned per model'''

		model_preds = np.array([])
		for iteration,model in enumerate(self.model_stack):
			for i, model_i in enumerate(model):
				#make a prediction using model_i
				y_pred = model_i.predict(X)
				y_pred_proba = model_i.predict_proba(X)[:,1]
				#if i == 0, then initialize
				if i == 0:
					model_i_preds = y_pred
					model_i_preds_proba = y_pred_proba
				else:
					model_i_preds = np.vstack((model_i_preds,y_pred)) #otherwise keep stacking
					model_i_preds_proba = np.vstack((model_i_preds_proba,y_pred_proba))

			#once finished with model_i, then average across the individual sub-models
			average_model_i = np.array(np.rint(model_i_preds.mean(axis=0)),dtype='int64').reshape(len(y_pred),1)
			average_model_i_proba = model_i_preds_proba.mean(axis=0).reshape(len(y_pred),1)

			#if it is the first model iteration, the nitialize the model_preds array
			if iteration == 0:
				model_preds = average_model_i
				model_preds_proba = average_model_i_proba
			else:
				model_preds = np.hstack((model_preds,average_model_i))
				model_preds_proba = np.hstack((model_preds_proba,average_model_i_proba))

		self.model_preds = model_preds
		self.model_preds_proba = model_preds_proba

	def _fit_level_2(self,X,y):
		'''method which fits teh level two model. it takes in a classifier as parameter
		and also the X and y values used in the classifier.  X has been transformed by the _fit_transform_level_1
		method already'''
		self.second_level_clf.fit(X,y)

	def fit(self,X,y):
		'''the method which trains both the level one models and the level 2 model'''
		use_probs = self.use_probs
		#train level one models by using the full X and y with CV
		self._fit_transform_level_1(X,y)
		#train the level 2 model using the crossvalidated out of fold predictions and the y target
		if use_probs == False:
			self._fit_level_2(X=self.y_pred_stack,y=self.y_actual_stack)
		if use_probs == True:
			self._fit_level_2(X=self.y_pred_proba_stack,y=self.y_actual_stack)

	def predict(self,X):
		'''method which is used to make a prediction using the stacked models
		it first transforms the X matrix using the level 1 models and then it performs a 
		prediction using the level 2 classifier
		returns a vector of predicted labels'''
		self._transform_level_1(X)

		if self.use_probs == False:
			y_pred = self.second_level_clf.predict(self.model_preds)
		if self.use_probs == True:
			y_pred = self.second_level_clf.predict(self.model_preds_proba)
		return y_pred

	def predict_proba(self,X):
		'''method which is used to make a prediction of probabilities using the stacked models
		it first transforms the X matrix using the level 1 models and then it performs a 
		prediction using the level 2 classifier
		returns a vector of predicted probabilities'''
		self._transform_level_1(X)
		
		if self.use_probs == False:
			y_pred_proba = self.second_level_clf.predict_proba(self.model_preds)[:,1]
		if self.use_probs == True:
			y_pred_proba = self.second_level_clf.predict_proba(self.model_preds_proba)[:,1]
		return y_pred_proba