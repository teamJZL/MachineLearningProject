package milestone2;

import weka.classifiers.Classifier;                // Step 2
import weka.classifiers.Evaluation;                // Step 3
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;                        // Step 1
import weka.core.FastVector;                        // Step 1
import weka.core.Instance;                        // Step 2. fill training set with one instance
import weka.core.Instances;  

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.core.converters.ArffLoader.ArffReader;

public class NB {

	public static void main(String[] args) throws Exception {
		 BufferedReader reader = new BufferedReader(new FileReader("data/anneal_train.arff"));
		 ArffReader arff = new ArffReader(reader);
		 Instances isTrainingSet = arff.getData();
		 isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);
		 
		 
		 Classifier cModel = (Classifier)new NaiveBayes();   
         cModel.buildClassifier(isTrainingSet);
         
         
         BufferedReader reader2 = new BufferedReader(new FileReader("data/anneal_train.arff"));
    	 ArffReader arff2 = new ArffReader(reader2);
    	 Instances isTestSet = arff2.getData();
    	 isTestSet.setClassIndex(isTestSet.numAttributes() - 1);

         // Step 3: Test the classifier
         //===========================================================================
    	 // Test the model
    	 Evaluation eTest = new Evaluation(isTrainingSet);
    	 eTest.evaluateModel(cModel, isTestSet);
      
    	 // Print the result  la Weka explorer:
    	 String strSummary = eTest.toSummaryString();
    	 System.out.println(strSummary);
      
    	 // Get the confusion matrix
    	 double[][] cmMatrix = eTest.confusionMatrix();
     
    	 // Print out the confusion matrix (from ianma.wordpress.com)
    	 for(int row_i=0; row_i<cmMatrix.length; row_i++){
    		 for(int col_i=0; col_i<cmMatrix.length; col_i++){
    			 System.out.print(cmMatrix[row_i][col_i]);
    			 System.out.print("|");
    		 }
    		 System.out.println();
    	 }	
     
	}

}
