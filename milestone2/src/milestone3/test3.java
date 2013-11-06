package milestone3;

import weka.classifiers.Classifier;                // Step 2
import weka.classifiers.Evaluation;                // Step 3
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;                        // Step 1
import weka.core.FastVector;                        // Step 1
import weka.core.Instance;                        // Step 2. fill training set with one instance
import weka.core.Instances;  
import weka.core.Utils;
import weka.classifiers.rules.PART;
import weka.classifiers.functions.*;
import weka.classifiers.functions.supportVector.*;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;

import weka.core.converters.ArffLoader.ArffReader;
import weka.classifiers.meta.*;

public class test3 {

	public static void main(String[] args) throws Exception {
		 BufferedReader reader = new BufferedReader(new FileReader("C:/Users/Vineet/wekafiles/dataout/anneal_train.arff"));
		 ArffReader arff = new ArffReader(reader);
		 Instances isTrainingSet = arff.getData();
		 isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);
		 
		 
		 BufferedReader reader2 = new BufferedReader(new FileReader("C:/Users/Vineet/wekafiles/dataout/anneal_test.arff"));
	     ArffReader arff2 = new ArffReader(reader2);
   	 	 Instances isTestSet = arff2.getData();
   	 	 isTestSet.setClassIndex(isTestSet.numAttributes() - 1);
   	 	 
   	// setup classifier1
 	    CVParameterSelection ps1 = new CVParameterSelection();
 	    ps1.setClassifier(new SMO());
 	    ps1.setNumFolds(10);  // using 10-fold CV
 	    ps1.addCVParameter("P 1.0e-14 1.0e-10 10");
 	    

 	    // build and output best options
 	    ps1.buildClassifier(isTrainingSet);
 	    System.out.println(Utils.joinOptions(ps1.getBestClassifierOptions()));
 	    
 	    
 	    // Step 3: Test the classifier
         //===========================================================================
 	    //Test the model
 	    Evaluation eTest = new Evaluation(isTrainingSet);
 	    eTest.evaluateModel(ps1, isTestSet);
      
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
 	  
 		  	 		
   	// setup classifier 2
   	   	
	    CVParameterSelection ps2 = new CVParameterSelection();
	    ps2.setClassifier(new SMO());
	    ps2.setNumFolds(10);  // using 10-fold CV
	    ps2.addCVParameter("L 1.0e-6 1.0e-2 10");
	    

	    // build and output best options
	    ps2.buildClassifier(isTrainingSet);
	    System.out.println(Utils.joinOptions(ps2.getBestClassifierOptions()));
	    
	        
	       
	    // Step 3: Test the classifier
        //===========================================================================
	    //Test the model
	    Evaluation eTest2 = new Evaluation(isTrainingSet);
	    eTest2.evaluateModel(ps2, isTestSet);
     
	    // Print the result  la Weka explorer:
	    String strSummary2 = eTest2.toSummaryString();
	    System.out.println(strSummary2);
     
	    // Get the confusion matrix
	    double[][] cmMatrix2 = eTest2.confusionMatrix();
    
	    // Print out the confusion matrix (from ianma.wordpress.com)
   	 	for(int row_i=0; row_i<cmMatrix2.length; row_i++){
   	 		for(int col_i=0; col_i<cmMatrix2.length; col_i++){
   	 			System.out.print(cmMatrix2[row_i][col_i]);
   	 			System.out.print("|");
   	 		}
   	 		System.out.println();
   	 	}
   	 		
   	 		
   	    // classifier 3
   	 	String[] op = new String[1];
   	    op[0] = "-M";
   	 	Classifier ps3 = (Classifier)new SMO(); 
   	    ps3.setOptions(op);
   	    
        ps3.buildClassifier(isTrainingSet);
        
        
        // Step 3: Test the classifier
        //===========================================================================
	    //Test the model
	    Evaluation eTest3 = new Evaluation(isTrainingSet);
	    eTest3.evaluateModel(ps3, isTestSet);
     
	    // Print the result  la Weka explorer:
	    String strSummary3 = eTest3.toSummaryString();
	    System.out.println(strSummary3);
     
	    // Get the confusion matrix
	    double[][] cmMatrix3 = eTest3.confusionMatrix();
    
	    // Print out the confusion matrix (from ianma.wordpress.com)
   	 	for(int row_i=0; row_i<cmMatrix3.length; row_i++){
   	 		for(int col_i=0; col_i<cmMatrix3.length; col_i++){
   	 			System.out.print(cmMatrix3[row_i][col_i]);
   	 			System.out.print("|");
   	 		}
   	 		System.out.println();
   	 	}	
	}
}

