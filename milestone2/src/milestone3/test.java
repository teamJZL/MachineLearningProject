package milestone3;

import weka.classifiers.Classifier;                // Step 2
import weka.classifiers.Evaluation;                // Step 3
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;                        // Step 1
import weka.core.FastVector;                        // Step 1
import weka.core.Instance;                        // Step 2. fill training set with one instance
import weka.core.Instances;  
import weka.core.Utils;
import weka.classifiers.trees.FT;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;

import weka.core.converters.ArffLoader.ArffReader;
import weka.classifiers.meta.*;

public class test {

	public static void main(String[] args) throws Exception {
		 BufferedReader reader = new BufferedReader(new FileReader("C:/Users/Vineet/wekafiles/dataout/anneal_train.arff"));
		 ArffReader arff = new ArffReader(reader);
		 Instances isTrainingSet = arff.getData();
		 isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);
		 
		 
		// setup classifier
	    CVParameterSelection ps = new CVParameterSelection();
	    ps.setClassifier(new FT());
	    ps.setNumFolds(10);  // using 10-fold CV
	    ps.addCVParameter("M 10 20 11");
	    

	    // build and output best options
	    ps.buildClassifier(isTrainingSet);
	    System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
	  
		 
	}
}
