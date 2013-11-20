package milestone4;

import weka.classifiers.Classifier;                // Step 2
import weka.classifiers.Evaluation;                // Step 3
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;                        // Step 1
import weka.core.FastVector;                       // Step 1
import weka.core.Instance;                         // Step 2. fill training set with one instance
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

public class ftada {

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("data/breast-cancer_train.arff"));
        ArffReader arff = new ArffReader(reader);
        Instances isTrainingSet = arff.getData();
        isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

        BufferedReader reader2 = new BufferedReader(new FileReader("data/breast-cancer_test.arff"));
        ArffReader arff2 = new ArffReader(reader2);
        Instances isTestSet = arff2.getData();
        isTestSet.setClassIndex(isTestSet.numAttributes() - 1);

        //===========================================================================
        // setup classifier 1
        //===========================================================================
        	//CVParameterSelection ps1 = new CVParameterSelection();
        	//ps1.setClassifier(new FT());
        	//ps1.setNumFolds(10);  // using 10-fold CV
        	//ps1.addCVParameter("M 10 20 11");
         
        AdaBoostM1 ps2 = new AdaBoostM1();
        ps2.setClassifier(new FT());
                
        Bagging ps1 = new Bagging();
        ps1.setClassifier(ps2);
        // build and output best options
        ps1.buildClassifier(isTrainingSet);
        
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

    }
}