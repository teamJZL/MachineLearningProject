package milestone6;

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

public class TuneFT {

    public static void main(String[] args) throws Exception {
        String dataset_name = "arrhythmia";
        String traindata_name = String.format("ms5_milestone5data/%s_train.arff", dataset_name);
        String testdata_name = String.format("ms5_data5bnew/%s_test.arff", dataset_name);
        
        BufferedReader reader = new BufferedReader(new FileReader(traindata_name));
        ArffReader arff = new ArffReader(reader);
        Instances isTrainingSet = arff.getData();
        isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

        BufferedReader reader2 = new BufferedReader(new FileReader(testdata_name));
        ArffReader arff2 = new ArffReader(reader2);
        Instances isTestSet = arff2.getData();
        isTestSet.setClassIndex(isTestSet.numAttributes() - 1);

        //===========================================================================
        // setup classifier 1
        //===========================================================================
        CVParameterSelection ps1 = new CVParameterSelection();
        ps1.setClassifier(new FT());
        ps1.setNumFolds(10);  // using 10-fold CV
        ps1.addCVParameter("M 10 20 11");
        ps1.addCVParameter("F 0 2 3");

        // build and output best options
        ps1.buildClassifier(isTrainingSet);
        System.out.println(Utils.joinOptions(ps1.getBestClassifierOptions()));
/*
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("models_milestone3c/hypothyroid21.model"));
        oos.writeObject(ps1);
        oos.flush();
        oos.close();
*/
        // Step 3: Test the classifier
        //===========================================================================
        //Test the model
        Evaluation eTest = new Evaluation(isTrainingSet);
        eTest.evaluateModel(ps1, isTestSet);

        // Print the result a la Weka explorer:
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