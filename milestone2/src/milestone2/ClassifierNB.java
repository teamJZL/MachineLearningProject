package milestone2;

import weka.classifiers.Classifier;                // Step 2
import weka.classifiers.Evaluation;                // Step 3
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;  

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.converters.ArffLoader.ArffReader;

public class ClassifierNB {

    public static void main(String[] args) throws Exception {
        String dataset_name = "arrhythmia";
        String traindata_name = String.format("ms5_milestone5data/%s_train.arff", dataset_name);
        String testdata_name = String.format("ms5_data5bnew/%s_test.arff", dataset_name);
        
        BufferedReader reader = new BufferedReader(new FileReader(testdata_name));
        ArffReader arff = new ArffReader(reader);
        Instances isTrainingSet = arff.getData();
        isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

        Classifier cModel = (Classifier)new NaiveBayes();   
        cModel.buildClassifier(isTrainingSet);


        BufferedReader reader2 = new BufferedReader(new FileReader(traindata_name));
        ArffReader arff2 = new ArffReader(reader2);
        Instances isTestSet = arff2.getData();
        isTestSet.setClassIndex(isTestSet.numAttributes() - 1);

        // Step 3: Test the classifier
        //===========================================================================
        // Test the model
        Evaluation eTest = new Evaluation(isTrainingSet);
        eTest.evaluateModel(cModel, isTestSet);

        // Print the result a la Weka explorer:
        String strSummary = eTest.toSummaryString();
        System.out.println(strSummary);

        // Get the confusion matrix
        double[][] cmMatrix = eTest.confusionMatrix();

        // Print out the confusion matrix (from ianma.wordpress.com)
        /*for(int row_i=0; row_i<cmMatrix.length; row_i++){
            for(int col_i=0; col_i<cmMatrix.length; col_i++){
                System.out.print(cmMatrix[row_i][col_i]);
                System.out.print("|");
            }
            System.out.println();
        }*/
    }
}