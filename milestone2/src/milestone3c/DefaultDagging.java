package milestone3c;

import weka.classifiers.Classifier;                // Step 2
import weka.classifiers.Evaluation;                // Step 3
import weka.core.Instances;  
import weka.classifiers.meta.Dagging;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;

import weka.core.converters.ArffLoader.ArffReader;

public class DefaultDagging {

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("data/credit-a_train.arff"));
        ArffReader arff = new ArffReader(reader);
        Instances isTrainingSet = arff.getData();
        isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

        Classifier cModel = (Classifier)new Dagging(); 
/*        String[] op = new String[4];
        op[0] = "-P";
   	    op[1] = "1e-14";
   	    op[2] = "-L";
	    op[3] = "1e-6";
   	    cModel.setOptions(op);*/
        cModel.buildClassifier(isTrainingSet);

        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("models_milestone3c/credit-a0.model"));
        oos.writeObject(cModel);
        oos.flush();
        oos.close();

        BufferedReader reader2 = new BufferedReader(new FileReader("data/credit-a_test.arff"));
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
        for(int row_i=0; row_i<cmMatrix.length; row_i++){
            for(int col_i=0; col_i<cmMatrix.length; col_i++){
                System.out.print(cmMatrix[row_i][col_i]);
                System.out.print("|");
            }
            System.out.println();
        }
    }
}