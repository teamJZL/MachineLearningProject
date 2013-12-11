package milestone6;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;  
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.FT;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;

import weka.core.converters.ArffLoader.ArffReader;

public class EnsembleFT {

    public static void main(String[] args) throws Exception {
        //=====================================================================
        // Edit dataset name and path here
        //=====================================================================
        String dataset_name = "sonar";
        String traindata_name = String.format("ms5_milestone5data/%s_train.arff", dataset_name);
        String testdata_name = String.format("ms5_data5bnew/%s_test.arff", dataset_name);

        BufferedReader reader = new BufferedReader(new FileReader(traindata_name));
        ArffReader arff = new ArffReader(reader);
        Instances isTrainingSet = arff.getData();
        isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

        //=====================================================================
        // For parameter tuning
        //   Explored FT parameters:
        //     -M Minimum number of instances at which a node can be split
        //     -F Functional Tree type
        //=====================================================================
        /*
        CVParameterSelection ps1 = new CVParameterSelection();
        ps1.setClassifier(new FT());
        ps1.setNumFolds(10);      //Using 10-fold Cross-validation
        ps1.addCVParameter("M 10 20 11");     //"-F" also tuned
        */

        //=====================================================================
        // Optimal FT parameters when tuned against 12 datasets
        //  (Dataset: anneal to hypothyroid)
        //=====================================================================
        Classifier tunedFT = (Classifier)new FT();
        String[] op = new String[4];
        op[0] = "-M";
        op[1] = "10";
        op[2] = "-F";
        op[3] = "0";
        tunedFT.setOptions(op);
        tunedFT.buildClassifier(isTrainingSet);

        //=====================================================================
        // Stacking ensemble methods
        //=====================================================================
        AdaBoostM1 ps1 = new AdaBoostM1();
        ps1.setClassifier(tunedFT);    //Swap tunedFT with "new FT()" to untune
        ps1.buildClassifier(isTrainingSet);

        Bagging ps2 = new Bagging();
        ps1.setClassifier(ps2);
        ps1.buildClassifier(isTrainingSet);

        //=====================================================================
        // Saving .model files
        //=====================================================================
        /*
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("models_milestone3c/hypothyroid20.model"));
        oos.writeObject(cModel);
        oos.flush();
        oos.close();
        */

        BufferedReader reader2 = new BufferedReader(new FileReader(testdata_name));
        ArffReader arff2 = new ArffReader(reader2);
        Instances isTestSet = arff2.getData();
        isTestSet.setClassIndex(isTestSet.numAttributes() - 1);

        Evaluation eTest = new Evaluation(isTrainingSet);
        eTest.evaluateModel(ps1, isTestSet);

        //=====================================================================
        // Print the classifier result a la Weka explorer to console
        //=====================================================================
        String strSummary = eTest.toSummaryString();
        System.out.println(strSummary);

        //=====================================================================
        // Print out prediction values for .predict
        //=====================================================================
        /*
        for (int i = 0; i < isTestSet.numInstances(); i++) {
            double pred = ps1.classifyInstance(isTestSet.instance(i));
            System.out.println(pred);
        }
        */

        //=====================================================================
        // Print out the confusion matrix (from ianma.wordpress.com)
        //=====================================================================
        /*
        double[][] cmMatrix = eTest.confusionMatrix();

        for(int row_i=0; row_i<cmMatrix.length; row_i++){
            for(int col_i=0; col_i<cmMatrix.length; col_i++){
                System.out.print(cmMatrix[row_i][col_i]);
                System.out.print("|");
            }
            System.out.println();
        }
        */
    }
}