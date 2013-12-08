package milestone5;
//ms4a
import weka.classifiers.Classifier;                // Step 2
import weka.classifiers.Evaluation;                // Step 3
import weka.core.Instances;  
import weka.classifiers.trees.FT;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;

import weka.core.converters.ArffLoader.ArffReader;

public class ftparam {

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader("ms5_milestone5data/arrhythmia_train.arff"));
        ArffReader arff = new ArffReader(reader);
        Instances isTrainingSet = arff.getData();
        isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

        Classifier cModel = (Classifier)new FT();  
        String[] op = new String[4];
   	    op[0] = "-M";
   	    op[1] = "10";
   	    op[2] = "-F";
   	    op[3] = "1";
   	    cModel.setOptions(op);
        cModel.buildClassifier(isTrainingSet);

        BufferedReader reader2 = new BufferedReader(new FileReader("ms5_milestone5data/arrhythmia_test.arff"));
        ArffReader arff2 = new ArffReader(reader2);
        Instances isTestSet = arff2.getData();
        isTestSet.setClassIndex(isTestSet.numAttributes() - 1);

        // Step 3: Test the classifier
        //===========================================================================
        // Test the model
        for (int i = 0; i < isTestSet.numInstances(); i++) {
      	   double pred = cModel.classifyInstance(isTestSet.instance(i));
      	   System.out.println(pred);
         }
    }
}
