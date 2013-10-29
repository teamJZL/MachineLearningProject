package milestone2;
import weka.classifiers.Classifier;     // Step 2
import weka.classifiers.Evaluation;     // Step 3
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;             // Step 1
import weka.core.FastVector;            // Step 1
import weka.core.Instance;              // Step 2. fill training set w/ instance
import weka.core.Instances;             // Step 2. empty training set

// output to .txt
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

// loading .arff
// http://weka.wikispaces.com/Use+WEKA+in+your+Java+code-
import weka.core.converters.ConverterUtils.DataSource;

// reference from:
// http://ianma.wordpress.com/2010/01/16/weka-with-java-eclipse-getting-started/
// http://weka.wikispaces.com/Programmatic+Use
// http://www.mkyong.com/java/how-to-write-to-file-in-java-fileoutputstream-example/
public class Driver {

    public static void main(String[] args) throws Exception{

        // Step 1: Express the problem with features
        //======================================================================
        // Declare two numeric attributes
        Attribute Attribute1 = new Attribute("firstNumeric");
        Attribute Attribute2 = new Attribute("secondNumeric");

        // Declare a nominal attribute along with its values
        FastVector fvNominalVal = new FastVector(3);
        fvNominalVal.addElement("blue");
        fvNominalVal.addElement("gray");
        fvNominalVal.addElement("black");
        Attribute Attribute3 = new Attribute("aNominal", fvNominalVal);

        // Declare the class attribute along with its values
        FastVector fvClassVal = new FastVector(2);
        fvClassVal.addElement("positive");
        fvClassVal.addElement("negative");
        Attribute ClassAttribute = new Attribute("theClass", fvClassVal);

        // Declare the feature vector
        FastVector fvWekaAttributes = new FastVector(4);
        fvWekaAttributes.addElement(Attribute1);    
        fvWekaAttributes.addElement(Attribute2);    
        fvWekaAttributes.addElement(Attribute3);    
        fvWekaAttributes.addElement(ClassAttribute);

        // Step 2: Train a classifier
        //======================================================================
        // Create an empty training set
        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);

        // Set class index
        isTrainingSet.setClassIndex(3);

        // Create the instance
        Instance iExample = new Instance(4);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), 1.0);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), 0.5);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), "positive");

        // add the instance
        isTrainingSet.add(iExample);

        // Create a Naive Bayes classifier
        Classifier cModel = (Classifier)new NaiveBayes();   
        cModel.buildClassifier(isTrainingSet);

        // Step 3: Test the classifier
        //======================================================================
        // Test the model
        Evaluation eTest = new Evaluation(isTrainingSet);
        eTest.evaluateModel(cModel, isTrainingSet);

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

        // Step 4: Use the classifier
        //======================================================================
        // Specify that the instance belong to the training set
        // in order to inherit from the set description
        iExample.setDataset(isTrainingSet);

        // Get the likelihood of each classes 
        // fDistribution[0] is the probability of being positive
        // fDistribution[1] is the probability of being negative
        double[] fDistribution = cModel.distributionForInstance(iExample);
        System.out.println();
        System.out.println("Probability of being positive: ");
        System.out.println(fDistribution[0]);
        System.out.println("Probability of being negative: ");
        System.out.println(fDistribution[1]);
        System.out.println();

        // Output to .txt
        //======================================================================
        // http://www.mkyong.com/java/how-to-write-to-file-in-java-fileoutputstream-example/
        FileOutputStream fop = null;
        File file;
        String content = "This is the text content";

        try {

            file = new File("output.txt");
            fop = new FileOutputStream(file);

            // if file doesnt exists, then create it
            if (!file.exists()) {
                file.createNewFile();
            }

            // get the content in bytes
            byte[] contentInBytes = content.getBytes();

            fop.write(contentInBytes);
            fop.flush();
            fop.close();

            System.out.println("Output to .txt: Done");

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (fop != null) {
                    fop.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
